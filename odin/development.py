"""Frozen development-only measurements; never selects on official test scores."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import torch
from odin.data import sha256
from odin.evaluate import Scorer
from odin.sampling import TokenSampler
from odin.train import load_checkpoint


def repetition_rate(text):
    words = re.findall(r'\w+',text.casefold())
    ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    return (len(ngrams)-len(set(ngrams)))/max(1,len(ngrams))


def development_protocol(data, prompts, device, context=512, windows=256):
    data,prompts=Path(data),Path(prompts)
    suite=json.loads(prompts.read_text())
    return {'context':context,'windows_per_domain':windows,'window_seed':90210,'prompt_file_sha256':sha256(prompts),
            'tokenizer_sha256':sha256(data/'tokenizer.json'),'domain_files':{name:sha256(data/file) for name,file in [('web','dev.bin'),('stories','stories-dev.bin')]},
            'temperature':suite['temperature'],'seeds':suite['seeds'],'max_new_tokens':suite['max_new_tokens'],'top_k':40,
            'device':device,'precision':'bfloat16-autocast' if device=='cuda' else 'float32','generation_cache':True}


def validate_development_report(report, checkpoint_hash, tokens, protocol, suite):
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if report.get('checkpoint_sha256')!=checkpoint_hash or report.get('trained_tokens')!=tokens:
        raise ValueError('Development report checkpoint identity mismatch')
    if report.get('protocol')!=protocol or report.get('protocol_sha256')!=digest:
        raise ValueError('Development report protocol mismatch')
    expected=Counter((prompt,seed) for prompt in suite['generation'] for seed in suite['seeds'])
    rows=report.get('generation',[])
    if not expected or Counter((row['prompt'],row['seed']) for row in rows)!=expected:
        raise ValueError('Development report generation samples incomplete')
    rates=[repetition_rate(row['text']) for row in rows]
    if any(not math.isclose(rate,row['repeated_4gram_rate'],abs_tol=1e-12) for rate,row in zip(rates,rows)):
        raise ValueError('Development report sample repetition mismatch')
    if not math.isclose(sum(rates)/len(rates),report['mean_repeated_4gram_rate'],abs_tol=1e-12):
        raise ValueError('Development report repetition aggregate mismatch')
    for name in ('web','stories'):
        domain=report['domains'][name]
        if not math.isfinite(domain['nll']) or domain['nll']<0 or domain['windows']!=protocol['windows_per_domain'] or domain['tokens']!=protocol['context']*protocol['windows_per_domain']:
            raise ValueError('Development report domain measurement invalid')


def select_candidate(reports, baseline):
    if set(reports) != {'A','B','C'}:
        raise ValueError('Require all three pilot reports')
    protocol = baseline.get('protocol_sha256')
    if not protocol:
        raise ValueError('Missing baseline protocol')
    for report in [baseline,*reports.values()]:
        if report.get('protocol_sha256')!=protocol:
            raise ValueError('Mismatched development protocols')
        values=[report['domains'][domain]['nll'] for domain in ('web','stories')]+[report['mean_repeated_4gram_rate']]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('Invalid development measurement')
    reference_web = reports['A']['domains']['web']['nll']
    repetition_limit = baseline['mean_repeated_4gram_rate']+0.05
    rejected, eligible = {}, []
    scores={}
    for name,report in reports.items():
        web,story=report['domains']['web']['nll'],report['domains']['stories']['nll']
        scores[name]=0.8*web+0.2*story
        if web>reference_web+0.10:
            rejected[name]='web regression >0.10 nats against pilot A'
        elif report['mean_repeated_4gram_rate']>repetition_limit:
            rejected[name]='repetition regression >0.05 against published baseline'
        else:
            eligible.append(name)
    return {'selected':min(eligible,key=lambda name:(scores[name],name)) if eligible else None,
            'weighted_nll':scores,'rejected':rejected,'policy':'80% web/20% story NLL, fixed web and repetition guards; development only',
            'limitations':'One seed, early pilots, heuristic gates; not proof of architectural superiority or general intelligence.'}


@torch.inference_mode()
def evaluate_development(checkpoint,data,prompts,device='cpu',windows=256):
    torch.set_num_threads(4)
    data,prompts=Path(data),Path(prompts)
    checkpoint_hash=sha256(checkpoint)
    model,tok,state=load_checkpoint(checkpoint,device)
    if sha256(checkpoint)!=checkpoint_hash:
        raise ValueError('Checkpoint changed while loading')
    if sha256(data/'tokenizer.json')!=state['fingerprint']['tokenizer.json']:
        raise ValueError('Development tokenizer differs from checkpoint tokenizer')
    scorer=Scorer(model,tok,8)
    suite=json.loads(prompts.read_text())
    protocol=development_protocol(data,prompts,device,model.config.context,windows)
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    started=time.perf_counter()
    domains={}
    for name,file in [('web','dev.bin'),('stories','stories-dev.bin')]:
        sampler=TokenSampler([(np.memmap(data/file,dtype=np.uint16,mode='r'),1.0)],model.config.context,tok.vocab_size)
        generator=torch.Generator().manual_seed(90210)
        loss_sum=0.0
        for offset in range(0,windows,8):
            n=min(8,windows-offset)
            x,y=sampler.batch(n,generator)
            with scorer.amp():
                loss=model(x.to(device),y.to(device))[1]
            loss_sum+=float(loss)*n
        nll=loss_sum/windows
        domains[name]={'nll':nll,'perplexity':math.exp(nll),'windows':windows,'tokens':windows*model.config.context}
        print(json.dumps({'domain':name,**domains[name]}),flush=True)
    generated=[]
    for prompt in suite['generation']:
        for seed in suite['seeds']:
            result=scorer.generate(prompt,max_new_tokens=suite['max_new_tokens'],temperature=suite['temperature'],seed=seed)
            result.pop('trace',None)
            row={'prompt':prompt,'seed':seed,**result,'repeated_4gram_rate':repetition_rate(result['text'])}
            generated.append(row)
            print(json.dumps(row),flush=True)
    comparisons=[]
    for row in suite['comparison']:
        pairs=[scorer.encode_pair(row['prompt'],text) for text in row['candidates']]
        scores=[value/max(1,len(pair[1])) for (value,_),pair in zip(scorer.score_tokens(pairs),pairs)]
        predicted=max(range(len(scores)),key=lambda i:scores[i])
        comparisons.append({**row,'mean_log_likelihood':scores,'predicted':predicted,'correct':predicted==row['expected']})
    return {'schema':1,'checkpoint_sha256':checkpoint_hash,'parameter_count':model.parameter_count,'trained_tokens':state['tokens'],
            'protocol':protocol,'protocol_sha256':protocol_hash,'domains':domains,'generation':generated,'comparison':comparisons,
            'comparison_accuracy':sum(row['correct'] for row in comparisons)/len(comparisons),
            'mean_repeated_4gram_rate':sum(row['repeated_4gram_rate'] for row in generated)/len(generated),
            'mean_generated_tokens':sum(row['tokens'] for row in generated)/len(generated),
            'elapsed_seconds':time.perf_counter()-started,
            'limitations':'Local development suite, not independent validation. Fixed random windows may overlap; repetition metric does not establish coherence or truth.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--prompts',default='experiments/prompts.json')
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    result=evaluate_development(a.checkpoint,a.data,a.prompts,a.device)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
