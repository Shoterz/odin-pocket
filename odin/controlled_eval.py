"""Separate domain losses and public development questions, with strict identity checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from odin.data import sha256
from odin.development import repetition_rate
from odin.evaluate import Scorer
from odin.experiment import atomic_json
from odin.sampling import TokenSampler
from odin.train import load_checkpoint


def validate_questions(a,b):
    ids=lambda rows:[r['id'] for r in rows]
    if not a or ids(a)!=ids(b) or len(set(ids(a)))!=len(a):
        raise ValueError('Question coverage or order differs')
    if any(type(r['correct']) is not bool for r in [*a,*b]):
        raise ValueError('Invalid question score')


def paired_interval(control,candidate):
    validate_questions(control,candidate)
    delta=np.array([int(b['correct'])-int(a['correct']) for a,b in zip(control,candidate)])
    rng=np.random.default_rng(20260930)
    bootstrap=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {'difference':float(delta.mean()),'lower':float(np.quantile(bootstrap,.025)),
            'upper':float(np.quantile(bootstrap,.975)),'n':len(delta)}


def select(reports,reference):
    base=reports[reference]
    if base['partition']!='screen': raise ValueError('Selection requires screening partition')
    rejected,eligible={},[]
    for name,r in reports.items():
        for key in ('protocol_sha256','partition','trained_tokens'):
            if r[key]!=base[key]: raise ValueError('Mismatched '+key)
        validate_questions(base['questions'],r['questions'])
        for domain in ('fineweb','wiki','stories'):
            loss=r['domains'][domain]['nll']
            if not math.isfinite(loss) or loss<0: raise ValueError('Invalid domain loss')
        accuracy=sum(x['correct'] for x in r['questions'])/len(r['questions'])
        baseline=sum(x['correct'] for x in base['questions'])/len(base['questions'])
        if any(r['domains'][d]['nll']>base['domains'][d]['nll']+.05 for d in ('fineweb','wiki')):
            rejected[name]='General-domain NLL regression >0.05'
        elif accuracy<baseline-.03:
            rejected[name]='Comprehension regression >3 percentage points'
        else: eligible.append(name)
    def rank(name):
        r=reports[name]
        return (-sum(x['correct'] for x in r['questions']),
                sum(r['domains'][d]['nll'] for d in ('fineweb','wiki'))/2,
                name!=reference,name)
    winner=min(eligible,key=rank)
    return {'selected':winner,'reference':reference,'rejected':rejected,
            'policy':'ARC token-mean likelihood accuracy first, mean FineWeb/WikiText NLL tie-break; per-domain +0.05 and accuracy -0.03 guards. Stories diagnostic only.',
            'paired_vs_reference':{name:paired_interval(base['questions'],r['questions']) for name,r in reports.items()},
            'limitations':'One screening seed; heuristic nonregression thresholds, not a significance claim.'}


def protocol(data,prompts,partition,device,windows=256,generation=False):
    data=Path(data)
    return {'partition':partition,'device':device,'windows':windows,'context':512,
            'window_seed':48117,'choice_score':'mean continuation token log likelihood',
            'generation':generation,'prompts_sha256':sha256(prompts),
            'data_manifest_sha256':sha256(data/'manifest.json'),'evaluation_sha256':sha256(__file__),
            'source_sha256':{name:sha256(Path(__file__).parent/name) for name in ('evaluate.py','inference.py','model.py','sampling.py','tokenizer.py','train.py')},
            'domain_sha256':{n:sha256(data/f'{n}-dev.bin') for n in ('fineweb','wiki','stories')},
            'questions_sha256':sha256(data/'questions.json'),'tokenizer_sha256':sha256(data/'tokenizer.json')}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def validate_report(report,checkpoint,data,prompts,partition,device,windows=256,generation=False):
    expected=protocol(data,prompts,partition,device,windows,generation)
    if report['checkpoint_sha256']!=sha256(checkpoint) or report['protocol']!=expected or report['protocol_sha256']!=digest(expected):
        raise ValueError('Stale evaluation identity')
    questions=[r for r in json.loads((Path(data)/'questions.json').read_text()) if r['partition']==partition]
    expected_ids=[r['id'] for r in questions]
    if [r['id'] for r in report['questions']]!=expected_ids:
        raise ValueError('Incomplete evaluation questions')
    for r,q in zip(report['questions'],questions):
        if r['expected']!=q['expected'] or len(r['scores'])!=len(q['choices']) or not all(math.isfinite(v) for v in r['scores']):
            raise ValueError('Invalid question measurements')
        predicted=max(range(len(r['scores'])),key=lambda i:r['scores'][i])
        if r['predicted']!=predicted or r['correct']!=(predicted==q['expected']):
            raise ValueError('Question score mismatch')
    for r in report['domains'].values():
        if not math.isfinite(r['nll']) or r['nll']<0 or r['tokens']!=windows*512:
            raise ValueError('Invalid domain evaluation')
    if set(report['domains'])!={'fineweb','wiki','stories'}: raise ValueError('Missing domain')
    if generation:
        suite=json.loads(Path(prompts).read_text())
        if [(r['prompt'],r['seed']) for r in report['generation']]!=[(p,s) for p in suite['generation'] for s in suite['seeds']]:
            raise ValueError('Incomplete generation')
    return True


@torch.inference_mode()
def evaluate(checkpoint,data,prompts,partition='screen',device='cpu',windows=256,generation=False):
    if partition not in ('screen','confirmation'): raise ValueError('Invalid partition')
    torch.set_num_threads(4)
    checkpoint_hash=sha256(checkpoint)
    model,tok,state=load_checkpoint(checkpoint,device)
    data=Path(data)
    if sha256(checkpoint)!=checkpoint_hash or sha256(data/'tokenizer.json')!=state['fingerprint']['tokenizer.json']:
        raise ValueError('Checkpoint or tokenizer identity mismatch')
    if model.config.context!=512: raise ValueError('Frozen evaluation requires context 512')
    p=protocol(data,prompts,partition,device,windows,generation)
    scorer=Scorer(model,tok,8)
    domains={}
    for name in ('fineweb','wiki','stories'):
        sampler=TokenSampler([(np.memmap(data/f'{name}-dev.bin',dtype=np.uint16,mode='r'),1.)],512,tok.vocab_size)
        rng=torch.Generator().manual_seed(p['window_seed'])
        loss=0.
        for start in range(0,windows,8):
            n=min(8,windows-start)
            x,y=sampler.batch(n,rng)
            with scorer.amp(): value=model(x.to(device),y.to(device))[1]
            loss+=float(value)*n
        domains[name]={'nll':loss/windows,'tokens':windows*512}
    questions=[]
    for r in json.loads((data/'questions.json').read_text()):
        if r['partition']!=partition: continue
        pairs=[scorer.encode_pair(r['prompt'],c) for c in r['choices']]
        raw=[s[0] for s in scorer.score_tokens(pairs)]
        scores=[s/max(1,len(pair[1])) for s,pair in zip(raw,pairs)]
        predicted=int(np.argmax(scores))
        questions.append({'id':r['id'],'expected':r['expected'],'scores':scores,'raw_scores':raw,
                          'predicted':predicted,'correct':predicted==r['expected']})
    generated=[]
    if generation:
        suite=json.loads(Path(prompts).read_text())
        for prompt in suite['generation']:
            for seed in suite['seeds']:
                r=scorer.generate(prompt,max_new_tokens=suite['max_new_tokens'],temperature=suite['temperature'],seed=seed)
                r.pop('trace',None)
                generated.append({'prompt':prompt,'seed':seed,**r,'repetition_rate':repetition_rate(r['text'])})
    result={'checkpoint_sha256':checkpoint_hash,'trained_tokens':state['tokens'],'parameter_count':model.parameter_count,
            'protocol':p,'protocol_sha256':digest(p),'partition':partition,'domains':domains,
            'questions':questions,'accuracy':sum(r['correct'] for r in questions)/len(questions),'generation':generated}
    validate_report(result,checkpoint,data,prompts,partition,device,windows,generation)
    return result


def blind_review(reports,output):
    # Mapping is separate so a reviewer can judge text before learning model labels.
    output=Path(output)
    rng=random.Random(3102026)
    names=list(reports)
    keys=[[(r['prompt'],r['seed']) for r in reports[n]['generation']] for n in names]
    if not keys or not keys[0] or any(k!=keys[0] for k in keys): raise ValueError('Generation coverage differs')
    items=[]
    mapping={}
    for i,(prompt,seed) in enumerate(keys[0]):
        order=names.copy();rng.shuffle(order)
        for j,name in enumerate(order):
            label=f'item-{i+1:02d}-{j+1}'
            mapping[label]={'candidate':name,'checkpoint_sha256':reports[name]['checkpoint_sha256']}
            items.append({'id':label,'prompt':prompt,'text':reports[name]['generation'][i]['text'],
                          'relevance':None,'entity_consistency':None,'causal_factual_consistency':None,'notes':''})
    review_path=output/'blind-review.json'
    if review_path.exists():
        prior=json.loads(review_path.read_text())
        identity=lambda rows:[{k:r[k] for k in ('id','prompt','text')} for r in rows]
        if identity(prior['items'])!=identity(items):
            raise ValueError('Existing blind review belongs to different generations')
    else:
        atomic_json(review_path,{'rubric':'Each applicable dimension: 0 clear failure, 1 mixed/unclear, 2 coherent and correct; use null for not applicable. Judge independently before reading mapping. Locally authored prompts, not independent validation.','items':items})
    atomic_json(output/'blind-review-mapping.json',mapping)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',default='data/controlled/common')
    p.add_argument('--prompts',default='experiments/prompts.json')
    p.add_argument('--partition',choices=['screen','confirmation'],default='screen')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--generation',action='store_true')
    p.add_argument('--output',required=True)
    args=vars(p.parse_args());output=args.pop('output')
    result=evaluate(**args)
    atomic_json(output,result)
    print(json.dumps({k:result[k] for k in ('checkpoint_sha256','trained_tokens','domains','accuracy')}),flush=True)
