"""Frozen qualitative evaluation and measured CPU inference, separate from benchmarks."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
from odin.data import sha256
from odin.evaluate import Scorer
from odin.train import load_checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='results/product.json')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    started=time.perf_counter()
    checkpoint_hash=sha256(a.checkpoint)
    checkpoint_bytes=Path(a.checkpoint).stat().st_size
    model,tok,state=load_checkpoint(a.checkpoint,'cpu')
    if sha256(a.checkpoint)!=checkpoint_hash:
        raise RuntimeError('Checkpoint changed while loading; measure an immutable snapshot')
    load_seconds=time.perf_counter()-started
    scorer=Scorer(model,tok,4)
    prompts=json.loads((ROOT/'submission/frozen-product-prompts.json').read_text())
    scorer.generate('The sky is',max_new_tokens=4,temperature=0)
    generated=[]
    for prompt in prompts['generation']:
        result=scorer.generate(prompt,max_new_tokens=96,temperature=0.7,seed=42)
        generated.append({'prompt':prompt,**result})
        print(json.dumps({'prompt':prompt,'text':result['text'],'tokens_per_second':result['tokens_per_second']}),flush=True)
    comparison=[]
    for row in prompts['comparison']:
        pairs=[scorer.encode_pair(row['prompt'],' '+c) for c in row['candidates']]
        scores=[ll/max(1,len(pair[1])) for (ll,_),pair in zip(scorer.score_tokens(pairs),pairs)]
        predicted=max(range(len(scores)),key=lambda i:scores[i])
        comparison.append({**row,'mean_log_likelihood':scores,'predicted':predicted,'correct':predicted==row['expected']})
    report={'schema':1,'checkpoint_sha256':checkpoint_hash,'parameter_count':model.parameter_count,'trained_tokens':state['tokens'],'device':'cpu','cpu':platform.processor(),'threads':a.threads,'load_seconds':load_seconds,'weight_file_bytes':checkpoint_bytes,'prompt_set_sha256':sha256(ROOT/'submission/frozen-product-prompts.json'),'generation':generated,'comparison':comparison,'comparison_accuracy':sum(x['correct'] for x in comparison)/len(comparison),'limitations':'Six locally authored comparisons and six fixed prompts; qualitative demonstration only, not independent validation or an official benchmark. CPU latency is host- and load-dependent.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
