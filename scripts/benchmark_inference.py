"""Paired cached/uncached generation benchmark; raw outputs retained for parity."""
import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
from odin.data import sha256
from odin.evaluate import Scorer
from odin.train import load_checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='results/efficiency.json')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    if a.repeats<1:
        p.error('--repeats must be positive')
    torch.set_num_threads(a.threads)
    checkpoint_hash=sha256(a.checkpoint)
    checkpoint_bytes=Path(a.checkpoint).stat().st_size
    model,tok,state=load_checkpoint(a.checkpoint,'cpu')
    if sha256(a.checkpoint)!=checkpoint_hash:
        raise RuntimeError('Checkpoint changed while loading; measure an immutable snapshot')
    scorer=Scorer(model,tok)
    prompts=json.loads((ROOT/'submission/frozen-product-prompts.json').read_text())['generation']
    for mode in (False,True):
        scorer.generate(prompts[0],max_new_tokens=16,temperature=0,use_cache=mode)
    rows=[]
    for repeat in range(a.repeats):
        for prompt in prompts:
            results={}
            for mode in ((False,True) if repeat%2==0 else (True,False)):
                results['cached' if mode else 'uncached']=scorer.generate(prompt,max_new_tokens=96,temperature=0.7,seed=42,use_cache=mode)
            same=results['cached']['text']==results['uncached']['text']
            rows.append({'repeat':repeat,'prompt':prompt,'identical_text':same,**results})
            print(json.dumps({'repeat':repeat,'identical_text':same,**{k:round(v['tokens_per_second'],2) for k,v in results.items()}}),flush=True)
    medians={mode:statistics.median(row[mode]['tokens_per_second'] for row in rows) for mode in ('cached','uncached')}
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    report={'checkpoint_sha256':checkpoint_hash,'device':'cpu','cpu':cpu,'threads':a.threads,'torch':str(torch.__version__),'requested_new_tokens':96,'temperature':0.7,'seed':42,'repeats':a.repeats,'median_tokens_per_second':medians,'speedup':medians['cached']/medians['uncached'],'all_outputs_identical':all(row['identical_text'] for row in rows),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'checkpoint_bytes':checkpoint_bytes,'results':rows,'limitations':'Single local host under current load. Alternating mode order, two warmups. Peak RSS is process-wide including loading and both modes; it is not isolated KV-cache memory. Context rollover recomputes the retained window for correctness.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
