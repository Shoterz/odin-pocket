"""Explicit GPU smoke: full-sized configurations, compilation and restart parity."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.cache/triton'))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from odin.model import ModelConfig
from odin.train import train,load_checkpoint
from odin.data import sha256

data=ROOT.parent.parent/'data/prepared'
reports=[]
torch.use_deterministic_algorithms(True)
for name in ('pocket','pocket-deep'):
    config=ModelConfig(**json.loads((ROOT/f'configs/{name}.json').read_text()))
    args=dict(data=data,config=config,steps=4,batch_size=32,accumulation=1,device='cuda',eval_every=2,compile_model=True,deterministic=True)
    root=ROOT/'runs/smoke-verified'/name
    full=train(output=root/'full',**args)
    train(output=root/'resumed',stop_after=2,**args)
    resumed=train(output=root/'resumed',resume=root/'resumed/latest.pt',**args)
    a,_,sa=load_checkpoint(full)
    b,_,sb=load_checkpoint(resumed)
    differences=[(a.state_dict()[k]-v).abs().max().item() for k,v in b.state_dict().items()]
    maximum=max(differences)
    if maximum>1e-6:
        raise RuntimeError(f'GPU restart mismatch {name}: {maximum}')
    reports.append({'config':name,'parameters':a.parameter_count,'steps':sa['step'],'tokens':sa['tokens'],'max_abs_resume_difference':maximum,'passed':True,'recipe':sa['recipe'],'source_sha256':sa['provenance']['source_sha256']})
    del a,b,sa,sb
    torch.cuda.empty_cache()
out=ROOT/'results/refinement/gpu-smoke.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports),flush=True)
