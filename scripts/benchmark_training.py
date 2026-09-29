"""Small local throughput experiment; random tokens, never a submission model."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.cache/triton'))
import torch
from odin.model import LanguageModel,ModelConfig

torch.set_num_threads(4)
torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32=True
rows=[]
for compiled, batch in ((False,16),(True,16),(True,32)):
    model=LanguageModel(ModelConfig()).cuda()
    fn=torch.compile(model) if compiled else model
    opt=torch.optim.AdamW(model.parameters(),lr=0.0006,fused=True)
    x=torch.randint(16384,(batch,512),device='cuda')
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for i in range(110):
        if i==10:
            torch.cuda.synchronize()
            warmup=time.perf_counter()-started
            started=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=fn(x,x.roll(-1,1))[1]
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite benchmark loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    row={'compiled':compiled,'batch_size':batch,'steps':100,'warmup_seconds':warmup,'seconds':elapsed,'tokens_per_second':100*batch*512/elapsed,'peak_vram':torch.cuda.max_memory_allocated(),'final_random_batch_loss':loss.item()}
    print(json.dumps(row),flush=True)
    rows.append(row)
    del fn,model,opt,x,loss
    torch.cuda.empty_cache()
(ROOT/'results/refinement').mkdir(parents=True,exist_ok=True)
(ROOT/'results/refinement/training-throughput.json').write_text(json.dumps(rows,indent=2)+'\n')
