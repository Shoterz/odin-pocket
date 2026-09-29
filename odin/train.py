"""Resumable next-token training with checkpoint-bound provenance and telemetry."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import torch
from odin.data import sha256
from odin.model import LanguageModel, ModelConfig
from odin.tokenizer import Tokenizer


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = LanguageModel(ModelConfig(**state['config']))
    model.load_state_dict(state['model'])
    model.to(device).eval()
    from tokenizers import Tokenizer as Backend
    tokenizer = Tokenizer(Backend.from_str(state['tokenizer_json']))
    return model, tokenizer, state


def train(*, data, config, output, steps, batch_size=8, accumulation=1, lr=0.0006, device='cpu', seed=20260929, eval_every=250, stop_after=None, resume=None, threads=4):
    if steps < 1 or batch_size < 1 or accumulation < 1 or eval_every < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Training counts and learning rate must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    data, out = Path(data), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'latest.pt').exists() and not resume:
        raise ValueError('Run exists; resume it or choose a new output')
    tokenizer = Tokenizer.load(data/'tokenizer.json')
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f'Tokenizer has {tokenizer.vocab_size} tokens, model expects {config.vocab_size}')
    fingerprint = {n:sha256(data/n) for n in ('train.bin','dev.bin','tokenizer.json','manifest.json')}
    arrays = {n:np.memmap(data/f'{n}.bin', dtype=np.uint16, mode='r') for n in ('train','dev')}
    if min(len(x) for x in arrays.values()) <= config.context:
        raise ValueError('Token corpora must exceed context length')
    if max(int(x.max()) for x in arrays.values()) >= config.vocab_size:
        raise ValueError('Corpus contains token IDs outside vocabulary')
    model = LanguageModel(config).to(device)
    params = list(model.parameters())
    groups = [{'params':[p for p in params if p.dim() >= 2], 'weight_decay':0.1}, {'params':[p for p in params if p.dim() < 2], 'weight_decay':0.0}]
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9,0.95), eps=1e-8, fused=(device=='cuda'))
    recipe = dict(steps=steps, batch_size=batch_size, accumulation=accumulation, lr=lr, seed=seed)
    step, tokens, previous_seconds = 0, 0, 0.0
    run_id=str(uuid.uuid4())
    code_hashes={name:sha256(Path(__file__).parent/name) for name in ('train.py','model.py','tokenizer.py')}
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state['fingerprint'] != fingerprint or state['config'] != config.to_dict() or state['recipe'] != recipe:
            raise ValueError('Resume requires identical data, tokenizer, configuration and training recipe')
        model.load_state_dict(state['model'])
        if state['provenance'].get('source_sha256',code_hashes)!=code_hashes:
            raise ValueError('Training code changed; restore the recorded source before resuming')
        run_id=state.get('run_id','legacy-'+sha256(resume))
        optimizer.load_state_dict(state['optimizer'])
        step, tokens, previous_seconds = state['step'], state['tokens'], state['training_seconds']
        torch.set_rng_state(state['rng'])
        if device == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        metrics_path=out/'metrics.jsonl'
        if metrics_path.exists():
            summary_path=out/'summary.json'
            if not summary_path.exists():
                raise ValueError('Existing metrics lack run identity; resume to a new output directory')
            prior=json.loads(summary_path.read_text())
            if prior.get('fingerprint')!=fingerprint or prior.get('recipe')!=recipe or prior.get('config')!=config.to_dict():
                raise ValueError('Existing output belongs to another training run')
            kept,seen_steps=[],set()
            for line in metrics_path.read_text().splitlines():
                try:
                    row=json.loads(line)
                except ValueError:
                    continue
                if isinstance(row.get('step'),int) and row['step']<=step and row['step'] not in seen_steps:
                    kept.append(line)
                    seen_steps.add(row['step'])
            temporary=metrics_path.with_suffix('.jsonl.tmp')
            temporary.write_text('\n'.join(kept)+'\n')
            os.replace(temporary,metrics_path)
    hardware = torch.cuda.get_device_name() if device=='cuda' else platform.processor() or 'CPU'
    provenance = {'initialization':'random', 'pretrained_weights':False, 'distillation':False, 'seed':seed, 'hardware':hardware, 'device':device, 'precision':'bfloat16 autocast / float32 optimizer' if device=='cuda' else 'float32', 'torch':str(torch.__version__), 'python':platform.python_version(), 'source_sha256':code_hashes}
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device=='cuda' else nullcontext()
    def batch(split, generator=None):
        a = arrays[split]
        starts = torch.randint(len(a)-config.context, (batch_size,), generator=generator).tolist()
        x = torch.tensor(np.stack([a[s:s+config.context].astype(np.int64) for s in starts]), device=device)
        y = torch.tensor(np.stack([a[s+1:s+config.context+1].astype(np.int64) for s in starts]), device=device)
        return x,y
    def validate():
        model.eval()
        g = torch.Generator().manual_seed(991)
        with torch.inference_mode(), amp():
            losses = [model(*batch('dev',g))[1].item() for _ in range(8)]
        model.train()
        return sum(losses)/len(losses)
    started = time.perf_counter()
    last_time, last_tokens = started, tokens
    log = open(out/'metrics.jsonl', 'a' if resume else 'w', buffering=1)
    def record(loss=None, gradient=None):
        nonlocal last_time, last_tokens
        dev = validate()
        now = time.perf_counter()
        row = {'run_id':run_id,'step':step, 'tokens':tokens, 'train_loss':loss, 'dev_loss':dev, 'dev_perplexity':math.exp(min(dev,700)), 'training_seconds':previous_seconds+now-started, 'tokens_per_second':(tokens-last_tokens)/max(now-last_time,1e-9), 'learning_rate':optimizer.param_groups[0]['lr'], 'gradient_norm':gradient, 'peak_vram_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
        log.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
        last_time, last_tokens = now, tokens
    def save():
        state = {'schema':1,'run_id':run_id,'config':config.to_dict(),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'tokens':tokens,'recipe':recipe,'training_seconds':previous_seconds+time.perf_counter()-started,'provenance':provenance,'fingerprint':fingerprint,'tokenizer_json':(data/'tokenizer.json').read_text(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,'parameter_count':model.parameter_count}
        atomic_save(state, out/'latest.pt')
        summary = {k:v for k,v in state.items() if k not in ('model','optimizer','rng','cuda_rng','tokenizer_json')}
        temp = out/'summary.json.tmp'
        temp.write_text(json.dumps(summary,indent=2)+'\n')
        os.replace(temp,out/'summary.json')
    try:
        if not resume:
            record()
        model.train()
        end = min(steps,stop_after) if stop_after else steps
        while step < end:
            warmup = max(1, min(200, int(steps*0.02)))
            factor = (step+1)/warmup if step < warmup else 0.1+0.9*0.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = lr*factor
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(accumulation):
                x,y = batch('train')
                with amp():
                    _, loss = model(x,y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss_sum += loss.item()/accumulation
                (loss/accumulation).backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
            optimizer.step()
            step += 1
            tokens += batch_size*config.context*accumulation
            if step % eval_every == 0 or step == end:
                record(loss_sum,grad)
                save()
    except KeyboardInterrupt:
        # Saving a partial accumulation would not support exact resume. Last completed
        # checkpoint remains the recovery point; never mislabel an interrupted step.
        print('Interrupted; recover from latest completed checkpoint.', flush=True)
        raise
    finally:
        log.close()
    return out/'latest.pt'


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/prepared')
    p.add_argument('--config',default='configs/pocket.json')
    p.add_argument('--output',default='runs/pocket')
    p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=0.0006)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--resume')
    p.add_argument('--threads',type=int,default=4)
    args=vars(p.parse_args())
    args['config']=ModelConfig(**json.loads(Path(args['config']).read_text()))
    train(**args)
