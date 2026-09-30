"""Bounded sequential GPU comparisons with frozen inputs and matched confirmation."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.cache/triton'))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from odin.controlled_data import make_mixture,prepare
from odin.controlled_eval import select,validate_report,paired_interval,blind_review
from odin.data import sha256
from odin.experiment import atomic_json,bind_decision
from odin.model import ModelConfig
from odin.release import export
from odin.sampling import training_sources
from odin.controlled_train import charged_tokens

RESULTS=ROOT/'results/controlled'
COMMON=ROOT/'data/controlled/common'
PROMPTS=ROOT/'experiments/prompts.json'
CONFIG=ROOT/'configs/pocket.json'
SCREEN_STEPS=7813
CONFIRM_STEPS=61036
MAX_TOKENS=(6*SCREEN_STEPS+2*CONFIRM_STEPS)*16384
MIXTURES={'control':{'fineweb':.6,'wiki':.4,'stories':0.},
          'stories10':{'fineweb':.5,'wiki':.4,'stories':.1},
          'stories20':{'fineweb':.4,'wiki':.4,'stories':.2}}
OPTIMIZERS={'lr6-w200':(.0006,200),'lr6-w1000':(.0006,1000),
            'lr12-w200':(.0012,200),'lr12-w1000':(.0012,1000)}


def code_hashes():
    return {n:sha256(ROOT/'odin'/n) for n in ('controlled_train.py','train.py','model.py','tokenizer.py','sampling.py')}


def fingerprint(data):
    names={'train.bin','dev.bin','tokenizer.json','manifest.json'}|{s['file'] for s in training_sources(data)}
    return {n:sha256(data/n) for n in sorted(names)}


def validate_state(state,recipe,config,files,code,source_names):
    if state['recipe']!=recipe or state['config']!=config or state['fingerprint']!=files or state['provenance']['source_sha256']!=code:
        raise ValueError('Checkpoint identity differs from experiment')
    if type(state['step']) is not int or not 0<=state['step']<=recipe['steps']:
        raise ValueError('Checkpoint step outside planned budget')
    expected=state['step']*recipe['batch_size']*recipe['accumulation']*config['context']
    counts=state['source_tokens']
    if state['tokens']!=expected or set(counts)!=set(source_names) or any(type(n) is not int or n<0 for n in counts.values()) or sum(counts.values())!=expected:
        raise ValueError('Checkpoint source token accounting mismatch')
    return True


def status(stage,**kwargs):
    r={'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**kwargs}
    atomic_json(RESULTS/'status.json',r)
    print(json.dumps(r),flush=True)


def run(command,label):
    status(label,command=command)
    attempt=RESULTS/'attempts'/f'{time.time_ns()}-{label}.json'
    started=time.monotonic()
    record={'label':label,'started_at':datetime.now(timezone.utc).isoformat(),'command':command,'completed':False}
    atomic_json(attempt,record)
    with (RESULTS/f'{label}.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        try:
            while child.poll() is None:
                if shutil.disk_usage(ROOT).free<8*1024**3: raise RuntimeError('Less than 8GiB free disk')
                time.sleep(10)
            if child.returncode: raise RuntimeError(f'{label} failed; inspect its log')
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
            raise
        finally:
            atomic_json(attempt,{**record,'completed':child.returncode==0,'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started})


def recipe(steps,lr,warmup,seed):
    return {'steps':steps,'batch_size':32,'accumulation':1,'lr':lr,'seed':seed,
            'compile_model':True,'deterministic':True,'warmup_steps':warmup}


def train_run(name,mixture,r):
    out=ROOT/'runs/controlled'/name
    data=ROOT/'data/controlled'/mixture
    config=json.loads(CONFIG.read_text())
    files=fingerprint(data);code=code_hashes();names=[s['file'] for s in training_sources(data)]
    latest=out/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=True)
        validate_state(state,r,config,files,code,names)
        atomic_json(out/'summary.json',{k:v for k,v in state.items() if k not in ('model','optimizer','rng','cuda_rng','tokenizer_json')})
        complete=state['step']==r['steps']
        del state
    else: complete=False
    if not complete:
        retained=json.loads((out/'summary.json').read_text())['tokens'] if latest.exists() else 0
        consumed=sum(charged_tokens(p) for p in (ROOT/'runs/controlled').glob('*/work.jsonl'))
        if consumed+r['steps']*16384-retained>MAX_TOKENS:
            raise RuntimeError('Actual charged work would exceed budget, including replayed steps')
        cmd=[sys.executable,'-m','odin.controlled_train','--data',str(data),'--config',str(CONFIG),
             '--output',str(out),'--steps',str(r['steps']),'--batch-size','32','--accumulation','1',
             '--lr',str(r['lr']),'--warmup-steps',str(r['warmup_steps']),'--seed',str(r['seed']),
             '--device','cuda','--eval-every','1000','--compile-model','--deterministic']
        if latest.exists(): cmd+=['--resume',str(latest)]
        run(cmd,'train-'+name)
    state=torch.load(latest,map_location='cpu',weights_only=True)
    validate_state(state,r,config,files,code,names)
    if state['step']!=r['steps']: raise ValueError('Training did not reach planned budget')
    snapshot=out/'submission.pt'
    if not snapshot.exists(): export(latest,snapshot)
    saved=torch.load(snapshot,map_location='cpu',weights_only=True)
    validate_state(saved,r,config,files,code,names)
    if saved['step']!=state['step'] or saved['run_id']!=state['run_id'] or any(not torch.equal(v,saved['model'][k]) for k,v in state['model'].items()):
        raise ValueError('Export differs from completed checkpoint')
    return snapshot


def evaluation(checkpoint,name,partition='screen',generation=False):
    target=RESULTS/f'{name}.json'
    if not target.exists():
        cmd=[sys.executable,'-m','odin.controlled_eval','--checkpoint',str(checkpoint),'--data',str(COMMON),
             '--prompts',str(PROMPTS),'--partition',partition,'--device','cuda','--output',str(target)]
        if generation: cmd+=['--generation']
        run(cmd,'evaluate-'+name)
    r=json.loads(target.read_text())
    validate_report(r,checkpoint,COMMON,PROMPTS,partition,'cuda',generation=generation)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if r['trained_tokens']!=state['tokens'] or r['parameter_count']!=state['parameter_count']:
        raise ValueError('Evaluation checkpoint metadata differs')
    return r


def freeze():
    manifest=json.loads((COMMON/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha256(COMMON/name)!=digest: raise ValueError('Frozen corpus changed: '+name)
    for name,weights in MIXTURES.items():
        make_mixture(COMMON,ROOT/'data/controlled'/name,weights)
    expected={'screen_steps':SCREEN_STEPS,'confirmation_steps':CONFIRM_STEPS,'maximum_training_tokens':MAX_TOKENS,
              'mixtures':MIXTURES,'optimizers':OPTIMIZERS,'screen_seed':20260930,'confirmation_seed':20261001,
              'common_manifest_sha256':sha256(COMMON/'manifest.json'),'prompt_sha256':sha256(PROMPTS),
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'odin').glob('*.py'))},
              'runner_sha256':sha256(__file__),'config_sha256':sha256(CONFIG),
              'plan_sha256':sha256(ROOT/'docs/superpowers/plans/2026-09-30-controlled-refinement.md'),
              'mixture_manifest_sha256':{n:sha256(ROOT/'data/controlled'/n/'manifest.json') for n in MIXTURES}}
    bind_decision(RESULTS/'experiment-lock.json',expected)


def smoke():
    from odin.controlled_train import train
    from odin.train import load_checkpoint
    config=ModelConfig(**json.loads(CONFIG.read_text()))
    root=ROOT/'runs/controlled-smoke'/sha256(ROOT/'odin/controlled_train.py')[:12]
    args=dict(data=ROOT/'data/controlled/stories10',config=config,steps=8,batch_size=32,accumulation=1,
              device='cuda',eval_every=4,compile_model=True,deterministic=True,warmup_steps=2)
    full=train(output=root/'full',**args)
    train(output=root/'part',stop_after=4,**args)
    resumed=train(output=root/'part',resume=root/'part/latest.pt',**args)
    a,_,sa=load_checkpoint(full);b,_,sb=load_checkpoint(resumed)
    diff=max((v-b.state_dict()[k]).abs().max().item() for k,v in a.state_dict().items())
    if diff!=0 or sa['source_tokens']!=sb['source_tokens'] or sum(sa['source_tokens'].values())!=sa['tokens']:
        raise ValueError('GPU resume or accounting parity failed')
    atomic_json(RESULTS/'gpu-smoke.json',{'passed':True,'max_abs_resume_difference':diff,
                'source_tokens':sa['source_tokens'],'source_sha256':code_hashes(),'parameters':a.parameter_count})


def compute_report():
    summaries=[json.loads(p.read_text()) for p in sorted((ROOT/'runs/controlled').glob('*/summary.json'))]
    attempts=[json.loads(p.read_text()) for p in sorted((RESULTS/'attempts').glob('*-train-*.json'))]
    charged=sum(charged_tokens(p) for p in (ROOT/'runs/controlled').glob('*/work.jsonl'))
    report={'runs':summaries,'tokens':sum(s['tokens'] for s in summaries),'charged_tokens_including_replays':charged,
            'training_attempt_elapsed_seconds':sum(a.get('elapsed_seconds',0) for a in attempts),
            'attempts_missing_end_record':[a for a in attempts if 'elapsed_seconds' not in a],
            'training_seconds':sum(s['training_seconds'] for s in summaries),'maximum_training_tokens':MAX_TOKENS,
            'limitations':'Charged tokens include reserved partial and replayed steps. Attempt time includes startup/compile/checkpoint work; missing end records have unknown duration. Retained training_seconds excludes rollback work. Preparation, evaluation and smoke are separate.'}
    if charged>MAX_TOKENS: raise ValueError('Training exceeded planned budget')
    atomic_json(RESULTS/'compute.json',report)
    return report


def main(args):
    RESULTS.mkdir(parents=True,exist_ok=True)
    lock=(RESULTS/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(ROOT/'data/common',COMMON,Path.home()/'.cache/huggingface/datasets',PROMPTS)
    for name,weights in MIXTURES.items(): make_mixture(COMMON,ROOT/'data/controlled'/name,weights)
    if args.smoke:
        smoke();return
    verified=json.loads((RESULTS/'gpu-smoke.json').read_text())
    if not verified['passed'] or verified['source_sha256']!=code_hashes(): raise ValueError('Missing matching GPU smoke')
    freeze()
    # Historical references never enter recipe selection or confirmation statistics.
    for name,path in [('published',ROOT.parent.parent/'runs/pocket/submission.pt'),('previous-B',ROOT/'runs/refinement/B/submission.pt')]:
        evaluation(path,'reference-'+name)
    optimizer_reports={}
    for name,(lr,warmup) in OPTIMIZERS.items():
        freeze()
        ckpt=train_run(name,'control',recipe(SCREEN_STEPS,lr,warmup,20260930))
        optimizer_reports[name]=evaluation(ckpt,'screen-'+name)
        compute_report()
    decision=select(optimizer_reports,'lr6-w200')
    decision['checkpoint_hashes']={n:r['checkpoint_sha256'] for n,r in optimizer_reports.items()}
    bind_decision(RESULTS/'optimizer-decision.json',decision)
    best=decision['selected']; lr,warmup=OPTIMIZERS[best]
    mixture_reports={'control':optimizer_reports[best]}
    for name in ('stories10','stories20'):
        freeze()
        ckpt=train_run(name,name,recipe(SCREEN_STEPS,lr,warmup,20260930))
        mixture_reports[name]=evaluation(ckpt,'screen-'+name)
        compute_report()
    mix=select(mixture_reports,'control')
    mix['checkpoint_hashes']={n:r['checkpoint_sha256'] for n,r in mixture_reports.items()}
    bind_decision(RESULTS/'mixture-decision.json',mix)
    if mix['selected']=='control':
        status('completed-no-mixture-winner',optimizer=best,decision=mix)
        return
    candidate=mix['selected']
    # Both train before any confirmation evaluation is exposed.
    checkpoints={}
    for name in ('control',candidate):
        freeze()
        checkpoints[name]=train_run('confirm-'+name,name,recipe(CONFIRM_STEPS,lr,warmup,20261001))
        compute_report()
    reports={name:evaluation(path,'confirm-'+name,'confirmation',True) for name,path in checkpoints.items()}
    interval=paired_interval(reports['control']['questions'],reports[candidate]['questions'])
    domain_ok=all(reports[candidate]['domains'][d]['nll']<=reports['control']['domains'][d]['nll']+.05 for d in ('fineweb','wiki'))
    atomic_json(RESULTS/'confirmation.json',{'candidate':candidate,'paired_accuracy_interval':interval,
                'general_domain_guard_passed':domain_ok,'statistical_gain_demonstrated':interval['lower']>0 and domain_ok,
                'checkpoint_hashes':{n:r['checkpoint_sha256'] for n,r in reports.items()},
                'release_decision':'Pending blinded quality review; no automatic publication.'})
    blind_review(reports,RESULTS)
    status('completed-needs-blind-review',candidate=candidate,confirmation_interval=interval,domain_guard=domain_ok)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    def stop(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    os.chdir(ROOT)
    try: main(args)
    except BlockingIOError: raise
    except BaseException as exc:
        status('failed',error=repr(exc));raise
