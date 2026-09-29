"""Resumable local GPU experiment: three pilots, development selection, final report.

No publication or baseline replacement occurs automatically. All candidate outputs
remain in this research worktree until a human-readable quality review is complete.
"""
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
from odin.corpus import verify_refinement_inputs,make_view
from odin.data import sha256
from odin.development import development_protocol,validate_development_report,select_candidate
from odin.experiment import atomic_json,checkpoint_stage,bind_decision,recover_summary
from odin.model import ModelConfig
from odin.release import export,verify_prepared
from odin.reporting import validate_report
from odin.sampling import training_sources

TOTAL_STEPS=183106
PILOT_STEPS=15259
RECIPE={'steps':TOTAL_STEPS,'batch_size':32,'accumulation':1,'lr':0.0006,'seed':20260929,'compile_model':True,'deterministic':True}
CANDIDATES={'A':('configs/pocket.json','data/edu'),'B':('configs/pocket.json','data/mixed'),'C':('configs/pocket-deep.json','data/mixed')}
RESULTS=ROOT/'results/refinement'
PROMPTS=ROOT/'experiments/prompts.json'


def install_shutdown_handler():
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Shutdown signal {signum}')
    signal.signal(signal.SIGTERM,stop)


def status(stage,**extra):
    row={'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**extra}
    atomic_json(RESULTS/'status.json',row)
    print(json.dumps(row),flush=True)


def run(command, label):
    status(label,command=command)
    path=RESULTS/f'{label}.log'
    with path.open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if shutil.disk_usage(ROOT).free<8*1024**3:
                    raise RuntimeError('Less than 8GiB free disk; stopping before checkpoint write failure')
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'{label} failed ({child.returncode}); inspect {path}')
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=30)
            raise


def audit_corpus(base,cache):
    common=ROOT/'data/common'
    baseline=base/'data/prepared'
    sources=[]
    for path in sorted(cache.rglob('*.arrow')):
        if any(n in str(path) for n in ('hellaswag','ai2_arc','winogrande','piqa','wikitext')) and any(n in path.name for n in ('validation','test')):
            sources.append({'file':path.name,'sha256':sha256(path)})
    wiki=sorted(cache.glob('Salesforce___wikitext/wikitext-103-raw-v1/**/wikitext-train*.arrow'))
    audit=verify_refinement_inputs(baseline,sources,wiki,base/'data/raw/fineweb-edu-000.parquet',ROOT/'data/raw/TinyStoriesV2-GPT4-train.txt')
    atomic_json(RESULTS/'input-audit.json',audit)
    started=time.monotonic()
    while not (common/'manifest.json').exists():
        status('waiting-for-corpus',elapsed_seconds=time.monotonic()-started)
        if time.monotonic()-started>7200:
            raise TimeoutError('Corpus preparation did not finish within two hours')
        time.sleep(30)
    manifest=verify_prepared(common)
    for name,digest in manifest['files'].items():
        if sha256(common/name)!=digest:
            raise ValueError('Prepared source hash mismatch: '+name)
    if manifest['decontamination']['protected_files']!=sources:
        raise ValueError('Prepared holdout inventory mismatch')
    if manifest['decontamination']['additional_protected']['prompts_sha256']!=sha256(PROMPTS):
        raise ValueError('Current prompt suite was not protected during preparation')
    if manifest['sources'][-1]['sha256']!=audit['stories_sha256']:
        raise ValueError('Prepared story corpus differs from audited source')
    # The first preparation began before the stronger input audit was added.
    # Verify its complete inputs and outputs, then attach the audit before any run.
    if 'input_verification' not in manifest:
        manifest['input_verification']=audit
        atomic_json(common/'manifest.json',manifest)
    elif manifest['input_verification']!=audit:
        raise ValueError('Prepared input audit differs from current audit')
    for name,weight in [('edu',0.0),('mixed',0.2)]:
        path=ROOT/'data'/name
        if not (path/'manifest.json').exists():
            make_view(common,path,weight)
        view=json.loads((path/'manifest.json').read_text())
        if view['common_manifest_sha256']!=sha256(common/'manifest.json'):
            raise ValueError('Existing mixture derives from a different common corpus')
        expected=[{'file':'train.bin','weight':1-weight}]+([{'file':'stories-train.bin','weight':weight}] if weight else [])
        if view['train_sampling']!=expected:
            raise ValueError('Existing mixture has unexpected sampling weights')
    return common


def fingerprint(data):
    names=set(['train.bin','dev.bin','tokenizer.json','manifest.json']+[s['file'] for s in training_sources(data)])
    return {name:sha256(data/name) for name in sorted(names)}


def checked_checkpoint(path,step,config,data):
    state=torch.load(path,map_location='cpu',weights_only=True)
    code={name:sha256(ROOT/'odin'/name) for name in ('train.py','model.py','tokenizer.py','sampling.py')}
    checkpoint_stage(state,step,RECIPE,config.to_dict(),code,fingerprint(data))
    if state['parameter_count']>50_000_000 or state['provenance']['initialization']!='random':
        raise ValueError('Candidate violates model constraints')
    del state


def train_candidate(name,stop):
    config_file,data_file=CANDIDATES[name]
    config=ModelConfig(**json.loads((ROOT/config_file).read_text())).validate()
    data=ROOT/data_file
    out=ROOT/'runs/refinement'/name
    latest=out/'latest.pt'
    if latest.exists():
        code={name:sha256(ROOT/'odin'/name) for name in ('train.py','model.py','tokenizer.py','sampling.py')}
        previous=recover_summary(latest,out/'summary.json',RECIPE,config.to_dict(),code,fingerprint(data))
        checked_checkpoint(latest,previous['step'],config,data)
        if previous['step']>stop:
            raise ValueError('Checkpoint beyond requested stage; immutable pilot snapshot required')
        if previous['step']==stop:
            return latest
    command=[sys.executable,'-m','odin.train','--data',str(data),'--config',str(ROOT/config_file),'--output',str(out),'--steps',str(TOTAL_STEPS),
             '--batch-size','32','--accumulation','1','--lr','0.0006','--device','cuda','--eval-every','1000','--stop-after',str(stop),'--compile-model','--deterministic']
    if latest.exists():
        command+=['--resume',str(latest)]
    run(command,f'train-{name}-{stop}')
    checked_checkpoint(latest,stop,config,data)
    return latest


def development(checkpoint,name,common,tokens):
    out=RESULTS/f'{name}-development.json'
    if not out.exists():
        run([sys.executable,'-m','odin.development','--checkpoint',str(checkpoint),'--data',str(common),'--prompts',str(PROMPTS),'--output',str(out),'--device','cuda'],f'evaluate-{name}')
    report=json.loads(out.read_text())
    validate_development_report(report,sha256(checkpoint),tokens,development_protocol(common,PROMPTS,'cuda'),json.loads(PROMPTS.read_text()))
    return report


def write_report(base,decision,pilots,baseline,final,official):
    rows=[]
    previous=json.loads((base/'results/official.json').read_text())
    for task in ('hellaswag','arc_easy','piqa','winogrande'):
        before,after=previous['results'][task]['acc,none'],official['results'][task]['acc,none']
        rows.append(f'| {task} | {before:.2%} | {after:.2%} | {(after-before)*100:+.2f} pp |')
    sums=[json.loads((ROOT/f'runs/refinement/{name}/summary.json').read_text()) for name in CANDIDATES]
    pilotrows=[]
    for name,r in pilots.items():
        pilotrows.append(f"| {name} | {r['parameter_count']:,} | {r['domains']['web']['nll']:.4f} | {r['domains']['stories']['nll']:.4f} | {r['mean_repeated_4gram_rate']:.3f} |")
    total_seconds=sum(s['training_seconds'] for s in sums)
    total_tokens=sum(s['tokens'] for s in sums)
    compute=sum(6*s['parameter_count']*s['tokens'] for s in sums)
    content=f'''# ODIN refinement: measured results

Candidate **{decision['selected']}** was selected using development data only. Final checkpoint SHA256: `{final['checkpoint_sha256']}`.

The published v0.1.0 model remains unchanged. This report does not establish a winning submission; review all saved continuations before promoting the candidate.

## Controlled pilots

Each pilot processed {PILOT_STEPS*16384:,} tokens, with one seed and a shared tokenizer. A: expanded educational corpus, original architecture. B: same architecture, 20% public TinyStories mixture. C: same mixture, deeper/narrower architecture. The short pilot and single seed limit inference about long-run architecture quality.

| Candidate | Parameters | Web NLL | Story NLL | Repeated 4-grams |
|---|---:|---:|---:|---:|
{chr(10).join(pilotrows)}

Selection details: `results/refinement/decision.json`. Baseline and all 32 continuations per candidate are saved in `*-development.json`; no cherry-picked samples.

## Final official evaluation

| Task | Published baseline | Candidate | Difference |
|---|---:|---:|---:|
{chr(10).join(rows)}

WikiText-103 token perplexity: {previous['wikitext_103']['token_perplexity']:.3f} → {official['wikitext_103']['token_perplexity']:.3f}, with the same tokenizer and scoring protocol.

Final development web/story NLL: {final['domains']['web']['nll']:.4f} / {final['domains']['stories']['nll']:.4f}. Baseline: {baseline['domains']['web']['nll']:.4f} / {baseline['domains']['stories']['nll']:.4f}.
Final repeated 4-gram fraction: {final['mean_repeated_4gram_rate']:.3f}; baseline {baseline['mean_repeated_4gram_rate']:.3f}. This is a repetition metric, not a factuality or coherence score.

## Compute and provenance

Local RTX4070 12GB. All three runs combined: {total_tokens:,} processed tokens, {total_seconds/3600:.3f} recorded training hours, approximate dense training compute {compute:.3e} FLOPs. These include losing pilots and selected pilot once; exclude preprocessing, throughput probes, smoke tests and evaluations, which have separate logs. The published baseline's earlier training is additional historical cost.

Input verification: `results/refinement/input-audit.json`; corpus manifests: `data/common/manifest.json`, `data/edu/manifest.json`, `data/mixed/manifest.json`. TinyStories is public synthetic data generated upstream by GPT-4, explicitly allowed by the rules; no new teacher calls or pretrained weights were used. Source pin and license are in the manifests. Exact overlap filtering cannot rule out paraphrases or short overlap.

## Reproduction

Use the same pinned environment as v0.1.0. Prepare `data/common` with `python -m odin.corpus --help`, then run `python scripts/refinement.py --baseline-root /path/to/original/baseline --dataset-cache /path/to/huggingface/datasets`. The runner refuses changed recipes, data, source files, report identities or selection decisions. Cached/uncached CPU measurements and original product prompts are in `final-efficiency.json` and `final-product.json`.
'''
    (ROOT/'docs/refinement-results.md').write_text(content)
    atomic_json(RESULTS/'compute.json',{'training_seconds_all_candidates':total_seconds,'tokens_all_candidates':total_tokens,'approximate_6NT_flops':compute,'runs':sums})


def main(args):
    RESULTS.mkdir(parents=True,exist_ok=True)
    lock=(RESULTS/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base,cache=Path(args.baseline_root).resolve(),Path(args.dataset_cache).resolve()
    smoke=json.loads((RESULTS/'gpu-smoke.json').read_text())
    if len(smoke)!=2 or not all(row['passed'] for row in smoke):
        raise ValueError('Full-size GPU smoke not verified')
    code={name:sha256(ROOT/'odin'/name) for name in ('train.py','model.py','tokenizer.py','sampling.py')}
    if {row['config'] for row in smoke}!={'pocket','pocket-deep'} or any(row.get('source_sha256')!=code or not row['recipe'].get('deterministic') or not row['recipe'].get('compile_model') for row in smoke):
        raise ValueError('GPU smoke uses different source or execution settings')
    common=audit_corpus(base,cache)
    frozen={'recipe':RECIPE,'pilot_steps':PILOT_STEPS,'candidates':CANDIDATES,'config_sha256':{name:sha256(ROOT/paths[0]) for name,paths in CANDIDATES.items()},'prompt_sha256':sha256(PROMPTS),'common_manifest_sha256':sha256(common/'manifest.json'),
            'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'odin').glob('*.py'))}|{'scripts/refinement.py':sha256(__file__)}}
    bind_decision(RESULTS/'experiment-lock.json',frozen)
    original=base/'runs/pocket/submission.pt'
    baseline=development(original,'baseline',common,983040000)
    pilots={}
    for name in CANDIDATES:
        pilot=ROOT/f'runs/refinement/{name}/pilot.pt'
        config_path,data_path=CANDIDATES[name]
        if not pilot.exists():
            latest=train_candidate(name,PILOT_STEPS)
            export(latest,pilot)
        checked_checkpoint(pilot,PILOT_STEPS,ModelConfig(**json.loads((ROOT/config_path).read_text())),ROOT/data_path)
        pilots[name]=development(pilot,name,common,PILOT_STEPS*16384)
    decision={**select_candidate(pilots,baseline),'pilot_hashes':{name:r['checkpoint_sha256'] for name,r in pilots.items()},'baseline_hash':baseline['checkpoint_sha256']}
    bind_decision(RESULTS/'decision.json',decision)
    if decision['selected'] is None:
        status('completed-no-eligible-pilot',decision=decision)
        return
    name=decision['selected']
    final_path=ROOT/f'runs/refinement/{name}/submission.pt'
    if not final_path.exists():
        latest=train_candidate(name,TOTAL_STEPS)
        export(latest,final_path)
    config_path,data_path=CANDIDATES[name]
    checked_checkpoint(final_path,TOTAL_STEPS,ModelConfig(**json.loads((ROOT/config_path).read_text())),ROOT/data_path)
    final=development(final_path,'final',common,TOTAL_STEPS*16384)
    official_path=RESULTS/'final-official.json'
    wiki=list(cache.glob('Salesforce___wikitext/wikitext-103-raw-v1/**/wikitext-test.arrow'))
    if len(wiki)!=1:
        raise ValueError('Expected one pinned WikiText raw test file')
    if not official_path.exists():
        os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
        run([sys.executable,'-m','odin.evaluate','--checkpoint',str(final_path),'--output',str(official_path),'--device','cuda','--batch-size','8','--dataset-cache',str(cache),'--wiki-test',str(wiki[0])],'final-official')
    official=json.loads(official_path.read_text())
    validate_report(official,sha256(final_path))
    for script,output in [('measure_product.py','final-product.json'),('benchmark_inference.py','final-efficiency.json')]:
        path=RESULTS/output
        if not path.exists():
            run([sys.executable,str(ROOT/'scripts'/script),'--checkpoint',str(final_path),'--output',str(path)],output.removesuffix('.json'))
        if json.loads(path.read_text())['checkpoint_sha256']!=sha256(final_path):
            raise ValueError('Final measurement checkpoint mismatch')
    write_report(base,decision,pilots,baseline,final,official)
    status('completed-needs-quality-review',selected=name,checkpoint=str(final_path),report=str(ROOT/'docs/refinement-results.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-root',default=str(ROOT.parent.parent))
    p.add_argument('--dataset-cache',default='/home/ylz/.cache/huggingface/datasets')
    args=p.parse_args()
    os.chdir(ROOT)
    install_shutdown_handler()
    try:
        main(args)
    except BlockingIOError:
        # A second invocation must not overwrite the active runner's status.
        raise
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise
