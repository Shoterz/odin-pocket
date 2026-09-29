"""Build inspectable local release artifacts; never publish or invent missing metrics."""
import argparse
import json
import math
from pathlib import Path
import shutil
import torch
from odin.data import sha256,require_benchmark_coverage
from odin.train import atomic_save,load_checkpoint
from odin.reporting import validate_report


def verify_prepared(data):
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text())
    for name in ('train.bin','dev.bin','tokenizer.json','train.jsonl','dev.jsonl'):
        if not (data/name).exists() or sha256(data/name)!=manifest['files'].get(name):
            raise ValueError('Prepared corpus hash mismatch: '+name)
    require_benchmark_coverage(manifest['decontamination']['protected_files'])
    return manifest



def export(checkpoint,output):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key in ('optimizer','rng','cuda_rng'):
        state.pop(key,None)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(state,output)
    print(json.dumps({'weights':str(output),'sha256':sha256(output),'parameters':state['parameter_count'],'tokens':state['tokens']}),flush=True)
    return output


def verify_evidence_identity(state,data,run):
    data,run=Path(data),Path(run)
    for name,digest in state['fingerprint'].items():
        if not (data/name).is_file() or sha256(data/name)!=digest:
            raise ValueError('Evidence does not match checkpoint data: '+name)
    summary=json.loads((run/'summary.json').read_text())
    for key in ('run_id','fingerprint','config','recipe','step','tokens'):
        if summary.get(key)!=state.get(key):
            raise ValueError('Evidence run identity mismatch: '+key)
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    if not rows or any(row.get('run_id')!=state['run_id'] or not math.isfinite(row.get('dev_loss',float('nan'))) for row in rows):
        raise ValueError('Training metrics do not match checkpoint run')
    if rows[-1].get('step')!=state['step'] or rows[-1].get('tokens')!=state['tokens']:
        raise ValueError('Training metrics do not reach the checkpoint')
    if any(a['step']>=b['step'] or a['tokens']>=b['tokens'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Training metrics are not monotonic')


def package(checkpoint,report_path,run,data,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=sha256(checkpoint)
    report=json.loads(Path(report_path).read_text())
    validate_report(report,checkpoint_hash)
    model,tokenizer,state=load_checkpoint(checkpoint)
    if sha256(checkpoint)!=checkpoint_hash:
        raise ValueError('Checkpoint changed while packaging; use an immutable export')
    if state['parameter_count']!=model.parameter_count or model.parameter_count>50000000 or state['provenance']['initialization']!='random' or state['provenance']['distillation']:
        raise ValueError('Checkpoint does not satisfy Track 01 baseline')
    if tokenizer.vocab_size!=model.config.vocab_size:
        raise ValueError('Checkpoint tokenizer and vocabulary disagree')
    verify_evidence_identity(state,data,run)
    manifest=verify_prepared(data)
    protected={item['file']:item['sha256'] for item in manifest['decontamination']['protected_files']}
    if report['wikitext_103']['source_sha256']!=protected['wikitext-test.arrow']:
        raise ValueError('WikiText evaluation source differs from protected corpus')
    for src,name in ((Path(report_path),'official-results.json'),(Path(data)/'manifest.json','data-manifest.json'),(Path(run)/'metrics.jsonl','training-metrics.jsonl'),(Path(run)/'summary.json','training-summary.json')):
        shutil.copyfile(src,output/name)
    metrics=report['results']
    rows=[]
    for key in ('hellaswag','arc_easy','piqa','winogrande'):
        r=metrics[key]
        rows.append(f"| {key} | {100*r['acc,none']:.2f}% | {r['sample_len']:,} |")
    wiki=report['wikitext_103']
    content=f'''# ODIN Pocket model card

Checkpoint SHA-256: `{checkpoint_hash}`. All values below are measured, bound to this checkpoint.

Parameters: **{state['parameter_count']:,}**. Training tokens processed: **{state['tokens']:,}**.
Training hardware: **{state['provenance']['hardware']}**. Recorded training wall time: **{state['training_seconds']/3600:.3f} hours**.
Approximate dense-transformer training compute (6 × parameters × tokens; excludes evaluation and some attention overhead): **{6*state['parameter_count']*state['tokens']:.3e} FLOPs**.

## Official zero-shot evaluation

| Task | Accuracy | Examples |
|---|---:|---:|
{chr(10).join(rows)}
| WikiText-103 raw test | {wiki['token_perplexity']:.3f} token perplexity | {wiki['tokens']:,} tokens |

WikiText word perplexity: {wiki['word_perplexity']:.3f}; bits per byte: {wiki['bits_per_byte']:.4f}. Context {wiki['context']}, stride {wiki['stride']}; every target counted once. Token perplexity is not comparable across different tokenizers. Harness version {report['harness_version']}; task definitions and versions are in the JSON report. Results do not establish reasoning reliability or instruction-following ability.

## Data and intended use

FineWeb-Edu and WikiText-103 training text, with {manifest['tokens']['train']:,} unique stored training tokens and {manifest['tokens']['dev']:,} development tokens. Sampling repeats training windows; processed tokens are not unique tokens. Byte-level BPE fitted only to training text. Random initialization, no pretrained weights or distillation.

Local next-token completion, candidate-continuation scoring, and education about compact model behavior. English-focused base model. Not a factual authority or medical/legal/financial advisor. May repeat, hallucinate, and reproduce biases or fragments of training text. No safety alignment training was performed. The benchmark overlap filter misses short and paraphrased overlaps; it is not proof of zero contamination.

AI assistance: OpenAI Codex wrote and reviewed implementation, tests, interface and documentation. The model was trained locally from scratch; Codex was not used to generate training targets.
'''
    (output/'model-card.md').write_text(content)
    readiness={'local_model_and_full_evaluation':True,'public_repository':False,'hosted_demo_video':False,'devpost_team_and_submission':False,'checkpoint_sha256':checkpoint_hash}
    (output/'readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
    print(content,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('checkpoint');e.add_argument('output')
    r=sub.add_parser('package');r.add_argument('--checkpoint',required=True);r.add_argument('--report',required=True);r.add_argument('--run',default='runs/pocket');r.add_argument('--data',default='data/prepared');r.add_argument('--output',default='submission/evidence')
    a=p.parse_args()
    if a.command=='export':
        export(a.checkpoint,a.output)
    else:
        package(a.checkpoint,a.report,a.run,a.data,a.output)
