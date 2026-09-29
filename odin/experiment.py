"""Small integrity helpers for the resumable refinement experiment."""
import json
from pathlib import Path


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def checkpoint_stage(state, expected_step, recipe, config, source_hashes, fingerprint):
    tokens=expected_step*recipe['batch_size']*recipe['accumulation']*config['context']
    if state['step']!=expected_step or state['tokens']!=tokens:
        raise ValueError('Checkpoint step or token count differs from experiment stage')
    if state['recipe']!=recipe or state['config']!=config or state['fingerprint']!=fingerprint:
        raise ValueError('Checkpoint recipe, configuration, or corpus differs from experiment')
    if state['provenance']['source_sha256']!=source_hashes:
        raise ValueError('Checkpoint code differs from experiment')
    return True


def bind_decision(path, decision):
    path=Path(path)
    decision=json.loads(json.dumps(decision,allow_nan=False))
    if path.exists():
        if json.loads(path.read_text())!=decision:
            raise ValueError('Existing decision differs; refusing to silently reselect')
    else:
        atomic_json(path,decision)


def recover_summary(checkpoint, summary_path, recipe, config, source_hashes, fingerprint):
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    step=state['step']
    if type(step) is not int or not 0<=step<=recipe['steps']:
        raise ValueError('Invalid checkpoint recovery step')
    checkpoint_stage(state,step,recipe,config,source_hashes,fingerprint)
    summary={k:v for k,v in state.items() if k not in ('model','optimizer','rng','cuda_rng','tokenizer_json')}
    atomic_json(summary_path,summary)
    return summary
