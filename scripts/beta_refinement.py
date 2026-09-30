"""One bounded beta2 candidate, frozen control, and automatic development evaluation."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT/'.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT/'.cache/triton'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from odin import beta_eval
from odin.controlled_eval import blind_review, digest
from odin.controlled_train import charged_tokens
from odin.data import sha256
from odin.experiment import atomic_json, bind_decision
from odin.release import export
from scripts.controlled_refinement import fingerprint, validate_state, code_hashes as old_code_hashes

RESULTS = ROOT/'results/beta-refinement'
RUN = ROOT/'runs/beta-refinement/candidate'
CONTROL = ROOT/'runs/controlled/warmup-confirm-1000/submission.pt'
DATA = ROOT/'data/controlled/control'
COMMON = ROOT/'data/controlled/common'
PROMPTS = ROOT/'experiments/beta-prompts.json'
CONFIG = ROOT/'configs/pocket.json'
PLAN = ROOT/'docs/superpowers/plans/2026-09-30-beta2-refinement.md'
CAP = 1000013824
CUTOFF = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc).timestamp()
RECIPE = dict(steps=61036, batch_size=32, accumulation=1, lr=.0006,
              seed=20261001, compile_model=True, deterministic=True, warmup_steps=1000, beta2=.999)


def require_budget(charged, retained, cap=CAP):
    if any(type(n) is not int or n < 0 for n in (charged, retained, cap)) or retained > cap or charged < retained:
        raise ValueError('Invalid charged/retained budget')
    if charged + cap - retained > cap:
        raise ValueError('Uncheckpointed/replayed work prevents finishing within charged cap; no retry')


def require_resources(now, cutoff, free_bytes):
    if now >= cutoff:
        raise RuntimeError('Experiment deadline reached')
    if free_bytes < 8 * 1024**3:
        raise RuntimeError('Insufficient free disk')


@contextmanager
def writer_lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_verified(evidence, protocol):
    if evidence.get('passed') is not True or evidence.get('protocol_sha256') != digest(protocol):
        raise ValueError('Missing successful verification bound to current protocol')


def train_code():
    return {n:sha256(ROOT/'odin'/n) for n in ('beta_train.py','controlled_train.py','train.py','model.py','tokenizer.py','sampling.py')}


def frozen_protocol():
    manifest = json.loads((COMMON/'manifest.json').read_text())
    for name, value in manifest['files'].items():
        if sha256(COMMON/name) != value:
            raise ValueError('Frozen corpus changed: '+name)
    old = torch.load(CONTROL, map_location='cpu', weights_only=True)
    old_recipe = {k:v for k,v in RECIPE.items() if k != 'beta2'}
    validate_state(old, old_recipe, json.loads(CONFIG.read_text()), fingerprint(DATA), old_code_hashes(), ['fineweb.bin','wiki.bin'])
    if old['tokens'] != CAP or old['parameter_count'] != 49295872:
        raise ValueError('Wrong historical control budget or size')
    import lm_eval.api.task
    harness_path = Path(lm_eval.api.task.__file__)
    arc_path = harness_path.parents[1]/'tasks/arc/arc_easy.yaml'
    protocol = {'schema':1, 'recipe':RECIPE, 'candidate_charged_cap':CAP,
                'verification_charged_cap':3*32*16384, 'cutoff_utc':'2026-10-01T04:00:00+00:00',
                'child_training_timeout_seconds':14400, 'data':fingerprint(DATA),
                'config':json.loads(CONFIG.read_text()), 'control_sha256':sha256(CONTROL),
                'training_source_sha256':train_code(),
                'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
                                 (Path(__file__), PLAN, PROMPTS, ROOT/'odin/beta_eval.py', ROOT/'odin/release.py',
                                  ROOT/'odin/experiment.py', ROOT/'scripts/controlled_refinement.py',
                                  ROOT/'scripts/owned_child.py')},
                'evaluation':beta_eval.protocol(COMMON, PROMPTS, 'cuda'),
                'harness':{'version':importlib.metadata.version('lm_eval'), 'task_sha256':sha256(harness_path), 'arc_yaml_sha256':sha256(arc_path)},
                'gates':{'fineweb_ppl_ratio_max':.97, 'wiki_ppl_ratio_max':1.02,
                         'raw_accuracy_delta_min_exclusive':0, 'acc_norm_delta_min':-.01,
                         'additional_coherent_outputs_min':4, 'quality_review_required':True}}
    bind_decision(RESULTS/'experiment-lock.json', protocol)
    return protocol


def status(stage, **extra):
    value = {'stage':stage, 'updated_at':datetime.now(timezone.utc).isoformat(), 'pid':os.getpid(), **extra}
    atomic_json(RESULTS/'status.json', value)
    print(json.dumps(value), flush=True)


def run_child(command, label, timeout_seconds=3600):
    cutoff = min(CUTOFF, time.time()+timeout_seconds)
    require_resources(time.time(), cutoff, shutil.disk_usage(ROOT).free)
    record = {'label':label, 'command':command, 'started_at':datetime.now(timezone.utc).isoformat()}
    attempt = RESULTS/'attempts'/f'{time.time_ns()}-{label}.json'
    atomic_json(attempt, {**record, 'completed':False})
    began = time.monotonic()
    with (RESULTS/f'{label}.log').open('a') as log:
        owned = [sys.executable, str(ROOT/'scripts/owned_child.py'), str(os.getpid()), *command]
        child = subprocess.Popen(owned, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            status(label, child_pid=child.pid, command=command)
            while child.poll() is None:
                require_resources(time.time(), cutoff, shutil.disk_usage(ROOT).free)
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'{label} failed, code {child.returncode}; inspect log')
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
            atomic_json(attempt, {**record, 'completed':child.returncode == 0,
                                  'returncode':child.returncode, 'elapsed_seconds':time.monotonic()-began})


def command(module, output, beta2=None, smoke=False):
    args = [sys.executable, '-m', module, '--data',str(DATA), '--config',str(CONFIG),
            '--output',str(output), '--steps','61036', '--batch-size','32', '--accumulation','1',
            '--lr','.0006', '--warmup-steps','1000', '--seed','20261001',
            '--device','cuda', '--eval-every', '32' if smoke else '1000', '--compile-model','--deterministic']
    if beta2 is not None:
        args += ['--beta2',str(beta2)]
    if smoke:
        args += ['--stop-after','32']
    return args


def local_arc_task():
    """Use real installed ARC request/metric code with a local fixture, never download."""
    import yaml
    from datasets import Dataset
    import lm_eval.api.task
    from lm_eval.api.task import ConfigurableTask
    class LocalARC(ConfigurableTask):
        def download(self, *args, **kwargs):
            doc = {'question':'fixture', 'choices':{'label':['A','B'],'text':['a','b']}, 'answerKey':'A'}
            ds = Dataset.from_list([doc])
            self.dataset = {'train':ds, 'validation':ds, 'test':ds}
    path = Path(lm_eval.api.task.__file__).parents[1]/'tasks/arc/arc_easy.yaml'
    return LocalARC(config=yaml.safe_load(path.read_text()))


def verify_scoring():
    from odin.train import load_checkpoint
    from odin.evaluate import Scorer, harness_adapter
    task = local_arc_task()
    questions = json.loads((COMMON/'questions.json').read_text())
    saved = json.loads((ROOT/'results/warmup-confirmation/confirmation-long.json').read_text())
    if saved['checkpoint_sha256'] != sha256(CONTROL):
        raise ValueError('Historical scoring checkpoint mismatch')
    old = {r['id']:r for r in saved['questions']}
    model, tok, _ = load_checkpoint(CONTROL, 'cpu')
    torch.set_num_threads(4)
    scorer = Scorer(model, tok, 8)
    adapter = harness_adapter(scorer)
    compared = 0; live = 0
    for i, q in enumerate(questions):
        labels = [str(j) for j in range(len(q['choices']))]
        doc = {'question':q['prompt'].removeprefix('Question: ').removesuffix('\nAnswer:'),
               'choices':{'label':labels, 'text':[c[1:] for c in q['choices']]}, 'answerKey':labels[q['expected']]}
        requests = task.construct_requests(doc, task.doc_to_text(doc), metadata=('arc_easy',i,1))
        expected = [(q['prompt'], c) for c in q['choices']]
        if [r.args for r in requests] != expected:
            raise ValueError('Harness prompt/answer boundary mismatch')
        if q['id'] in old:
            r = old[q['id']]
            actual = task.process_results(doc, [(s,False) for s in r['raw_scores']])
            mapped = beta_eval.standard_question(r,q)
            if (actual['acc'],actual['acc_norm']) != (mapped['correct'],mapped['correct_norm']):
                raise ValueError('Harness metric mismatch')
            compared += 1
        if i < 4:
            a = adapter.loglikelihood(requests)
            b = scorer.score_tokens([scorer.encode_pair(*p) for p in expected])
            if a != b:
                raise ValueError('Harness adapter likelihood mismatch')
            live += len(a)
    return {'request_documents':len(questions), 'saved_metric_documents':compared,
            'live_cpu_likelihood_pairs':live, 'device':'cpu', 'harness_version':importlib.metadata.version('lm_eval')}


def smoke(protocol):
    evidence_path = RESULTS/'verification.json'
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        require_verified(evidence, protocol)
        return
    scoring = verify_scoring()
    parent = ROOT/'runs/beta-refinement/verification'
    paths = {}
    for label, module, beta in (('legacy','odin.controlled_train',None), ('control','odin.beta_train',.95), ('candidate','odin.beta_train',.999)):
        out = parent/label
        if out.exists():
            raise ValueError('Existing verification attempt; explicit inspection required, no automatic retry')
        run_child(command(module,out,beta,smoke=True), 'smoke-'+label)
        paths[label] = out/'latest.pt'
    states = {k:torch.load(p,map_location='cpu',weights_only=True) for k,p in paths.items()}
    a,b,c = (states[k] for k in ('legacy','control','candidate'))
    if any(not torch.equal(v,b['model'][k]) for k,v in a['model'].items()) or not torch.equal(a['rng'],b['rng']):
        raise ValueError('New trainer changed beta2 .95 compiled GPU trajectory')
    for key, values in a['optimizer']['state'].items():
        for name, value in values.items():
            other = b['optimizer']['state'][key][name]
            if not (torch.equal(value,other) if torch.is_tensor(value) else value == other):
                raise ValueError('Control optimizer state mismatch')
    for state in states.values():
        if state['tokens'] != 32*16384 or state['source_tokens'] != a['source_tokens'] or any(not bool(torch.isfinite(v).all()) for v in state['model'].values()):
            raise ValueError('Smoke token counts or weights invalid')
    if any(g['betas'] != (.9,.999) for g in c['optimizer']['param_groups']):
        raise ValueError('Candidate optimizer beta2 was not applied')
    consumed = sum(charged_tokens(p.parent/'work.jsonl') for p in paths.values())
    if consumed != protocol['verification_charged_cap']:
        raise ValueError('Verification charged budget mismatch')
    evidence = {'passed':True, 'protocol_sha256':digest(protocol), 'scoring':scoring,
                'gpu':'NVIDIA GeForce RTX 4070', 'torch':str(torch.__version__),
                'compiled_control_weights_and_moments_exact':True, 'charged_tokens':consumed,
                'checkpoints':{k:sha256(p) for k,p in paths.items()}}
    atomic_json(evidence_path,evidence)
    status('verified', evidence=evidence)


def verify_candidate(protocol):
    state = torch.load(RUN/'latest.pt',map_location='cpu',weights_only=True)
    validate_state(state, RECIPE, protocol['config'], protocol['data'], train_code(), ['fineweb.bin','wiki.bin'])
    require_budget(charged_tokens(RUN/'work.jsonl'),state['tokens'])
    if state['tokens'] != CAP or state['parameter_count'] != 49295872:
        raise ValueError('Candidate incomplete or wrong parameter count')
    if any(g['betas'] != (.9,.999) for g in state['optimizer']['param_groups']):
        raise ValueError('Candidate optimizer state disagrees with recipe')
    old = torch.load(CONTROL,map_location='cpu',weights_only=True)
    if state['source_tokens'] != old['source_tokens']:
        raise ValueError('Candidate/control data exposure differs')
    return state


def evaluation(checkpoint, label):
    target = RESULTS/f'{label}.json'
    if not target.exists():
        run_child([sys.executable,'-m','odin.beta_eval','--checkpoint',str(checkpoint),
                   '--data',str(COMMON),'--prompts',str(PROMPTS),'--device','cuda','--output',str(target)], 'evaluate-'+label)
    report = json.loads(target.read_text())
    beta_eval.validate_report(report,checkpoint,COMMON,PROMPTS,'cuda')
    return report


def main(mode):
    RESULTS.mkdir(parents=True,exist_ok=True)
    with writer_lock(ROOT/'results/controlled/runner.lock'):
        require_resources(time.time(),CUTOFF,shutil.disk_usage(ROOT).free)
        protocol = frozen_protocol()
        if mode == 'prepare':
            status('prepared',protocol_sha256=digest(protocol)); return
        if mode == 'smoke':
            smoke(protocol); return
        require_verified(json.loads((RESULTS/'verification.json').read_text()),protocol)
        if mode == 'run':
            if RUN.exists():
                raise ValueError('Existing candidate run; explicit inspection required, no automatic restart')
            require_budget(0,0)
            run_child(command('odin.beta_train',RUN,.999), 'train-candidate',14400)
        state = verify_candidate(protocol)
        snapshot = RUN/'submission.pt'
        if not snapshot.exists():
            export(RUN/'latest.pt',snapshot)
        exported = torch.load(snapshot,map_location='cpu',weights_only=True)
        if exported['run_id'] != state['run_id'] or exported['tokens'] != CAP or any(not torch.equal(v,exported['model'][k]) for k,v in state['model'].items()):
            raise ValueError('Export does not match completed candidate')
        del state, exported
        reports = {name:evaluation(path,name) for name,path in (('control',CONTROL),('candidate',snapshot))}
        comparison = beta_eval.compare(reports['control'],reports['candidate'])
        comparison['checkpoint_sha256'] = {k:r['checkpoint_sha256'] for k,r in reports.items()}
        atomic_json(RESULTS/'comparison.json',comparison)
        blind_review(reports,RESULTS)
        status('completed-needs-quality-review',numeric_gate_passed=comparison['numeric_gate_passed'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','smoke','run','evaluate'])
    args = p.parse_args()
    def stop(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        main(args.mode)
    except BlockingIOError:
        raise
    except BaseException as exc:
        status('failed',error=repr(exc)); raise
