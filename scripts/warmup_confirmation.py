"""Separate, preregistered long warmup comparison using the frozen training engine."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts import controlled_refinement as engine
from odin.controlled_eval import paired_interval,blind_review
from odin.controlled_train import charged_tokens
from odin.data import sha256
from odin.experiment import atomic_json,bind_decision

OLD_RESULTS=ROOT/'results/controlled'
RESULTS=ROOT/'results/warmup-confirmation'
PLAN=ROOT/'docs/superpowers/plans/2026-09-30-warmup-confirmation.md'
RECIPES={name:engine.recipe(61036,.0006,warmup,20261001) for name,warmup in [('short',200),('long',1000)]}
RUN_NAMES={'short':'warmup-confirm-200','long':'warmup-confirm-1000'}
HISTORICAL_RUNS=[*engine.OPTIMIZERS,'stories10','stories20']
NEW_BUDGET=2000027648


def require_pair_budget(charged,retained):
    if set(charged)!=set(RECIPES) or set(retained)!=set(RECIPES):
        raise ValueError('Pair accounting must cover both runs')
    target=NEW_BUDGET//2
    for name in RECIPES:
        if any(type(x) is not int or x<0 for x in (charged[name],retained[name])) or not retained[name]<=target or charged[name]<retained[name]:
            raise ValueError('Invalid pair accounting')
    if sum(charged.values())+sum(target-n for n in retained.values())>NEW_BUDGET:
        raise ValueError('The matched pair cannot finish within the charged budget after replay; no training launched')


def verify_pair_budget():
    charged,retained={},{}
    files=engine.fingerprint(ROOT/'data/controlled/control')
    config=json.loads(engine.CONFIG.read_text())
    for name,run in RUN_NAMES.items():
        out=ROOT/'runs/controlled'/run
        charged[name]=charged_tokens(out/'work.jsonl')
        checkpoint=out/'latest.pt'
        if checkpoint.exists():
            state=engine.torch.load(checkpoint,map_location='cpu',weights_only=True)
            engine.validate_state(state,RECIPES[name],config,files,engine.code_hashes(),['fineweb.bin','wiki.bin'])
            retained[name]=state['tokens']
            del state
        else:
            retained[name]=0
    require_pair_budget(charged,retained)


@contextmanager
def results_at(path):
    old=engine.RESULTS
    engine.RESULTS=path
    try: yield
    finally: engine.RESULTS=old


def verify_and_freeze():
    with results_at(OLD_RESULTS): engine.freeze()
    smoke=json.loads((OLD_RESULTS/'gpu-smoke.json').read_text())
    if not smoke['passed'] or smoke['source_sha256']!=engine.code_hashes():
        raise ValueError('GPU verification differs from current trainer')
    history={name:ROOT/'runs/controlled'/name/'work.jsonl' for name in HISTORICAL_RUNS}
    if sum(charged_tokens(p) for p in history.values())!=768049152:
        raise ValueError('Historical training accounting changed')
    unknown={p.name for p in (ROOT/'runs/controlled').iterdir() if p.is_dir()}-set(HISTORICAL_RUNS)-set(RUN_NAMES.values())
    if unknown: raise ValueError('Unplanned training directories: '+str(sorted(unknown)))
    verify_pair_budget()
    protocol={'schema':1,'recipes':RECIPES,'run_names':RUN_NAMES,'new_charged_token_cap':NEW_BUDGET,
              'cumulative_charged_token_cap':engine.MAX_TOKENS,
              'mixture':{'fineweb':.6,'wiki':.4,'stories':0},'partition':'confirmation','question_count':285,
              'evaluation_order':'both training runs finish, then both confirmation evaluations, then historical references',
              'primary_comparison':'long minus short; positive paired-bootstrap lower bound and per-domain NLL <= short+0.05; generation review required',
              'source_sha256':{'wrapper':sha256(__file__),'engine':sha256(engine.__file__),'plan':sha256(PLAN),
                               'prior_experiment_lock':sha256(OLD_RESULTS/'experiment-lock.json'),
                               'quality_rubric':sha256(ROOT/'docs/controlled-quality-rubric.md')},
              'historical_ledger_sha256':{n:sha256(p) for n,p in history.items()},
              'reference_checkpoint_sha256':{'published':sha256(ROOT.parent.parent/'runs/pocket/submission.pt'),
                                             'previous-B':sha256(ROOT/'runs/refinement/B/submission.pt')}}
    bind_decision(RESULTS/'experiment-lock.json',protocol)
    return protocol


def summarize_pair(short,long,expected_questions=285):
    for r in (short,long):
        if r['partition']!='confirmation' or r['trained_tokens']!=1000013824 or r['parameter_count']!=49295872 or len(r['questions'])!=expected_questions:
            raise ValueError('Mismatched confirmation partition, budget, parameters or coverage')
        if set(r['domains'])!={'fineweb','wiki','stories'}:
            raise ValueError('Missing confirmation domain')
        if any(not math.isfinite(v['nll']) or v['nll']<0 for v in r['domains'].values()):
            raise ValueError('Invalid confirmation loss')
    if short['protocol_sha256']!=long['protocol_sha256']:
        raise ValueError('Mismatched confirmation protocol')
    interval=paired_interval(short['questions'],long['questions'])
    changes={d:long['domains'][d]['nll']-short['domains'][d]['nll'] for d in ('fineweb','wiki','stories')}
    guards=all(changes[d]<=.05 for d in ('fineweb','wiki'))
    return {'paired_accuracy_interval':interval,'long_minus_short_domain_nll':changes,
            'general_domain_guard_passed':guards,'primary_gain_demonstrated':interval['lower']>0 and guards,
            'checkpoint_sha256':{'short':short['checkpoint_sha256'],'long':long['checkpoint_sha256']},
            'release_decision':'Pending qualitative review; no automatic promotion.',
            'limitations':'One fresh training seed; interval resamples questions, not training runs. Historical comparisons are descriptive.'}


def execute_pair(train,evaluate):
    checkpoints={name:train(name,r) for name,r in RECIPES.items()}
    return {name:evaluate(path,name) for name,path in checkpoints.items()}


def write_compute():
    runs=[]
    new_charged=0
    for name in RUN_NAMES.values():
        out=ROOT/'runs/controlled'/name
        new_charged+=charged_tokens(out/'work.jsonl')
        if (out/'summary.json').exists(): runs.append(json.loads((out/'summary.json').read_text()))
    history=sum(charged_tokens(ROOT/'runs/controlled'/n/'work.jsonl') for n in HISTORICAL_RUNS)
    attempts=[json.loads(p.read_text()) for p in sorted((RESULTS/'attempts').glob('*-train-*.json'))]
    if new_charged>NEW_BUDGET or history+new_charged>engine.MAX_TOKENS:
        raise ValueError('Charged compute cap exceeded')
    r={'runs':runs,'new_retained_tokens':sum(s['tokens'] for s in runs),'new_charged_tokens':new_charged,
       'historical_screen_charged_tokens':history,'cumulative_charged_tokens':history+new_charged,
       'new_training_seconds':sum(s['training_seconds'] for s in runs),
       'new_training_attempt_elapsed_seconds':sum(a.get('elapsed_seconds',0) for a in attempts),
       'attempts_missing_end_record':[a for a in attempts if 'elapsed_seconds' not in a],
       'limitations':'Charged tokens include reserved partial and replayed steps. Missing attempt end times are unknown. Retained time excludes rollback; preparation/evaluation are separate.'}
    atomic_json(RESULTS/'compute.json',r)
    return r


def write_report(reports,summary,compute):
    rows=[]
    for name,r in reports.items():
        n=len(r['questions']);correct=sum(q['correct'] for q in r['questions'])
        rows.append(f"| {name} | {r['trained_tokens']:,} | {correct}/{n} | {correct/n:.2%} | {r['domains']['fineweb']['nll']:.4f} | {r['domains']['wiki']['nll']:.4f} |")
    ci=summary['paired_accuracy_interval']
    conclusion='The primary statistical and domain-loss gates passed.' if summary['primary_gain_demonstrated'] else 'The primary comparison did not establish an improvement under the preregistered gates.'
    text=f'''# Long warmup confirmation results

{conclusion} Qualitative review is pending; no model has been promoted automatically.

Both new models trained from random initialization for 1,000,013,824 tokens with the same fresh seed, 60/40 FineWeb/WikiText mixture, learning rate 0.0006 and architecture. Only warmup differs: short=200 steps, long=1000. Both finished before any confirmation evaluation. Historical models have different budgets/recipes and are descriptive references.

| Model | Training tokens | Correct | Confirmation accuracy | FineWeb NLL ↓ | WikiText NLL ↓ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Long minus short accuracy: {ci['difference']*100:+.2f} percentage points; paired 95% question-bootstrap interval [{ci['lower']*100:+.2f}, {ci['upper']*100:+.2f}]. General-domain regression guard passed: {summary['general_domain_guard_passed']}. This does not measure training-seed uncertainty or establish broad intelligence. Scores use the frozen 285-question ARC validation confirmation split and mean continuation-token likelihood; they are not official test scores.

New charged tokens: {compute['new_charged_tokens']:,}; cumulative including screens: {compute['cumulative_charged_tokens']:,}. New training subprocess elapsed time: {compute['new_training_attempt_elapsed_seconds']/3600:.3f} hours. Preparation and evaluation are additional.

Inspect every item in `results/warmup-confirmation/blind-review.json` using `docs/controlled-quality-rubric.md` before reading its mapping. Some historical text may be recognizable. Author review is not independent human validation. No architecture experiment or public release is triggered by this numerical report.
'''
    path=ROOT/'docs/warmup-confirmation-results.md'
    # Preserve any later human/agent interpretation in a separate reviewed report.
    tmp=path.with_suffix('.md.tmp');tmp.write_text(text);tmp.replace(path)


def main(check_only=False):
    RESULTS.mkdir(parents=True,exist_ok=True)
    # Shared lock prevents the old pipeline from spending the same remaining budget.
    lock=(OLD_RESULTS/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with results_at(RESULTS):
        verify_and_freeze()
        if check_only:
            engine.status('preflight-verified');return
        def train(name,r):
            verify_and_freeze()
            checkpoint=engine.train_run(RUN_NAMES[name],'control',r)
            write_compute()
            return checkpoint
        def evaluate(path,name):
            verify_and_freeze()
            return engine.evaluation(path,'confirmation-'+name,'confirmation',True)
        reports=execute_pair(train,evaluate)
        summary=summarize_pair(reports['short'],reports['long'])
        bind_decision(RESULTS/'primary-comparison.json',summary)
        for name,path in [('published',ROOT.parent.parent/'runs/pocket/submission.pt'),('previous-B',ROOT/'runs/refinement/B/submission.pt')]:
            reports[name]=evaluate(path,name)
        atomic_json(RESULTS/'historical-comparisons.json',{n:paired_interval(reports[n]['questions'],reports['long']['questions']) for n in ('published','previous-B')})
        blind_review(reports,RESULTS)
        compute=write_compute()
        write_report(reports,summary,compute)
        engine.status('completed-needs-quality-review',primary_gain_demonstrated=summary['primary_gain_demonstrated'],
                      report=str(ROOT/'docs/warmup-confirmation-results.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
    def stop(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    os.chdir(ROOT)
    try: main(args.check_only)
    except BlockingIOError: raise
    except BaseException as exc:
        with results_at(RESULTS): engine.status('failed',error=repr(exc))
        raise
