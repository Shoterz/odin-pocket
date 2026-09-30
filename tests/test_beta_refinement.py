import copy
import json
import math
import pytest
import torch
from test_training import fixture_data


def test_beta2_controls_optimizer_state_and_preserves_legacy_trajectory(tmp_path):
    from odin import beta_train, controlled_train
    data, cfg = fixture_data(tmp_path)
    args = dict(data=data, config=cfg, steps=8, batch_size=2, lr=.003,
                seed=7, eval_every=4, warmup_steps=2)
    paths = [controlled_train.train(output=tmp_path/'legacy', **args),
             beta_train.train(output=tmp_path/'control', beta2=.95, **args),
             beta_train.train(output=tmp_path/'candidate', beta2=.999, **args)]
    old, control, candidate = [torch.load(p, weights_only=True) for p in paths]
    assert all(torch.equal(v, control['model'][k]) for k, v in old['model'].items())
    assert torch.equal(old['rng'], control['rng'])
    assert old['source_tokens'] == control['source_tokens'] == candidate['source_tokens']
    assert any(not torch.equal(v, candidate['model'][k]) for k, v in control['model'].items())
    assert candidate['recipe']['beta2'] == .999
    assert all(g['betas'] == (.9, .999) for g in candidate['optimizer']['param_groups'])


def test_beta2_exact_resume_and_changed_beta_rejected(tmp_path):
    from odin.beta_train import train
    data, cfg = fixture_data(tmp_path)
    args = dict(data=data, config=cfg, steps=8, batch_size=2, lr=.003,
                seed=7, eval_every=4, warmup_steps=2, beta2=.999)
    full = train(output=tmp_path/'full', **args)
    partial = train(output=tmp_path/'part', stop_after=4, **args)
    with pytest.raises(ValueError, match='identical'):
        train(output=tmp_path/'bad', resume=partial, **{**args, 'beta2': .95})
    resumed = train(output=tmp_path/'part', resume=partial, **args)
    a, b = [torch.load(p, weights_only=True) for p in (full, resumed)]
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert a['source_tokens'] == b['source_tokens']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0, 1, True])
def test_beta2_rejects_invalid_values_before_creating_run(tmp_path, value):
    from odin.beta_train import train
    data, cfg = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='beta2'):
        train(data=data, config=cfg, output=tmp_path/'bad', steps=8, warmup_steps=2, beta2=value)
    assert not (tmp_path/'bad').exists()


def test_existing_charged_work_cannot_be_silently_restarted(tmp_path):
    from odin.beta_train import train
    data, cfg = fixture_data(tmp_path)
    out = tmp_path/'run'; out.mkdir()
    (out/'work.jsonl').write_text('{"charged_tokens":32}\n')
    with pytest.raises(ValueError, match='Existing'):
        train(data=data, config=cfg, output=out, steps=8, warmup_steps=2)


def test_standard_metrics_distinguish_raw_from_character_length():
    from odin.beta_eval import standard_question
    row = {'id':'q', 'expected':1, 'raw_scores':[-3., -4.], 'scores':[-3., -4.], 'correct':False, 'predicted':0}
    q = {'id':'q', 'expected':1, 'choices':[' a', ' bbbbbbbb']}
    result = standard_question(row, q)
    assert result['predicted'] == 0 and result['correct'] is False
    assert result['predicted_norm'] == 1 and result['correct_norm'] is True
    assert result['character_scores'] == [-3., -.5]
    assert result['token_mean_correct'] is False
    q['choices'][0] = 'a'
    with pytest.raises(ValueError):
        standard_question(row, q)


def example_report():
    return {'protocol_sha256':'same', 'trained_tokens':1000013824, 'parameter_count':49295872,
            'domains':{'fineweb':{'nll':3.}, 'wiki':{'nll':3.}, 'stories':{'nll':3.}},
            'questions':[{'id':str(i), 'expected':0, 'correct':i<50, 'correct_norm':i<50} for i in range(100)],
            'generation':[]}


def test_numeric_gate_requires_all_domains_and_both_accuracy_metrics():
    from odin.beta_eval import compare
    a = example_report(); b = copy.deepcopy(a)
    b['domains']['fineweb']['nll'] = 3. + math.log(.96)
    b['questions'][50]['correct'] = True
    assert compare(a, b, expected_questions=100)['numeric_gate_passed'] is True
    b['domains']['wiki']['nll'] = 3. + math.log(1.03)
    assert compare(a, b, expected_questions=100)['numeric_gate_passed'] is False
    b['domains']['wiki']['nll'] = 3.
    b['questions'][0]['correct_norm'] = False
    b['questions'][1]['correct_norm'] = False
    assert compare(a, b, expected_questions=100)['numeric_gate_passed'] is False
    b['protocol_sha256'] = 'changed'
    with pytest.raises(ValueError, match='protocol'):
        compare(a, b, expected_questions=100)


def test_budget_counts_uncheckpointed_work_and_rejects_overspend():
    from scripts.beta_refinement import require_budget
    require_budget(0, 0, 100)
    require_budget(40, 40, 100)
    require_budget(100, 100, 100)
    for charged, retained in ((41,40), (20,40), (101,101), (-1,0), (True,0)):
        with pytest.raises(ValueError):
            require_budget(charged, retained, 100)


def test_deadline_and_disk_guards_stop_before_work():
    from scripts.beta_refinement import require_resources
    require_resources(now=10, cutoff=20, free_bytes=9*1024**3)
    with pytest.raises(RuntimeError, match='deadline'):
        require_resources(now=20, cutoff=20, free_bytes=9*1024**3)
    with pytest.raises(RuntimeError, match='disk'):
        require_resources(now=10, cutoff=20, free_bytes=7*1024**3)


def test_shared_lock_rejects_second_writer(tmp_path):
    from scripts.beta_refinement import writer_lock
    with writer_lock(tmp_path/'lock'):
        with pytest.raises(BlockingIOError):
            with writer_lock(tmp_path/'lock'):
                pytest.fail('second writer entered')


def test_verification_requires_exact_protocol_and_success():
    from scripts.beta_refinement import require_verified
    protocol = {'source':'a', 'data':'b'}
    from odin.controlled_eval import digest
    evidence = {'protocol_sha256':digest(protocol), 'passed':True}
    require_verified(evidence, protocol)
    for bad in ({**evidence,'passed':False}, {**evidence,'protocol_sha256':'old'}):
        with pytest.raises(ValueError):
            require_verified(bad, protocol)


def test_question_metrics_match_real_harness_processing():
    from scripts.beta_refinement import local_arc_task
    from odin.beta_eval import standard_question
    task = local_arc_task()
    doc = {'question':'A bird.', 'choices':{'label':['A','B'], 'text':['a','bbbbbbbb']}, 'answerKey':'B'}
    requests = task.construct_requests(doc, task.doc_to_text(doc), metadata=('arc_easy',0,1))
    assert [r.args for r in requests] == [('Question: A bird.\nAnswer:', ' a'), ('Question: A bird.\nAnswer:', ' bbbbbbbb')]
    raw = [-3., -4.]
    real = task.process_results(doc, [(s,False) for s in raw])
    row = {'id':'q', 'expected':1, 'raw_scores':raw, 'scores':raw, 'correct':False, 'predicted':0}
    q = {'id':'q', 'expected':1, 'choices':[' a',' bbbbbbbb']}
    actual = standard_question(row,q)
    assert real['acc'] == actual['correct'] == 0
    assert real['acc_norm'] == actual['correct_norm'] == 1


def test_evaluation_roundtrip_rejects_corrupted_aggregate(tmp_path):
    from odin.beta_eval import evaluate, validate_report
    from odin.tokenizer import Tokenizer
    from odin.model import LanguageModel, ModelConfig
    from odin.data import encode_documents, sha256
    tok = Tokenizer.train(['A cup sits on the table. '*100], 260)
    tok.save(tmp_path/'tokenizer.json')
    for name in ('fineweb','wiki','stories'):
        encode_documents(['A cup sits on the table. '*300],tok,tmp_path/f'{name}-dev.bin')
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'questions.json').write_text(json.dumps([
        {'id':part, 'partition':part, 'prompt':'The cup is', 'choices':[' blue.',' near the table.'], 'expected':0}
        for part in ('screen','confirmation')]))
    prompts = tmp_path/'prompts.json'
    prompts.write_text(json.dumps({'generation':['The cup'],'seeds':[42], 'temperature':.8,'max_new_tokens':4}))
    cfg = ModelConfig(vocab_size=tok.vocab_size,width=16,layers=1,heads=2,ff=32,context=512)
    model = LanguageModel(cfg)
    ckpt = tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'config':cfg.to_dict(),'tokens':0,
                'fingerprint':{'tokenizer.json':sha256(tmp_path/'tokenizer.json')},
                'tokenizer_json':(tmp_path/'tokenizer.json').read_text()},ckpt)
    report = evaluate(ckpt,tmp_path,prompts,windows=2)
    assert len(report['questions']) == 2 and len(report['generation']) == 1
    assert validate_report(report,ckpt,tmp_path,prompts,'cpu',windows=2)
    report['accuracy'] += .5
    with pytest.raises(ValueError, match='aggregate'):
        validate_report(report,ckpt,tmp_path,prompts,'cpu',windows=2)


def test_child_timeout_records_failure_and_stops_process(tmp_path, monkeypatch):
    import sys
    from scripts import beta_refinement as runner
    monkeypatch.setattr(runner,'RESULTS',tmp_path)
    with pytest.raises(RuntimeError, match='deadline'):
        runner.run_child([sys.executable,'-c','import time; time.sleep(60)'],'timeout-test',timeout_seconds=.05)
    records = list((tmp_path/'attempts').glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['completed'] is False
    assert record['returncode'] != 0


def test_reporting_failure_after_spawn_reaps_child(tmp_path, monkeypatch):
    import os
    import signal
    import sys
    from scripts import beta_refinement as runner
    monkeypatch.setattr(runner,'RESULTS',tmp_path)
    seen = []
    def broken_status(stage, **details):
        seen.append(details['child_pid'])
        raise OSError('reporting failed')
    monkeypatch.setattr(runner,'status',broken_status)
    try:
        with pytest.raises(OSError, match='reporting failed'):
            runner.run_child([sys.executable,'-c','import time; time.sleep(60)'],'status-failure')
        with pytest.raises(ProcessLookupError):
            os.kill(seen[0],0)
    finally:
        if seen:
            try:
                os.kill(seen[0],signal.SIGKILL)
                os.waitpid(seen[0],0)
            except (ProcessLookupError, ChildProcessError):
                pass


def test_supervisor_death_stops_training_child(tmp_path):
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    from scripts import beta_refinement as runner
    marker = tmp_path/'child-ready'
    child_code = f'from pathlib import Path; import time; Path({str(marker)!r}).write_text("ready"); time.sleep(60)'
    code = (f'import sys; sys.path.insert(0,{str(runner.ROOT)!r}); '
            'from pathlib import Path; from scripts import beta_refinement as r; '
            f'r.RESULTS=Path({str(tmp_path)!r}); '
            f'r.run_child([sys.executable,"-c",{child_code!r}],"owned-child")')
    parent = subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    child = None
    def alive(pid):
        stat = Path(f'/proc/{pid}/stat')
        try:
            return stat.read_text().split(') ',1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
    try:
        limit = time.monotonic()+15
        while not (tmp_path/'status.json').exists() and time.monotonic()<limit:
            assert parent.poll() is None
            time.sleep(.05)
        child = json.loads((tmp_path/'status.json').read_text())['child_pid']
        while not marker.exists() and time.monotonic()<limit:
            assert parent.poll() is None
            time.sleep(.05)
        assert marker.read_text() == 'ready'
        assert alive(child)
        parent.kill(); parent.wait(timeout=5)
        limit = time.monotonic()+3
        while alive(child) and time.monotonic()<limit:
            time.sleep(.05)
        assert not alive(child), 'training survived supervisor death'
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=5)
        if child is not None and alive(child):
            os.kill(child,signal.SIGKILL)
