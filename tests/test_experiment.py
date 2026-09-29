import json
import pytest
from odin.experiment import checkpoint_stage, bind_decision, recover_summary


def test_stage_requires_expected_recipe_and_provenance():
    state={'step':5,'tokens':81920,'recipe':{'steps':10,'batch_size':32,'accumulation':1},
           'config':{'context':512},'provenance':{'source_sha256':{'train.py':'abc'}},'fingerprint':{'manifest.json':'def'}}
    assert checkpoint_stage(state,5,state['recipe'],state['config'],state['provenance']['source_sha256'],state['fingerprint'])
    for change in [{'tokens':1},{'step':6},{'fingerprint':{'manifest.json':'changed'}},{'recipe':{'steps':11}}]:
        with pytest.raises(ValueError):
            checkpoint_stage({**state,**change},5,state['recipe'],state['config'],state['provenance']['source_sha256'],state['fingerprint'])


def test_existing_decision_cannot_be_silently_replaced(tmp_path):
    path=tmp_path/'decision.json'
    first={'selected':'B','pilot_hashes':{'A':'a','B':'b','C':'c'}}
    bind_decision(path,first)
    bind_decision(path,first)
    with pytest.raises(ValueError):
        bind_decision(path,{**first,'selected':'C'})
    assert json.loads(path.read_text())==first


def test_experiment_lock_survives_json_tuple_round_trip(tmp_path):
    frozen={'candidates':{'A':('configs/pocket.json','data/edu')}}
    path=tmp_path/'lock.json'
    bind_decision(path,frozen)
    bind_decision(path,frozen)


def test_recovery_uses_checkpoint_after_summary_write_interruption(tmp_path):
    import torch
    state={'step':5,'tokens':81920,'recipe':{'steps':10,'batch_size':32,'accumulation':1},
           'config':{'context':512},'provenance':{'source_sha256':{'train.py':'abc'}},'fingerprint':{'manifest.json':'def'},
           'model':{'weight':torch.tensor([1.0])},'optimizer':{},'rng':torch.get_rng_state(),'cuda_rng':None,'tokenizer_json':'{}'}
    checkpoint=tmp_path/'latest.pt';torch.save(state,checkpoint)
    summary=tmp_path/'summary.json';summary.write_text('{"step":4}')
    actual=recover_summary(checkpoint,summary,state['recipe'],state['config'],state['provenance']['source_sha256'],state['fingerprint'])
    assert actual['step']==5 and json.loads(summary.read_text())['tokens']==81920
    assert 'model' not in actual


def test_sigterm_cleans_up_training_child(tmp_path):
    import os,signal,subprocess,sys,time
    child_pid=tmp_path/'child.pid'
    worker='from scripts import refinement as r; from pathlib import Path; r.RESULTS=Path('+repr(str(tmp_path))+'); r.install_shutdown_handler(); r.run('+repr([sys.executable,'-c','import os,time; from pathlib import Path; Path('+repr(str(child_pid))+').write_text(str(os.getpid())); time.sleep(60)'])+',"test-child")'
    process=subprocess.Popen([sys.executable,'-c',worker],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+15
        while not child_pid.exists() and time.monotonic()<deadline and process.poll() is None:
            time.sleep(0.05)
        assert child_pid.exists()
        pid=int(child_pid.read_text())
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(pid,0)
    finally:
        if process.poll() is None:
            process.kill();process.wait()
