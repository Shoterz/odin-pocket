import json
from pathlib import Path
import pytest
from odin.release import verify_prepared, validate_report, verify_evidence_identity

def test_prepared_corpus_requires_matching_hashes(tmp_path):
    files={'train.bin':'incorrect','dev.bin':'incorrect','tokenizer.json':'incorrect'}
    for name in files:
        (tmp_path/name).write_text('broken')
    (tmp_path/'manifest.json').write_text(json.dumps({'files':files,'decontamination':{'protected_files':[]}}))
    with pytest.raises(ValueError):
        verify_prepared(tmp_path)

def test_partial_or_unrelated_results_never_pass_submission_gate():
    with pytest.raises(ValueError,match='full'):
        validate_report({'status':'partial'},'sha')
    with pytest.raises(ValueError,match='checkpoint'):
        validate_report({'status':'full','checkpoint_sha256':'other'},'sha')
    with pytest.raises(ValueError,match='Missing'):
        validate_report({'status':'full','checkpoint_sha256':'sha','results':{}},'sha')

def test_valid_but_unrelated_data_or_run_cannot_be_packaged(tmp_path):
    import hashlib
    data=tmp_path/'data';data.mkdir()
    run=tmp_path/'run';run.mkdir()
    (data/'train.bin').write_bytes(b'actual')
    state={'fingerprint':{'train.bin':hashlib.sha256(b'other').hexdigest()},'run_id':'correct','step':10,'tokens':100,'config':{},'recipe':{}}
    with pytest.raises(ValueError,match='checkpoint data'):
        verify_evidence_identity(state,data,run)
    state['fingerprint']['train.bin']=hashlib.sha256(b'actual').hexdigest()
    (run/'summary.json').write_text(json.dumps({**state,'run_id':'wrong'}))
    with pytest.raises(ValueError,match='run identity'):
        verify_evidence_identity(state,data,run)
    (run/'summary.json').write_text(json.dumps(state))
    (run/'metrics.jsonl').write_text(json.dumps({'run_id':'wrong','step':10,'tokens':100,'dev_loss':2.0})+'\n')
    with pytest.raises(ValueError,match='metrics'):
        verify_evidence_identity(state,data,run)

def test_full_label_cannot_hide_truncated_or_wrong_wikitext():
    import math
    report={'status':'full','limit':None,'checkpoint_sha256':'sha','results':{
        name:{'sample_len':count,'acc,none':0.5} for name,count in
        {'hellaswag':10042,'arc_easy':2376,'piqa':1838,'winogrande':1267}.items()}}
    wiki={'split':'test','subset':'wikitext-103-raw-v1','tokens':300000,'documents':61,
          'words':241211,'bytes':1287656,'nll':1200000,'context':512,'stride':256,
          'text_sha256':'bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a',
          'source_sha256':'2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea',
          'token_perplexity':math.exp(4),'word_perplexity':math.exp(1200000/241211),
          'bits_per_byte':1200000/(1287656*math.log(2))}
    report['wikitext_103']=wiki
    with pytest.raises(ValueError,match='WikiText'):
        validate_report(report,'sha')
    wiki['documents']=62
    validate_report(report,'sha')
    wiki['text_sha256']='wrong'
    with pytest.raises(ValueError,match='WikiText'):
        validate_report(report,'sha')
