import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from odin.server import Workbench, make_server


def test_missing_checkpoint_is_visible_and_not_fabricated(tmp_path):
    app=Workbench(tmp_path/'missing.pt',tmp_path)
    assert app.status()['ready'] is False
    with pytest.raises(RuntimeError,match='checkpoint'):
        app.generate({'prompt':'A small model'})

def test_training_evidence_cannot_come_from_another_run(tmp_path):
    app=Workbench(tmp_path/'missing.pt',tmp_path)
    app.metadata={'step':10,'run_id':'loaded-run'}
    (tmp_path/'metrics.jsonl').write_text(json.dumps({'step':5,'dev_loss':1.0,'run_id':'unrelated-run'})+'\n')
    assert app.results()['training']==[]

def test_exported_training_evidence_works_without_original_run_directory(tmp_path):
    app=Workbench(tmp_path/'missing.pt',tmp_path)
    app.metadata={'step':10,'run_id':'loaded-run'}
    row={'step':10,'dev_loss':2.0,'tokens':100,'run_id':'loaded-run'}
    (tmp_path/'training-metrics.jsonl').write_text(json.dumps(row)+'\n')
    assert app.results()['training']==[row]

def test_compare_preserves_existing_word_separator(tmp_path):
    from odin.model import ModelConfig,LanguageModel
    from odin.tokenizer import Tokenizer
    from odin.evaluate import Scorer
    tok=Tokenizer.train(['Water will eventually freeze. It will boil. ']*10,280)
    model=LanguageModel(ModelConfig(vocab_size=tok.vocab_size,width=16,layers=1,heads=2,ff=32,context=64))
    app=Workbench(tmp_path/'missing.pt',tmp_path)
    app.scorer=Scorer(model,tok)
    expected=app.scorer.score([('Water will eventually ','freeze.')])[0][0]
    result=app.compare({'prompt':'Water will eventually ','candidates':['freeze.','boil.']})
    assert result['candidates'][0]['log_likelihood']==pytest.approx(expected,abs=1e-5)


@pytest.mark.parametrize('payload',[
    {'prompt':''}, {'prompt':'x'*17000}, {'prompt':42},
    {'prompt':'hello','max_new_tokens':10000},
    {'prompt':'hello','temperature':float('nan')},
    {'prompt':'hello','max_new_tokens':True},
])
def test_generation_rejects_bad_inputs_before_inference(tmp_path,payload):
    app=Workbench(tmp_path/'missing.pt',tmp_path)
    with pytest.raises(ValueError):
        app.generate(payload)


def test_http_routes_and_cross_origin_guard(tmp_path):
    app=Workbench(tmp_path/'missing.pt',tmp_path)
    server=make_server(app,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with urlopen(base+'/api/status') as r:
            assert json.load(r)['ready'] is False
        with urlopen(base+'/') as r:
            assert r.status==200
            assert b'ODIN' in r.read()
        req=Request(base+'/api/generate',data=b'{"prompt":"hello"}',headers={'Content-Type':'application/json','Origin':'https://untrusted.example'})
        with pytest.raises(HTTPError) as e:
            urlopen(req)
        assert e.value.code==403
        req=Request(base+'/api/generate',data=b'not json',headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError) as e:
            urlopen(req)
        assert e.value.code==400
    finally:
        server.shutdown()
        server.server_close()
