import json
import numpy as np
import torch
import pytest
from odin.data import partition_documents, encode_documents, require_benchmark_coverage
from odin.model import ModelConfig, LanguageModel
from odin.tokenizer import Tokenizer
from odin.train import train, load_checkpoint

def test_document_deduplication_and_stable_split():
    docs = ['document number %d is different and has several words' % i for i in range(500)]
    a, b, stats = partition_documents(docs + docs[:10])
    assert len(a) + len(b) == 500
    assert not set(a) & set(b)
    assert stats['duplicates'] == 10
    ar, br, _ = partition_documents(reversed(docs))
    assert set(a) == set(ar) and set(b) == set(br)

def test_incomplete_benchmark_cache_is_rejected():
    with pytest.raises(ValueError,match='Missing protected benchmark'):
        require_benchmark_coverage([{'file':'wikitext-test.arrow'}])

def test_encode_retains_document_boundaries(tmp_path):
    tok = Tokenizer.train(['hello world hello world'], 260)
    count = encode_documents(['hello', 'world'], tok, tmp_path/'x.bin')
    actual = np.fromfile(tmp_path/'x.bin', dtype=np.uint16).tolist()
    assert actual == tok.encode('hello') + [tok.eos_id] + tok.encode('world') + [tok.eos_id]
    assert count == len(actual)

def fixture_data(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    tok = Tokenizer.train(['abc abc abc ' * 100], 260)
    tok.save(data/'tokenizer.json')
    for name in ('train', 'dev'):
        encode_documents(['abc abc abc ' * 200], tok, data/f'{name}.bin')
    (data/'manifest.json').write_text(json.dumps({'purpose':'test'}))
    cfg = ModelConfig(vocab_size=tok.vocab_size, width=32, layers=1, heads=4, ff=64, context=16)
    return data, cfg

@pytest.mark.parametrize('mixture', [False, True])
def test_training_reduces_loss_and_exact_resume(tmp_path, mixture):
    data, cfg = fixture_data(tmp_path)
    if mixture:
        import shutil
        shutil.copyfile(data/'train.bin', data/'stories.bin')
        (data/'manifest.json').write_text(json.dumps({'train_sampling':[
            {'file':'train.bin','weight':0.8},{'file':'stories.bin','weight':0.2}]}))
    args = dict(data=data, config=cfg, steps=8, batch_size=2, accumulation=1, lr=0.003, device='cpu', seed=7, eval_every=4)
    full = train(output=tmp_path/'full', **args)
    train(output=tmp_path/'part', stop_after=4, **args)
    with (tmp_path/'part/metrics.jsonl').open('a') as f:
        f.write(json.dumps({'step':8,'dev_loss':999.0})+'\n')
    resumed = train(output=tmp_path/'part', resume=tmp_path/'part/latest.pt', **args)
    m1, t1, s1 = load_checkpoint(full)
    m2, _, s2 = load_checkpoint(resumed)
    assert s1['step'] == s2['step'] == 8
    assert s1['tokens'] == 8 * 2 * 16
    for k, v in m1.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    log = [json.loads(x) for x in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    assert log[-1]['dev_loss'] < log[0]['dev_loss']
    assert s1['provenance']['initialization'] == 'random'
    resumed_log=[json.loads(x) for x in (tmp_path/'part/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in resumed_log] == [0,4,8]
    if mixture:
        assert 'stories.bin' in s1['fingerprint']
        with (data/'stories.bin').open('ab') as f:
            f.write(b'\x00\x00')
        with pytest.raises(ValueError, match='identical data'):
            train(output=tmp_path/'mutated', resume=resumed, **args)

def test_requested_cuda_does_not_silently_fall_back(tmp_path):
    if torch.cuda.is_available():
        pytest.skip('CUDA available')
    data, cfg = fixture_data(tmp_path)
    with pytest.raises(RuntimeError, match='CUDA'):
        train(data=data, config=cfg, output=tmp_path/'bad', steps=1, device='cuda')

def test_compiled_training_requires_cuda(tmp_path):
    data, cfg = fixture_data(tmp_path)
    with pytest.raises(ValueError,match='Compiled training requires CUDA'):
        train(data=data,config=cfg,output=tmp_path/'bad',steps=1,device='cpu',compile_model=True)
