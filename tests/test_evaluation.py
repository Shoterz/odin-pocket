import math
import torch
import pytest
from odin.evaluate import Scorer, scoring_windows
from odin.model import ModelConfig, LanguageModel
from odin.tokenizer import Tokenizer

def test_rolling_scores_every_target_once_with_context():
    jobs = scoring_windows([0], list(range(1,21)), 8)
    actual=[]
    for ids, targets, start in jobs:
        assert len(ids) <= 8
        assert start >= 0
        assert ids[start:] == ([0]+list(range(1,21)))[targets[0]-1:targets[-1]]
        actual.extend(targets)
    assert actual == list(range(1,21))

def test_uniform_likelihood_is_sum_not_mean():
    tok=Tokenizer.train(['some text more text'],260)
    model=LanguageModel(ModelConfig(vocab_size=tok.vocab_size,width=16,layers=1,heads=2,ff=32,context=8))
    for p in model.parameters():
        p.data.zero_()
    scorer=Scorer(model,tok,batch_size=2)
    result=scorer.score_tokens([([0],[1,2,3]),([1,2],list(range(3,30)))])
    assert result[0][0] == pytest.approx(-3*math.log(tok.vocab_size),abs=1e-5)
    assert result[1][0] == pytest.approx(-27*math.log(tok.vocab_size),abs=1e-4)
    assert not result[0][1]

def test_empty_continuation_and_batch_parity():
    torch.manual_seed(42)
    tok=Tokenizer.train(['some text more text'],260)
    model=LanguageModel(ModelConfig(vocab_size=tok.vocab_size,width=16,layers=1,heads=2,ff=32,context=8))
    pairs=[([0],[1,2,3]),([1,2],[3]),([0],[])]
    a=Scorer(model,tok,batch_size=1).score_tokens(pairs)
    b=Scorer(model,tok,batch_size=3).score_tokens(pairs)
    for x,y in zip(a,b):
        assert x[0] == pytest.approx(y[0],abs=1e-5)
        assert x[1] == y[1]
    assert a[-1] == (0.0,True)
    assert Scorer(model,tok).score([('some text ', '')]) == [(0.0,True)]
