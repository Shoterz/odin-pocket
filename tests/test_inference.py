import torch
from odin.model import LanguageModel,ModelConfig
from odin.inference import CachedDecoder


def test_cached_logits_match_full_causal_forward():
    torch.manual_seed(23)
    model=LanguageModel(ModelConfig(vocab_size=64,width=32,layers=2,heads=4,ff=64,context=16)).eval()
    cached=CachedDecoder(model)
    ids=[1,2,3]
    for token in [4,5,6,7,8]:
        expected=model(torch.tensor([ids]))[0][0,-1]
        actual=cached.next_logits(ids)
        assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)
        ids.append(token)


def test_context_rollover_and_prompt_change_reset_cache():
    torch.manual_seed(23)
    model=LanguageModel(ModelConfig(vocab_size=64,width=32,layers=2,heads=4,ff=64,context=8)).eval()
    cached=CachedDecoder(model)
    for ids in ([1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8],list(range(1,10)),[8,7,6],[8,7,6,5]):
        expected=model(torch.tensor([ids[-8:]]))[0][0,-1]
        assert torch.allclose(cached.next_logits(ids),expected,atol=1e-6,rtol=1e-5)


def test_cached_generation_preserves_greedy_output():
    from odin.tokenizer import Tokenizer
    from odin.evaluate import Scorer
    tok=Tokenizer.train(['a small language model predicts the next token.']*20,280)
    torch.manual_seed(23)
    model=LanguageModel(ModelConfig(vocab_size=tok.vocab_size,width=32,layers=2,heads=4,ff=64,context=32)).eval()
    scorer=Scorer(model,tok)
    cached=scorer.generate('a small',max_new_tokens=40,temperature=0,use_cache=True)
    plain=scorer.generate('a small',max_new_tokens=40,temperature=0,use_cache=False)
    assert cached['text']==plain['text']
    assert [x['id'] for x in cached['trace']]==[x['id'] for x in plain['trace']]
