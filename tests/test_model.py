import math
import torch
import pytest
from odin.model import ModelConfig, LanguageModel
from odin.tokenizer import Tokenizer

torch.set_num_threads(2)

def tiny():
    return LanguageModel(ModelConfig(vocab_size=64, width=32, layers=2, heads=4, ff=64, context=32))

def test_no_future_token_leakage():
    torch.manual_seed(4)
    m = tiny().eval()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 9, 8]])
    assert torch.allclose(m(a)[0][:, :3], m(b)[0][:, :3], atol=1e-6)
    assert not torch.allclose(m(a)[0][:, 3:], m(b)[0][:, 3:])

def test_loss_predicts_supplied_next_tokens():
    m = tiny()
    for p in m.parameters():
        p.data.zero_()
    _, loss = m(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    assert loss.item() == pytest.approx(math.log(64), abs=1e-5)
    loss.backward()

def test_cap_counts_tied_embedding_once_and_rejects_oversized():
    m = LanguageModel(ModelConfig())
    assert 45_000_000 < m.parameter_count < 50_000_000
    assert m.output.weight is m.embedding.weight
    with pytest.raises(ValueError, match='50,000,000'):
        ModelConfig(width=2048).validate()

def test_invalid_configuration_and_long_input():
    with pytest.raises(ValueError):
        ModelConfig(width=33, heads=4).validate()
    with pytest.raises(ValueError, match='context'):
        tiny()(torch.zeros(1, 33, dtype=torch.long))

def test_tokenizer_roundtrip_and_reload(tmp_path):
    tok = Tokenizer.train(['A small model reads books. ' * 20], 300)
    s = 'Hello, 世界! café\n<|endoftext|> 🦉'
    assert tok.decode(tok.encode(s)) == s
    tok.save(tmp_path / 'tokenizer.json')
    other = Tokenizer.load(tmp_path / 'tokenizer.json')
    assert other.encode(s) == tok.encode(s)
