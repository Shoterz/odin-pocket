import json
import numpy as np
import pytest
import torch
from odin.sampling import TokenSampler, training_sources


def test_weighted_sampler_draws_whole_windows_from_one_source():
    sampler = TokenSampler([(np.full(100, 3, dtype=np.uint16), 0.8),
                            (np.full(100, 9, dtype=np.uint16), 0.2)], context=8, vocab_size=10)
    x, y = sampler.batch(10000, generator=torch.Generator().manual_seed(91))
    assert x.shape == y.shape == (10000, 8)
    assert torch.equal(x, y)
    assert torch.all(x == x[:, :1])
    assert 0.18 < (x[:, 0] == 9).float().mean().item() < 0.22


@pytest.mark.parametrize('weights', [[0.8, -0.2], [float('nan'), 1], [0.2, 0.2], [0, 1]])
def test_invalid_weights_fail_closed(weights):
    with pytest.raises(ValueError):
        TokenSampler([(np.zeros(20, dtype=np.uint16), w) for w in weights], 4, 10)


def test_sources_reject_path_escape_and_duplicate(tmp_path):
    for entries in [[{'file':'../outside.bin','weight':1}],
                    [{'file':'train.bin','weight':0.5},{'file':'train.bin','weight':0.5}]]:
        (tmp_path/'manifest.json').write_text(json.dumps({'train_sampling':entries}))
        with pytest.raises(ValueError):
            training_sources(tmp_path)


def test_invalid_token_and_short_source_rejected():
    for a in [np.array([20]*30), np.array([0]*4)]:
        with pytest.raises(ValueError):
            TokenSampler([(a, 1)], context=4, vocab_size=10)
