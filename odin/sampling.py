"""Deterministic, source-weighted token-window sampling with strict validation."""
import json
import math
from pathlib import Path
import numpy as np
import torch


def training_sources(data):
    manifest = json.loads((Path(data)/'manifest.json').read_text())
    entries = manifest.get('train_sampling', [{'file':'train.bin', 'weight':1.0}])
    if not isinstance(entries, list) or not entries:
        raise ValueError('Training sources must be a nonempty list')
    names = []
    for entry in entries:
        name = entry['file']
        if not isinstance(name, str) or Path(name).name != name or not name.endswith('.bin'):
            raise ValueError('Training source must be a local .bin filename')
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate training source')
    _weights([entry['weight'] for entry in entries])
    return entries


def _weights(values):
    if not values or any(not isinstance(v, (int,float)) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Mixture weights must be finite and positive')
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError('Mixture weights must sum to one')


class TokenSampler:
    def __init__(self, sources, context, vocab_size):
        _weights([weight for _,weight in sources])
        self.arrays = [a for a,_ in sources]
        self.weights = torch.tensor([w for _,w in sources], dtype=torch.float64)
        self.context = context
        for a in self.arrays:
            if a.ndim != 1 or len(a) <= context:
                raise ValueError('Token corpora must exceed context length')
            if int(a.min()) < 0 or int(a.max()) >= vocab_size:
                raise ValueError('Corpus contains token IDs outside vocabulary')

    def batch(self, size, generator=None):
        if len(self.arrays) == 1:
            a = self.arrays[0]
            starts = torch.randint(len(a)-self.context, (size,), generator=generator).tolist()
            windows = [a[s:s+self.context+1] for s in starts]
        else:
            chosen = torch.multinomial(self.weights, size, replacement=True, generator=generator).tolist()
            windows = []
            for source in chosen:
                a = self.arrays[source]
                s = int(torch.randint(len(a)-self.context, (), generator=generator))
                windows.append(a[s:s+self.context+1])
        block = torch.from_numpy(np.stack(windows).astype(np.int64))
        return block[:, :-1], block[:, 1:]
