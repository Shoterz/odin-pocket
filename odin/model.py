"""Compact causal Transformer, with no pretrained-model loading path."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    width: int = 512
    layers: int = 12
    heads: int = 8
    ff: int = 1536
    context: int = 512
    rope_base: float = 10000.0

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in (self.vocab_size, self.width, self.layers, self.heads, self.ff, self.context)):
            raise ValueError('Model dimensions must be positive integers')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('Head width must be even and divide model width')
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError('Invalid rotary base')
        count = self.vocab_size * self.width + self.layers * (4 * self.width**2 + 3 * self.width * self.ff + 2 * self.width) + self.width
        if count > 50_000_000:
            raise ValueError(f'{count:,} parameters exceeds 50,000,000 cap')
        return self

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return normalized.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = c.heads, c.width // c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.proj = nn.Linear(c.width, c.width, bias=False)
        inv = c.rope_base ** (-torch.arange(0, self.dim, 2).float() / self.dim)
        theta = torch.outer(torch.arange(c.context).float(), inv)
        self.register_buffer('cos', theta.cos()[None, None], persistent=False)
        self.register_buffer('sin', theta.sin()[None, None], persistent=False)

    def rotate(self, x):
        a, b = x[..., ::2], x[..., 1::2]
        cos, sin = self.cos[:, :, :x.size(2)].to(x.dtype), self.sin[:, :, :x.size(2)].to(x.dtype)
        return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm, self.ff_norm = RMSNorm(c.width), RMSNorm(c.width)
        self.attn = Attention(c)
        self.gate = nn.Linear(c.width, c.ff, bias=False)
        self.up = nn.Linear(c.width, c.ff, bias=False)
        self.down = nn.Linear(c.ff, c.width, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        y = self.ff_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._init)
        # Residual scaling keeps depth stable at initialization.
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context:
            raise ValueError('Input must be a nonempty batch within model context')
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss
