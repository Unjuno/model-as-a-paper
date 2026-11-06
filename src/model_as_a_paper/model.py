"""MAP-44M: causal GQA decoder. No pretrained weights or custom CUDA extension."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_384
    dim: int = 512
    layers: int = 12
    heads: int = 8
    kv_heads: int = 2
    ffn_dim: int = 1536
    max_seq_len: int = 2048
    rope_base: float = 10_000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ('vocab_size', 'dim', 'layers', 'heads', 'kv_heads', 'ffn_dim', 'max_seq_len'):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError('dim must divide into heads, and heads into kv_heads')
        if (self.dim // self.heads) % 2:
            raise ValueError('RoPE head dimension must be even')
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError('rope_base must be finite and > 1')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be positive and finite')

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.head_dim = cfg.dim // cfg.heads
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # Diagnostic/reference route for backends without native GQA.
        self.reference_gqa = False

    def forward(self, x, cos, sin, cache=None, use_cache=False):
        batch, length, _ = x.shape
        q = rotate(self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2), cos, sin)
        k = rotate(self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        past = 0
        if cache is not None:
            ck, cv = cache
            if ck.shape != cv.shape or ck.shape[:2] != (batch, self.kv_heads) or ck.shape[-1] != self.head_dim:
                raise ValueError('Invalid KV cache shape')
            past = ck.shape[-2]
            k, v = torch.cat((ck, k), dim=-2), torch.cat((cv, v), dim=-2)
        next_cache = (k, v) if use_cache else None
        mask = None
        causal = past == 0
        if past and length > 1:
            # is_causal=True with unequal lengths is upper-left aligned in SDPA.
            # Cached decoding requires absolute positions instead.
            mask = torch.arange(k.shape[-2], device=x.device)[None, :] <= (
                torch.arange(length, device=x.device)[:, None] + past
            )
        if self.reference_gqa:
            factor = self.heads // self.kv_heads
            k, v = k.repeat_interleave(factor, 1), v.repeat_interleave(factor, 1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0,
            is_causal=causal, enable_gqa=not self.reference_gqa,
        )
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1)), next_cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin, cache=None, use_cache=False):
        y, next_cache = self.attention(self.attn_norm(x), cos, sin, cache, use_cache)
        x = x + y
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z)), next_cache


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        inv = config.rope_base ** (-torch.arange(0, config.dim // config.heads, 2, dtype=torch.float32) / (config.dim // config.heads))
        self.register_buffer('inv_freq', inv, persistent=False)
        # Initialize the shared matrix once, not once as embedding and again as head.
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02 / math.sqrt(2 * config.layers) if name.endswith(('attention.out.weight', 'down.weight')) else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, ids: Tensor, cache=None, use_cache=False):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('ids must have shape [batch, nonempty sequence]')
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError('cache must contain one entry per layer')
        past = 0 if cache is None else cache[0][0].shape[-2]
        if cache is not None and any(k.shape[-2] != past for k, _ in cache):
            raise ValueError('Layer cache lengths disagree')
        if past + ids.shape[1] > self.config.max_seq_len:
            raise ValueError('Context limit exceeded; silent sliding/truncation is disabled')
        positions = torch.arange(past, past + ids.shape[1], device=ids.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        x = self.embedding(ids)
        result = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, cos, sin, None if cache is None else cache[i], use_cache)
            if use_cache:
                result.append(kv)
        return self.lm_head(self.final_norm(x)), result if use_cache else None
