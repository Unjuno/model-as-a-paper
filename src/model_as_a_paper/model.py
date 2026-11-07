"""Bias-free pre-norm causal Transformer; FP32 master weights, optional autocast."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16384
    dim: int = 512
    layers: int = 12
    heads: int = 8
    kv_heads: int = 2
    ffn_dim: int = 1536
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ('vocab_size', 'dim', 'layers', 'heads', 'kv_heads', 'ffn_dim', 'max_seq_len'):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError('dim must divide heads; heads must divide kv_heads')
        if (self.dim // self.heads) % 2:
            raise ValueError('RoPE requires an even head dimension')
        if not math.isfinite(self.rope_base) or self.rope_base <= 1 or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('invalid RoPE base or normalization epsilon')

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.hd = cfg.dim // cfg.heads
        self.base = cfg.rope_base
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.kv_heads * self.hd, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.kv_heads * self.hd, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def rotary(self, x, offset):
        # Float32 phase arithmetic is deliberate, even under BF16 autocast.
        freq = self.base ** (-torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd)
        phase = torch.outer(torch.arange(offset, offset + x.shape[2], device=x.device,
                                        dtype=torch.float32), freq)
        cos = torch.cat((phase.cos(), phase.cos()), -1).to(x.dtype)[None, None]
        sin = torch.cat((phase.sin(), phase.sin()), -1).to(x.dtype)[None, None]
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), -1) * sin

    def project(self, x, offset):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.hd).transpose(1, 2)
        return self.rotary(q, offset), self.rotary(k, offset), v

    def forward(self, x, past=None, use_cache=False):
        offset = 0 if past is None else past[0].shape[2]
        q, k, v = self.project(x, offset)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # SDPA's non-square causal mask is upper-left aligned, NOT a decode mask.
        # Cached queries need absolute-position comparison, including multi-token appends.
        mask = None
        if offset:
            qp = torch.arange(offset, offset + x.shape[1], device=x.device)
            kp = torch.arange(k.shape[2], device=x.device)
            mask = qp[:, None] >= kp[None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=(offset == 0), dropout_p=0.0,
                                           enable_gqa=(self.heads != self.kv_heads))
        y = self.out(y.transpose(1, 2).contiguous().view(x.shape))
        return y, ((k, v) if use_cache else None)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps), RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x, past=None, use_cache=False):
        a, cache = self.attn(self.norm1(x), past, use_cache)
        x = x + a
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), cache


class PaperLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def optimizer_groups(self, weight_decay):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (decay if p.ndim >= 2 and name != 'embed.weight' else no_decay).append(p)
        return [{'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}]

    def forward(self, ids, cache=None, use_cache=False):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('ids must be a nonempty [batch, sequence] tensor')
        if (cache is not None or use_cache) and self.training:
            raise ValueError('KV cache is inference-only')
        if cache is not None and len(cache) != self.cfg.layers:
            raise ValueError('cache layer count mismatch')
        offset = 0 if cache is None else cache[0][0].shape[2]
        if offset + ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError('context overflow; no silent truncation or RoPE extrapolation')
        x = self.embed(ids)
        out_cache = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if cache is None else cache[i], use_cache)
            if use_cache: out_cache.append(c)
        # Output projection references the embedding directly: exactly one Parameter.
        return F.linear(self.norm(x), self.embed.weight), (out_cache if use_cache else None)
