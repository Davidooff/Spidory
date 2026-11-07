"""Normalized transformer (nGPT) parts, as Dory uses them, with SwiGLU and RoPE.

Following nGPT (Loshchilov et al., 2025) and the Dory paper:
- Hidden states have unit length. A sublayer moves the state toward its own normalized
  output by learned per-channel step sizes: h <- N(h + alpha * (N(out) - h)).
- Weight rows have unit length along the embedding dimension, restored after every
  optimizer step by `normalize_weights`.
- Queries and keys are normalized after RoPE and scaled by learned s_qk, and attention
  uses sqrt(head_dim) as its softmax scale, since q.k is now a cosine.
- Dory additions: a sigmoid gate on attention outputs, and L2 normalization of the gated
  attention outputs before the output projection and of the SwiGLU activations before the
  down projection.
Learned scales use nGPT's parametrization: a stored value starts at `scale` and the
effective value is stored * (init / scale), which sets how fast it learns.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


class Scale(nn.Module):
    """A learned per-channel scale with nGPT's parametrization."""

    def __init__(self, size: int, init: float, scale: float):
        super().__init__()
        self.ratio = init / scale
        self.value = nn.Parameter(torch.full((size,), scale))

    def forward(self) -> torch.Tensor:
        return self.value * self.ratio


def swiglu_width(d: int) -> int:
    """Keeps feed-forward parameters close to a 4x GELU layer: 3 matrices of 8d/3."""
    return int(math.ceil(8 * d / 3 / 64) * 64)


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotates pairs of channels by an angle that grows with position. x is (b, h, t, d)."""
    t, d = x.shape[-2], x.shape[-1]
    half = d // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class NBlock(nn.Module):
    """One nGPT layer: gated attention with RoPE, then SwiGLU, each blended in on the sphere."""

    def __init__(self, d: int, heads: int, alpha_init: float = 0.05):
        super().__init__()
        self.d, self.heads, self.head_dim = d, heads, d // heads
        width = swiglu_width(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.gate = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 2 * width, bias=False)
        self.down = nn.Linear(width, d, bias=False)
        root = 1 / math.sqrt(d)
        self.alpha_attn = Scale(d, alpha_init, root)
        self.alpha_mlp = Scale(d, alpha_init, root)
        self.s_qk = Scale(d, 1.0, root)
        self.s_u = Scale(width, 1.0, 1.0)
        self.s_v = Scale(width, 1.0, 1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        split = lambda z: z.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        s_qk = split(self.s_qk().expand(b, t, d))
        q = unit(rope(split(q))) * s_qk
        k = unit(rope(split(k))) * s_qk
        out = F.scaled_dot_product_attention(q, k, split(v), is_causal=True, scale=math.sqrt(self.head_dim))
        out = out.transpose(1, 2).reshape(b, t, d)
        out = unit(torch.sigmoid(self.gate(h)) * out)
        h = unit(h + self.alpha_attn() * (unit(self.proj(out)) - h))

        u, value = self.up(h).chunk(2, dim=-1)
        hidden = unit((u * self.s_u()) * F.silu(value * self.s_v() * math.sqrt(self.d)))
        return unit(h + self.alpha_mlp() * (unit(self.down(hidden)) - h))


class NExpert(nn.Module):
    """A small SwiGLU expert. Its step size starts at 0, so joining changes nothing at first."""

    def __init__(self, d: int, width: int):
        super().__init__()
        self.d = d
        self.up = nn.Linear(d, 2 * width, bias=False)
        self.down = nn.Linear(width, d, bias=False)
        self.s_u = Scale(width, 1.0, 1.0)
        self.s_v = Scale(width, 1.0, 1.0)
        self.alpha = Scale(d, 0.0, 1 / math.sqrt(d))

    def proposal(self, h: torch.Tensor) -> torch.Tensor:
        u, value = self.up(h).chunk(2, dim=-1)
        hidden = unit((u * self.s_u()) * F.silu(value * self.s_v() * math.sqrt(self.d)))
        return unit(self.down(hidden))


@torch.no_grad()
def normalize_weights(model: nn.Module, embedding: nn.Embedding) -> None:
    """Restores unit length along the embedding dimension, as nGPT does after every step."""
    embedding.weight.copy_(unit(embedding.weight))
    for module in model.modules():
        if isinstance(module, (NBlock, NExpert)):
            for name, layer in module.named_children():
                if not isinstance(layer, nn.Linear):
                    continue
                # Matrices that read the state are normalized per row; those that write it, per column.
                dim = 0 if name in ("proj", "down") else 1
                layer.weight.copy_(F.normalize(layer.weight, dim=dim, eps=1e-6))
