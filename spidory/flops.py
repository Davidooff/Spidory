"""Compute cost per token, calculated from the layer sizes.

The result does not depend on the PyTorch version. One FLOP is one add or one multiply, so one
multiply-add counts as 2. LayerNorm and activations are not counted. "Per token" means the cost
of one full forward pass divided by the sequence length.
"""
from __future__ import annotations

TERA = 1e12


def _block_macs(d: int, t: int) -> int:
    feed_forward = 8 * d * d * t
    qkv_and_proj = 4 * d * d * t
    attention = 2 * t * t * d  # scores, then values, summed across all heads
    return feed_forward + qkv_and_proj + attention


def tflops_by_k(cfg) -> dict[int, float]:
    """TFLOPS per token at each run count, using one expert when the model has experts."""
    d, t = cfg.d_model, cfg.seq_len
    block = _block_macs(d, t)
    fixed = (cfg.n_input + cfg.n_output) * block + cfg.vocab_size * d * t
    per_run = cfg.n_core * block
    if cfg.use_experts:
        per_run += 2 * d * cfg.expert_dim * t
    return {k: 2 * (fixed + k * per_run) / t / TERA for k in range(1, cfg.max_runs + 1)}


def average_tflops(table: dict[int, float], run_counts: list[int]) -> float:
    """Average cost, weighted by how often each run count was really chosen."""
    return sum(table[k + 1] * count for k, count in enumerate(run_counts)) / sum(run_counts)
