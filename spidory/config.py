"""Model, training and round settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 320
    n_heads: int = 8
    seq_len: int = 256
    n_input: int = 4
    n_core: int = 10
    n_output: int = 4
    n_experts: int = 4
    expert_dim: int = 120
    max_runs: int = 4
    target_mean: float = 2.5
    prefix: int = 32
    use_experts: bool = True
    active_experts: int = 1  # experts applied per sequence, weighted by the router
    arch: str = "ngpt"  # "ngpt": Dory-style normalized blocks, SwiGLU, RoPE; "prenorm": the v1-v2 blocks
    run_mode: str = "table"  # "table": run counts from targets; "joint": the run neuron picks them
    neuron: str = "net"  # "net": small network; "linear": one neuron per run count, per expert
    neuron_train: str = "sampled"  # "sampled", "average" (all 4 losses) or "mix" (state mixing)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    stage1_steps: int = 2100  # one run
    stage2_steps: int = 900  # sampled loops
    cycle_steps: int = 0  # stage 2b: run counts cycle 1, 2, 3, 4 across tokens
    heavy_pieces: int = 2  # memory-heavy steps (cycle, "average" neuron) split the batch into this many pieces
    steps: int = 5400  # whole schedule, including rounds
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_lr_steps: int = 200
    balance_penalty: float = 0.2
    tau_start: float = 1.0
    tau_end: float = 0.3
    one_run_share: float = 0.1  # stage 2 batches forced to one run
    explore: float = 0.2  # stage 3 share of uniform run counts
    neuron_weight: float = 5.0  # how strongly the neuron follows the loss differences between run counts
    run_penalty: float = 0.2  # two-sided penalty on each expert's average runs
    neuron_explore: float = 0.1  # share of sequences given a random run count, so all depths keep training
    eval_windows: int = 256
    log_every: int = 100
    save_every: int = 500
    keep_every: int = 20000
    seed: int = 0

    @property
    def experts_from(self) -> int:
        return self.stage1_steps // 4


@dataclass(frozen=True)
class RoundConfig:
    passes: int = 2
    steps: int = 300
    focus_share: float = 0.5
    lift: float = 1.5
    raise_by: float = 0.5
    others_tolerance: float = 0.01
    min_gain: float = 0.002


def to_dict(cfg) -> dict:
    return asdict(cfg)
