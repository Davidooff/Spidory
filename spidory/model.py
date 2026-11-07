"""Looped transformer with a shared core and gated residual experts.

There is no run-count neuron. Run counts come from a manual `k`, or from an average `mean_runs`
given as an input value plus each expert's learned-by-rounds target.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def amp(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        split = lambda z: z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
        x = x + self.proj(out.transpose(1, 2).reshape(b, t, d))
        return x + self.ffn(self.norm2(x))


class Expert(nn.Module):
    """Small bottleneck residual. Starts at zero, so joining does not disturb the model."""

    def __init__(self, d: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.GELU(), nn.Linear(width, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def update_targets(targets: torch.Tensor, usage: torch.Tensor, group: list[int], raise_by: float,
                   mean: float, low: float = 1.0, high: float = 4.0) -> torch.Tensor:
    """Raise the group, then shift the others so the usage-weighted average equals `mean`."""
    share = usage / usage.sum()
    in_group = torch.zeros_like(targets, dtype=torch.bool)
    in_group[group] = True
    raised = targets.clone()
    raised[in_group] = (raised[in_group] + raise_by).clamp(low, high)

    def solve(mask: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, float]:
        lo, hi = -(high - low), high - low
        for _ in range(60):
            mid = (lo + hi) / 2
            trial = base.clone()
            trial[mask] = (trial[mask] - mid).clamp(low, high)
            if (share * trial).sum() > mean:
                lo = mid
            else:
                hi = mid
        trial = base.clone()
        trial[mask] = (trial[mask] - lo).clamp(low, high)
        return trial, float((share * trial).sum() - mean)

    result, gap = solve(~in_group, raised)
    if abs(gap) > 1e-4 and in_group.any():
        # The others hit their limit, so the group gives back what is left.
        result, gap = solve(in_group, result)
    return result


class Spidory(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.seq_len, d)
        self.input_block = nn.ModuleList(Block(d, cfg.n_heads) for _ in range(cfg.n_input))
        self.core = nn.ModuleList(Block(d, cfg.n_heads) for _ in range(cfg.n_core))
        self.output_block = nn.ModuleList(Block(d, cfg.n_heads) for _ in range(cfg.n_output))
        self.norm = nn.LayerNorm(d)
        # Shared layers initialise first, so with one seed Plain and expert models start equal.
        self.apply(self._init)
        groups = 1
        if cfg.use_experts:
            groups = cfg.n_experts
            self.experts = nn.ModuleList(Expert(d, cfg.expert_dim) for _ in range(groups))
            self.router = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, groups))
            # The run neuron: picks a run count per sequence and expert. Trained last, on a frozen model.
            self.run_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
                                          nn.Linear(d // 2, groups * cfg.max_runs))
            self.experts.apply(self._init)
            self.router.apply(self._init)
            self.run_head.apply(self._init)
            for expert in self.experts:
                nn.init.zeros_(expert.net[-1].weight)
                nn.init.zeros_(expert.net[-1].bias)
            self.expert_embed = nn.Embedding(groups, d)
            nn.init.zeros_(self.expert_embed.weight)
        # Tells the output block how many runs happened. Plain has it too, for a fair comparison.
        self.run_embed = nn.Embedding(cfg.max_runs + 1, d)
        nn.init.zeros_(self.run_embed.weight)
        self.register_buffer("run_targets", torch.full((groups,), cfg.target_mean))
        self.register_buffer("expert_usage", torch.full((groups,), 1.0 / groups))
        self.register_buffer("run_head_ready", torch.tensor(False))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tok(tokens) + self.pos(torch.arange(tokens.size(1), device=tokens.device))
        for block in self.input_block:
            h = block(h)
        # Attention is causal, so these states depend only on the first `prefix` tokens.
        return h, h[:, : self.cfg.prefix].mean(dim=1)

    def _unit(self, h: torch.Tensor) -> torch.Tensor:
        """Dory's per-token L2 normalization, scaled by sqrt(d) to suit these standard blocks."""
        return F.normalize(h, dim=-1, eps=1e-6) * self.cfg.d_model ** 0.5

    def _run_once(self, s, x, gates):
        """Dory's loop: u = N(s + x), then s = N(F(u)). Experts join inside F."""
        h = self._unit(s + x)
        for block in self.core:
            h = block(h)
        if gates is not None:
            grad_path = self.training and torch.is_grad_enabled()
            for e, expert in enumerate(self.experts):
                weight = gates[:, e]
                if grad_path or weight.any():  # zero gates still carry gradient
                    h = h + weight[:, None, None] * expert(h)
        return self._unit(h)

    def _decode(self, h: torch.Tensor, ks: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        """The output block is told the run count and which expert ran."""
        h = h + self.run_embed(ks)[:, None]
        if self.cfg.use_experts and (experts >= 0).all():
            h = h + self.expert_embed(experts)[:, None]
        for block in self.output_block:
            h = block(h)
        return self.norm(h) @ self.tok.weight.T

    def _loop(self, x, gates, ks: torch.Tensor) -> torch.Tensor:
        """TBPTL depth 1: earlier runs have no gradients, and only the last run trains.

        Gradients still reach the input block through x, added at the last run.
        """
        state = x.detach().clone()
        with torch.no_grad():
            for run in range(1, int(ks.max())):
                rows = (ks > run).nonzero().squeeze(1)
                sub_gates = None if gates is None else gates[rows]
                state[rows] = self._run_once(state[rows], x[rows], sub_gates)
        return self._run_once(state, x, gates)

    def forward(self, tokens: torch.Tensor, ks=1, tau: float = 1.0, experts_on: bool = True,
                gate_override: torch.Tensor | None = None) -> dict:
        """`ks` is an int, a per-sequence tensor, or a function of the chosen experts."""
        cfg = self.cfg
        size, device = tokens.size(0), tokens.device
        h0, pooled = self._encode(tokens)
        router_logits, gates = None, gate_override
        if cfg.use_experts and experts_on and gate_override is None:
            router_logits = self.router(pooled)
            if self.training:
                gates = F.gumbel_softmax(router_logits, tau=tau, hard=True)
            else:
                gates = F.one_hot(router_logits.argmax(-1), cfg.n_experts).to(h0.dtype)
        chosen = gates.argmax(-1) if gates is not None else torch.full((size,), -1, device=device)
        if callable(ks):
            ks = ks(chosen)
        elif isinstance(ks, int):
            ks = torch.full((size,), ks, device=device)
        h = self._loop(self._unit(h0), gates, ks)
        return {"logits": self._decode(h, ks, chosen), "gates": gates, "router_logits": router_logits, "ks": ks}

    def pick_k(self, experts: torch.Tensor, targets: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """A target between two integers picks one of them, chosen by u in [0, 1)."""
        t = targets[experts.clamp(min=0)].round(decimals=4)
        low = t.floor()
        return (low + (u < t - low).to(t.dtype)).long().clamp(1, self.cfg.max_runs)

    def targets_for(self, mean_runs: float) -> torch.Tensor:
        """Per-expert targets for a requested average, keeping the expert offsets."""
        return update_targets(self.run_targets, self.expert_usage, [], 0.0, mean_runs,
                              high=float(self.cfg.max_runs))

    def _row_noise(self, tokens: torch.Tensor) -> torch.Tensor:
        """Repeatable value per sequence, from its first tokens."""
        head = tokens[:, :8]
        weights = (torch.arange(head.size(1), device=tokens.device) + 1) * 31
        mixed = (((head + 1) * weights).sum(1) * 2654435761) % (1 << 32)
        return mixed.float() / (1 << 32)

    def run_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """Returns (batch, experts, max_runs)."""
        return self.run_head(pooled).view(pooled.size(0), self.cfg.n_experts, self.cfg.max_runs)

    def _choose(self, tokens, pooled, k, mean_runs, learned=False):
        cfg = self.cfg
        size = tokens.size(0)
        experts = (self.router(pooled).argmax(-1) if cfg.use_experts
                   else torch.full((size,), -1, device=tokens.device))
        if k is not None:
            return experts, torch.full((size,), k, device=tokens.device)
        if learned:
            return experts, self.run_logits(pooled)[torch.arange(size), experts].argmax(-1) + 1
        mean = cfg.target_mean if mean_runs is None else mean_runs
        return experts, self.pick_k(experts, self.targets_for(mean), self._row_noise(tokens))

    @torch.no_grad()
    def decide(self, tokens: torch.Tensor, k: int | None = None, mean_runs: float | None = None,
               learned: bool = False):
        """Hard choice per sequence: expert index (-1 if none) and run count."""
        with amp(tokens.device):
            pooled = self._encode(tokens)[1]
            return self._choose(tokens, pooled, k, mean_runs, learned)

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, k: int | None = None, mean_runs: float | None = None,
              learned: bool = False):
        """Real inference: one expert, stopping at the chosen run count. Groups equal choices."""
        was_training = self.training
        self.eval()
        device = tokens.device
        logits = torch.empty(*tokens.shape, self.cfg.vocab_size, device=device)
        with amp(device):
            h0, pooled = self._encode(tokens)
            experts, ks = self._choose(tokens, pooled, k, mean_runs, learned)
            for expert, count in torch.stack([experts, ks], 1).unique(dim=0).tolist():
                rows = ((experts == expert) & (ks == count)).nonzero().squeeze(1)
                gates = None
                if expert >= 0:
                    gates = torch.zeros(len(rows), self.cfg.n_experts, device=device, dtype=h0.dtype)
                    gates[:, expert] = 1.0
                x = self._unit(h0[rows])
                h = x
                for _ in range(count):
                    h = self._run_once(h, x, gates)
                logits[rows] = self._decode(h, ks[rows], experts[rows]).to(logits.dtype)
        self.train(was_training)
        return logits, experts, ks

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def balance_penalty(router_logits: torch.Tensor) -> torch.Tensor:
    """KL from average routing to an even share, so one expert cannot take everything."""
    usage = F.softmax(router_logits, -1).mean(0)
    return (usage * (usage.clamp(min=1e-9).log() + math.log(usage.numel()))).sum()


def compute_loss(out: dict, targets: torch.Tensor, balance_weight: float):
    """Takes model output and tokens only: there is no dataset argument by design."""
    ce = F.cross_entropy(out["logits"].flatten(0, 1).float(), targets.flatten())
    parts = {"ce": ce.item()}
    loss = ce
    if out["router_logits"] is not None:
        penalty = balance_penalty(out["router_logits"])
        parts["balance"] = penalty.item()
        loss = loss + balance_weight * penalty
    return loss, parts
