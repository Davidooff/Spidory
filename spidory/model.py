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
from .ngpt import NBlock, NExpert, Scale, normalize_weights, unit


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
        self.ngpt = cfg.arch == "ngpt"
        layer = (lambda: NBlock(d, cfg.n_heads)) if self.ngpt else (lambda: Block(d, cfg.n_heads))
        self.tok = nn.Embedding(cfg.vocab_size, d)
        if self.ngpt:
            self.s_z = Scale(cfg.vocab_size, 1.0, 1 / math.sqrt(d))  # RoPE replaces positions
        else:
            self.pos = nn.Embedding(cfg.seq_len, d)
            self.norm = nn.LayerNorm(d)
        self.input_block = nn.ModuleList(layer() for _ in range(cfg.n_input))
        self.core = nn.ModuleList(layer() for _ in range(cfg.n_core))
        self.output_block = nn.ModuleList(layer() for _ in range(cfg.n_output))
        if cfg.between_layers > 0:
            gaps = 1 if cfg.between_mode == "shared" else cfg.max_runs - 1
            self.between = nn.ModuleList(nn.ModuleList(layer() for _ in range(cfg.between_layers))
                                         for _ in range(gaps))
            if cfg.between_merge:
                # Learned merge of state and input, starting as the plain sum s + x.
                self.merge_state = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(gaps))
                self.merge_input = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(gaps))
        elif cfg.between_merge:
            raise ValueError("between_merge needs a between block (between_layers > 0)")
        # Shared layers initialise first, so with one seed Plain and expert models start equal.
        self.apply(self._init)
        if cfg.between_merge:
            for layer_ in [*self.merge_state, *self.merge_input]:
                nn.init.eye_(layer_.weight)  # after the general init, so the merge starts as s + x
        groups = 1
        if cfg.use_experts:
            groups = cfg.n_experts
            expert = (lambda: NExpert(d, cfg.expert_dim)) if self.ngpt else (lambda: Expert(d, cfg.expert_dim))
            self.experts = nn.ModuleList(expert() for _ in range(groups))
            self.router = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, groups))
            # The run neuron: picks a run count per sequence and expert. Trained last, on a frozen model.
            self.run_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
                                          nn.Linear(d // 2, groups * cfg.max_runs))
            # The single-layer version: one neuron per run count, per expert.
            self.run_neuron = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, groups * cfg.max_runs))
            self.experts.apply(self._init)
            self.router.apply(self._init)
            self.run_head.apply(self._init)
            self.run_neuron.apply(self._init)
            if not self.ngpt:  # nGPT experts start silent through their step size instead
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
        self.normalize_weights()

    def normalize_weights(self) -> None:
        """nGPT keeps weight rows at unit length; called after every optimizer step."""
        if self.ngpt:
            normalize_weights(self, self.tok)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tok(tokens)
        if not self.ngpt:
            h = h + self.pos(torch.arange(tokens.size(1), device=tokens.device))
        for block in self.input_block:
            h = block(h)
        # Attention is causal, so these states depend only on the first `prefix` tokens.
        return h, h[:, : self.cfg.prefix].mean(dim=1)

    def _unit(self, h: torch.Tensor) -> torch.Tensor:
        """Dory's per-token L2 normalization: unit length with nGPT, sqrt(d) for the old blocks."""
        return F.normalize(h, dim=-1, eps=1e-6) * (1.0 if self.ngpt else self.cfg.d_model ** 0.5)

    _route_tau = 0.3  # set per step in training
    _loop_router_logits: list | None = None  # per-loop routing with gradients, for the routing losses
    _route_record: list | None = None  # when a list, each run's chosen experts are appended
    _route_locked = False  # when True, per-loop routing keeps the experts chosen at the start

    def _route_from_state(self, s: torch.Tensor, gates: torch.Tensor, run) -> torch.Tensor:
        """With route_per_loop, runs after the first pick experts from the state entering them."""
        later = torch.as_tensor(run, device=s.device)
        later = (later > 1).expand(len(s)) if later.dim() == 0 else later > 1
        if not later.any():
            return gates
        logits = self.router(s[:, : self.cfg.prefix].mean(dim=1))
        routed = self.route(logits, self._route_tau if self.training else self.route_tau, self.training)
        if self.training and torch.is_grad_enabled() and self._loop_router_logits is not None:
            self._loop_router_logits.append((logits[later], routed[later]))
        return torch.where(later[:, None], routed.to(gates.dtype), gates)

    def _run_once(self, s, x, gates, run=1):
        """Dory's loop: u = N(s + x), then s = N(F(u)). Experts join inside F.

        With a merging between block, the input was already merged in, so u = s.
        `run` (an int, or one value per row) counts from 1; per-loop routing uses it.
        """
        if gates is not None and self.cfg.route_per_loop and not self._route_locked:
            gates = self._route_from_state(s, gates, run)
        if gates is not None and self._route_record is not None:
            self._route_record.append(gates.argmax(-1).detach().cpu())
        merged = self.cfg.between_merge
        h = s if merged else s + x
        if self.cfg.boundary_norm:
            h = self._unit(h)
        for block in self.core:
            h = block(h)
        if gates is not None:
            grad_path = self.training and torch.is_grad_enabled()
            for e, expert in enumerate(self.experts):
                weight = gates[:, e]
                if grad_path or weight.any():  # zero gates still carry gradient
                    if self.ngpt:
                        step = weight[:, None, None] * expert.alpha()
                        h = unit(h + step * (expert.proposal(h) - h))
                    else:
                        h = h + weight[:, None, None] * expert(h)
        return self._unit(h) if self.cfg.boundary_norm else h

    def _decode(self, h: torch.Tensor, ks: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        """The output block is told the run count and which expert ran."""
        runs = self.run_embed(ks.clamp(max=self.cfg.max_runs))  # beyond training: the last count
        h = h + (runs if ks.dim() == 2 else runs[:, None])  # one count per token, or per sequence
        if self.cfg.use_experts and (experts >= 0).all():
            h = h + self.expert_embed(experts)[:, None]
        if self.ngpt:
            h = unit(h)
        for block in self.output_block:
            h = block(h)
        if self.ngpt:
            return (h @ self.tok.weight.T) * self.s_z()
        return self.norm(h) @ self.tok.weight.T

    @property
    def has_between(self) -> bool:
        return self.cfg.between_layers > 0

    def _between(self, s: torch.Tensor, gap: int, x: torch.Tensor) -> torch.Tensor:
        """The between block after centre run `gap` (counting from 1).

        With between_merge, it first merges state and input with learned weights.
        """
        index = 0 if self.cfg.between_mode == "shared" else min(gap, self.cfg.max_runs - 1) - 1
        if self.cfg.between_merge:
            s = self.merge_state[index](s) + self.merge_input[index](x)
            if self.cfg.boundary_norm:
                s = self._unit(s)
        for block in self.between[index]:
            s = block(s)
        return self._unit(s) if self.cfg.boundary_norm else s

    def _enter_last(self, raw: torch.Tensor, ks: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """State entering each sequence's last centre run.

        raw is the state after run k-1. The between block for gap k-1 runs here with gradients,
        so between blocks learn: in per-gap mode, gap g learns when k = g + 1.
        """
        if not self.has_between or not (ks > 1).any():
            return raw
        start = raw.clone()
        for k in ks.unique().tolist():
            if k >= 2:
                rows = (ks == k).nonzero().squeeze(1)
                start[rows] = self._between(raw[rows], k - 1, x[rows])
        return start

    def _loop_tokens(self, x, gates, ks: torch.Tensor) -> torch.Tensor:
        """Per-token run counts: ks is (batch, tokens).

        Every run passes all tokens through the core, because attention needs them. A token
        that has finished keeps its state, so later tokens can still read it. Each token learns
        from its own last run only (TBPTL depth 1 per token): run inputs are detached.
        """
        grad = torch.is_grad_enabled()
        top = int(ks.max())
        state, out = x, x
        for run in range(1, top + 1):
            # With a between block, the run input keeps the gradient of the block just before it.
            entering = state if (self.has_between and run > 1) or not grad else state.detach()
            new = self._run_once(entering, x, gates, run)
            out = torch.where((ks == run)[:, :, None], new, out)
            kept = new.detach() if grad else new
            state = torch.where((ks >= run)[:, :, None], kept, state.detach() if grad else state)
            if self.has_between and run < top:
                moved = self._between(state, run, x)
                state = torch.where((ks > run)[:, :, None], moved, state)
        return out

    def _loop(self, x, gates, ks: torch.Tensor) -> torch.Tensor:
        """TBPTL depth 1: earlier runs have no gradients, and only the last run trains.

        Gradients still reach the input block through x, added at the last run.
        """
        state = x.detach().clone()
        with torch.no_grad():
            for run in range(1, int(ks.max())):
                rows = (ks > run).nonzero().squeeze(1)
                sub_gates = None if gates is None else gates[rows]
                state[rows] = self._run_once(state[rows], x[rows], sub_gates, run)
                if self.has_between:
                    # The between block before each last run is applied later, with gradients.
                    later = (ks > run + 1).nonzero().squeeze(1)
                    if len(later):
                        state[later] = self._between(state[later], run, x[later])
        return self._run_once(self._enter_last(state, ks, x), x, gates, ks)

    def forward(self, tokens: torch.Tensor, ks=1, tau: float = 1.0, experts_on: bool = True,
                gate_override: torch.Tensor | None = None) -> dict:
        """`ks` is an int, a per-sequence tensor, or a function of the chosen experts."""
        cfg = self.cfg
        size, device = tokens.size(0), tokens.device
        h0, pooled = self._encode(tokens)
        router_logits, gates = None, gate_override
        self._route_tau = tau
        self._loop_router_logits = [] if self.training else None
        if cfg.use_experts and experts_on and gate_override is None:
            router_logits = self.router(pooled)
            gates = self.route(router_logits, tau if self.training else self.route_tau, self.training).to(h0.dtype)
        chosen = gates.argmax(-1) if gates is not None else torch.full((size,), -1, device=device)
        if callable(ks):
            ks = ks(chosen)
        elif isinstance(ks, int):
            ks = torch.full((size,), ks, device=device)
        loop = self._loop_tokens if ks.dim() == 2 else self._loop
        h = loop(self._unit(h0), gates, ks)
        loop_routing, self._loop_router_logits = self._loop_router_logits or [], None
        return {"logits": self._decode(h, ks, chosen), "gates": gates, "router_logits": router_logits, "ks": ks,
                "loop_routing": loop_routing}

    route_tau = 0.3  # routing temperature at test time: where training's sharpening ends

    def route(self, router_logits: torch.Tensor, tau: float, training: bool) -> torch.Tensor:
        """Expert weights per sequence, summing to 1, with `active_experts` of them above zero.

        One active expert uses straight-through Gumbel sampling, as before. Several use the
        top choices of noisy logits, weighted by the router's softmax over just those.
        """
        top = self.cfg.active_experts
        if top <= 1:
            if training:
                return F.gumbel_softmax(router_logits, tau=tau, hard=True)
            return F.one_hot(router_logits.argmax(-1), self.cfg.n_experts).to(router_logits.dtype)
        scores = router_logits
        if training:
            noise = -torch.log(-torch.log(torch.rand_like(router_logits).clamp(1e-9, 1 - 1e-9)))
            scores = router_logits + noise * tau
        chosen = scores.topk(top, dim=-1).indices
        mask = torch.zeros_like(router_logits, dtype=torch.bool).scatter_(-1, chosen, True)
        weights = F.softmax((router_logits / tau).masked_fill(~mask, float("-inf")), dim=-1)
        return weights

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
        head = self.run_neuron if self.cfg.neuron == "linear" else self.run_head
        return head(pooled).view(pooled.size(0), self.cfg.n_experts, self.cfg.max_runs)

    def _choose(self, tokens, pooled, k, mean_runs, learned=False, expert=None):
        cfg = self.cfg
        size = tokens.size(0)
        if expert is not None:
            if not cfg.use_experts or not 0 <= expert < cfg.n_experts:
                raise ValueError(f"expert must be 0 to {cfg.n_experts - 1}")
            experts = torch.full((size,), expert, device=tokens.device)
        else:
            experts = (self.router(pooled).argmax(-1) if cfg.use_experts
                       else torch.full((size,), -1, device=tokens.device))
        if k is not None:
            return experts, torch.full((size,), k, device=tokens.device)
        if learned and cfg.neuron_train == "mix":
            # The mix variant blends every run's state, so it always computes all of them.
            return experts, torch.full((size,), cfg.max_runs, device=tokens.device)
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
              learned: bool = False, expert: int | None = None):
        """Real inference: stops at the chosen run count. Groups sequences by top expert and count.

        `expert` locks every sequence to that one expert, in every run, instead of the router.
        `k` may go past max_runs to test extrapolation: later runs reuse the last run embedding
        and, per gap, the last between block.
        """
        was_training = self.training
        self.eval()
        device = tokens.device
        logits = torch.empty(*tokens.shape, self.cfg.vocab_size, device=device)
        with amp(device):
            h0, pooled = self._encode(tokens)
            experts, ks = self._choose(tokens, pooled, k, mean_runs, learned, expert)
            if expert is not None:
                all_gates = F.one_hot(experts, self.cfg.n_experts).to(h0.dtype)
            elif self.cfg.use_experts:
                all_gates = self.route(self.router(pooled), self.route_tau, False).to(h0.dtype)
            else:
                all_gates = None
            self._route_locked = expert is not None
            mix = learned and self.cfg.neuron_train == "mix" and k is None and mean_runs is None
            for expert, count in torch.stack([experts, ks], 1).unique(dim=0).tolist():
                rows = ((experts == expert) & (ks == count)).nonzero().squeeze(1)
                gates = None if expert < 0 else all_gates[rows]
                x = self._unit(h0[rows])
                h = x
                states = []
                for run in range(1, count + 1):
                    h = self._run_once(h, x, gates, run)
                    states.append(h)  # after the centre run, before any between block, as in training
                    if self.has_between and run < count:
                        h = self._between(h, run, x)
                shown = ks[rows]
                if mix:
                    # Test the way it trains: the neuron-weighted mix of every run's state.
                    probs = F.softmax(self.run_logits(pooled[rows])[:, expert].float(), dim=-1).to(h.dtype)
                    h = (probs[:, :, None, None] * torch.stack(states, 1)).sum(1)
                    shown = probs.argmax(-1) + 1
                logits[rows] = self._decode(h, shown, experts[rows]).to(logits.dtype)
        self._route_locked = False
        self.train(was_training)
        return logits, experts, ks

    @torch.no_grad()
    def infer_tokens(self, tokens: torch.Tensor, ks: torch.Tensor):
        """Inference with a run count per token, such as the training cycle."""
        was_training = self.training
        self.eval()
        with amp(tokens.device):
            logits = self(tokens, ks=ks)["logits"].float()
        self.train(was_training)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def run_penalty(run_probs: torch.Tensor, gates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Two-sided: punishes each expert's average runs for being above or below its target.

    run_probs is (batch, max_runs) for each sequence's chosen expert; gates is (batch, experts).
    """
    counts = torch.arange(1, run_probs.size(-1) + 1, device=run_probs.device, dtype=run_probs.dtype)
    expected = (run_probs * counts).sum(-1)
    weights = gates.detach().to(expected.dtype)
    usage = weights.sum(0)
    means = (weights * expected[:, None]).sum(0) / usage.clamp(min=1e-6)
    share = usage / usage.sum().clamp(min=1e-6)
    return (share * (means - targets.to(means.dtype)) ** 2).sum()


def balance_penalty(router_logits: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Switch Transformer load balancing: experts x sum(share of sequences sent x mean probability).

    Lowest (1.0) when sequences are spread evenly. The earlier loss balanced mean probabilities,
    which a router satisfies by giving every expert 25% on every input: it stopped deciding, and
    every sequence went to one expert. This one balances where sequences actually go.
    """
    probs = F.softmax(router_logits.float(), -1)
    chosen = (gates.detach() > 0).float()
    share = chosen.mean(0) / chosen.sum(-1).mean().clamp(min=1.0)
    return probs.size(-1) * (share * probs.mean(0)).sum()


def routing_uncertainty(router_logits: torch.Tensor) -> torch.Tensor:
    """Mean routing entropy per input, from 0 (decided) to 1 (every expert equal)."""
    probs = F.softmax(router_logits.float(), -1)
    entropy = -(probs * probs.clamp(min=1e-9).log()).sum(-1)
    return entropy.mean() / math.log(probs.size(-1))


def routing_losses(router_logits, gates, loop_routing, cfg) -> tuple[torch.Tensor, dict]:
    """Balance and sharpening, over the first routing and every per-loop routing."""
    pairs = [(router_logits, gates)] + list(loop_routing or [])
    balance = sum(balance_penalty(l, g) for l, g in pairs) / len(pairs)
    sharpen = sum(routing_uncertainty(l) for l, _ in pairs) / len(pairs)
    loss = cfg.balance_penalty * balance + cfg.router_sharpen * sharpen
    return loss, {"balance": balance.item(), "router_uncertainty": sharpen.item()}


def compute_loss(out: dict, targets: torch.Tensor, cfg):
    """Takes model output and tokens only: there is no dataset argument by design."""
    ce = F.cross_entropy(out["logits"].flatten(0, 1).float(), targets.flatten())
    parts = {"ce": ce.item()}
    loss = ce
    if out["router_logits"] is not None:
        routing, found = routing_losses(out["router_logits"], out["gates"], out.get("loop_routing"), cfg)
        parts.update(found)
        loss = loss + routing
    return loss, parts
