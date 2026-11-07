"""Stage 1 (one run) and stage 2 (sampled loops), with checkpoints and exact resume."""
from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig, to_dict
from .data import DATASETS, Corpus
from .joint import joint_step
from .model import Spidory, amp, compute_loss


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def compile_for_speed(model: Spidory) -> None:
    """Compiles each nGPT layer's forward, which fuses its many small operations.

    Only forward functions are replaced, so checkpoints keep the same names.
    """
    if not getattr(model, "ngpt", False) or next(model.parameters()).device.type != "cuda":
        return
    from .ngpt import NBlock

    for module in model.modules():
        if isinstance(module, NBlock):
            module.forward = torch.compile(module.forward, dynamic=True)


def make_optimizer(model: Spidory, cfg: TrainConfig) -> torch.optim.AdamW:
    # nGPT keeps weights on the unit sphere, so it uses no weight decay.
    decay = 0.0 if getattr(model, "ngpt", False) else cfg.weight_decay
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=decay)


def tau_at(step: int, cfg: TrainConfig) -> float:
    """Sharpens routing from the moment experts join."""
    span = max(1, cfg.steps - cfg.experts_from)
    t = min(1.0, max(0.0, (step - cfg.experts_from) / span))
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * t


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_lr_steps:
        return cfg.lr * (step + 1) / cfg.warmup_lr_steps
    t = min(1.0, step / max(1, cfg.steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def cycle_ks(size: int, tokens: int, top: int, device) -> torch.Tensor:
    """Each next token gets one more run, back to 1 after the maximum: 1, 2, 3, 4, 1, ...

    A random start per sequence lets every position see every run count.
    """
    offset = torch.randint(top, (size, 1))
    return ((torch.arange(tokens)[None] + offset) % top + 1).to(device)


def stage3_start(cfg: TrainConfig) -> int:
    return cfg.stage1_steps + cfg.stage2_steps + cfg.cycle_steps


def sample_ks(step: int, cfg: TrainConfig, model: Spidory, experts: torch.Tensor, tokens: int = 0):
    """Run counts for one batch, by stage: per sequence, or per token in the cycle stage."""
    size, device, top = experts.numel(), experts.device, model.cfg.max_runs
    if step < cfg.stage1_steps:
        return torch.ones(size, dtype=torch.long, device=device)
    if step < cfg.stage1_steps + cfg.stage2_steps:
        if torch.rand(1).item() < cfg.one_run_share:
            return torch.ones(size, dtype=torch.long, device=device)
        return torch.randint(1, top + 1, (size,)).to(device)
    if step < stage3_start(cfg):
        return cycle_ks(size, tokens, top, device)
    ks = model.pick_k(experts, model.run_targets, torch.rand(size).to(device))
    explore = torch.rand(size).to(device) < cfg.explore
    return torch.where(explore, torch.randint(1, top + 1, (size,)).to(device), ks)


def train_steps(model, optimizer, corpus: Corpus, cfg: TrainConfig, start: int, stop: int,
                focus: str | None = None, focus_share: float = 0.5, log=None, on_save=None) -> None:
    model.train()
    device = next(model.parameters()).device
    began = time.time()
    for step in range(start, stop):
        batch = corpus.train_batch(cfg.batch_size, focus, focus_share).to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        stage3 = step >= stage3_start(cfg)
        joint = stage3 and model.cfg.use_experts and model.cfg.run_mode == "joint"
        heavy = (cfg.stage1_steps + cfg.stage2_steps <= step < stage3_start(cfg)) or (
            joint and model.cfg.neuron_train == "average")
        # Memory-heavy steps keep several runs with gradients, so they go in pieces whose
        # gradients add up to one update over the whole batch.
        pieces = batch.split(max(1, -(-batch.size(0) // cfg.heavy_pieces))) if heavy else (batch,)
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for piece in pieces:
            with amp(device):
                if joint:
                    loss, parts = joint_step(model, piece[:, :-1], piece[:, 1:], cfg, tau_at(step, cfg))
                else:
                    out = model(piece[:, :-1], ks=lambda experts: sample_ks(step, cfg, model, experts, piece.size(1) - 1),
                                tau=tau_at(step, cfg), experts_on=step >= cfg.experts_from)
                    loss, parts = compute_loss(out, piece[:, 1:], cfg.balance_penalty)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at step {step}; stopping instead of wasting compute")
            (loss * piece.size(0) / batch.size(0)).backward()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * piece.size(0) / batch.size(0)
        parts = totals
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        model.normalize_weights()
        done = step + 1
        if log and (done % cfg.log_every == 0 or done == stop):
            rate = (done - start) * cfg.batch_size * model.cfg.seq_len / (time.time() - began)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            log({"step": done, "tok_per_s": round(rate), "peak_rss_mb": peak,
                 **{k: round(v, 4) for k, v in parts.items()}})
        if on_save and done % cfg.save_every == 0:
            on_save(done)


@torch.no_grad()
def heldout_loss(model: Spidory, corpus: Corpus, split: str, windows: int, k: int | None = None,
                 mean_runs: float | None = None, chunk: int = 32, learned: bool = False) -> dict[str, float]:
    """Mean next-token loss per dataset on the real inference path."""
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for name in DATASETS:
        batch = corpus.fixed_windows(name, split, windows).to(device)
        total = 0.0
        for row in batch.split(chunk):
            logits = model.infer(row[:, :-1], k=k, mean_runs=mean_runs, learned=learned)[0]
            total += F.cross_entropy(logits.flatten(0, 1).float(), row[:, 1:].flatten(),
                                     reduction="sum").item()
        result[name] = total / (windows * model.cfg.seq_len)
    return result


def rng_state(corpus: Corpus) -> dict:
    state = {"torch": torch.get_rng_state(), "python": json.dumps(random.getstate()),
             "corpus": json.dumps(corpus.rng.bit_generator.state)}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict, corpus: Corpus) -> None:
    torch.set_rng_state(state["torch"].cpu())
    version, internal, gauss = json.loads(state["python"])
    random.setstate((version, tuple(internal), gauss))
    corpus.rng.bit_generator.state = json.loads(state["corpus"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save(path: Path, model, optimizer, step: int, tcfg: TrainConfig, corpus: Corpus,
         extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "model_cfg": to_dict(model.cfg), "train_cfg": to_dict(tcfg),
                "rng": rng_state(corpus), "extra": extra or {}}, tmp)
    tmp.rename(path)


def load(path: Path, device: torch.device):
    """Checkpoints are device independent: everything maps to `device` on load."""
    state = torch.load(path, weights_only=True, map_location=device)
    tcfg = TrainConfig(**state["train_cfg"])
    # Checkpoints from before nGPT have no "arch" and use the old blocks.
    model = Spidory(ModelConfig(**{"arch": "prenorm", **state["model_cfg"]})).to(device)
    # Checkpoints from before the single-layer neuron existed get a fresh one.
    for key, value in model.state_dict().items():
        if key.startswith("run_neuron."):
            state["model"].setdefault(key, value)
    state["model"].setdefault("run_head_ready", torch.tensor(False, device=device))
    model.load_state_dict(state["model"])
    optimizer = make_optimizer(model, tcfg)
    optimizer.load_state_dict(state["optimizer"])
    return model, optimizer, state["step"], tcfg, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--plain", action="store_true", help="Dory only, no experts")
    parser.add_argument("--stage1", type=int, default=TrainConfig.stage1_steps)
    parser.add_argument("--stage2", type=int, default=TrainConfig.stage2_steps)
    parser.add_argument("--cycle", type=int, default=TrainConfig.cycle_steps,
                        help="stage 2b steps, where run counts cycle 1-4 across tokens")
    parser.add_argument("--total", type=int, default=TrainConfig.steps,
                        help="length of the shared schedule, including rounds")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=ModelConfig.seq_len)
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--expert-dim", type=int, default=ModelConfig.expert_dim)
    parser.add_argument("--arch", choices=["ngpt", "prenorm"], default=ModelConfig.arch)
    parser.add_argument("--active-experts", type=int, default=ModelConfig.active_experts)
    parser.add_argument("--between-layers", type=int, default=ModelConfig.between_layers)
    parser.add_argument("--between-mode", choices=["shared", "per_gap"], default=ModelConfig.between_mode)
    parser.add_argument("--between-merge", action="store_true", help="merge the input inside the between block")
    parser.add_argument("--no-boundary-norm", action="store_true", help="no normalization at loop boundaries")
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--warmup", type=int, default=TrainConfig.warmup_lr_steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "last.pt"
    corpus = Corpus(args.data, args.seq_len, seed=args.seed)
    stop = args.stage1 + args.stage2 + args.cycle

    if ckpt.exists():
        model, optimizer, start, cfg, state = load(ckpt, device)
        set_rng_state(state["rng"], corpus)
    else:
        cfg = TrainConfig(stage1_steps=args.stage1, stage2_steps=args.stage2, cycle_steps=args.cycle,
                          steps=args.total,
                          batch_size=args.batch_size, seed=args.seed, lr=args.lr,
                          warmup_lr_steps=args.warmup)
        model = Spidory(ModelConfig(seq_len=args.seq_len, use_experts=not args.plain, d_model=args.d_model,
                                    expert_dim=args.expert_dim, arch=args.arch,
                                    active_experts=args.active_experts, between_layers=args.between_layers,
                                    between_mode=args.between_mode, between_merge=args.between_merge,
                                    boundary_norm=not args.no_boundary_norm)).to(device)
        optimizer, start = make_optimizer(model, cfg), 0
    compile_for_speed(model)
    print(f"{args.name}: {model.num_params():,} params on {device}, step {start} of {stop}", flush=True)
    if start >= stop:
        return

    def log(row: dict) -> None:
        print(json.dumps(row), flush=True)
        with open(run / "log.jsonl", "a") as sink:
            sink.write(json.dumps(row) + "\n")

    def on_save(done: int) -> None:
        save(ckpt, model, optimizer, done, cfg, corpus)
        if done % cfg.keep_every == 0:
            save(run / f"step_{done}.pt", model, optimizer, done, cfg, corpus)

    train_steps(model, optimizer, corpus, cfg, start, stop, log=log, on_save=on_save)
    save(ckpt, model, optimizer, stop, cfg, corpus)


if __name__ == "__main__":
    main()
