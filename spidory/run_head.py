"""Train the run neuron last, on a frozen model.

For each held-out sequence, the target is the smallest run count whose loss is close to the
best count's loss. Training after everything else avoids collapse to always using the maximum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import DATASETS, Corpus
from .model import amp
from .train import load, pick_device, save


def best_run_targets(losses: torch.Tensor, tolerance: float) -> torch.Tensor:
    """losses is (sequences, max_runs). Returns run counts, starting at 1."""
    limit = losses.min(1, keepdim=True).values * (1 + tolerance)
    return (losses <= limit).float().argmax(1) + 1


@torch.no_grad()
def collect(model, corpus: Corpus, windows: int, chunk: int = 64):
    """Pooled input, chosen expert and per-run loss for each held-out sequence."""
    model.eval()
    device = next(model.parameters()).device
    pooled, experts, losses = [], [], []
    for name in DATASETS:
        for row in corpus.fixed_windows(name, "val", windows).to(device).split(chunk):
            inputs = row[:, :-1]
            with amp(device):
                pooled.append(model._encode(inputs)[1].float())
            per_run = []
            for k in range(1, model.cfg.max_runs + 1):
                logits, chosen, _ = model.infer(inputs, k=k)
                per_run.append(F.cross_entropy(logits.flatten(0, 1).float(), row[:, 1:].flatten(),
                                               reduction="none").view(row.size(0), -1).mean(1))
            experts.append(chosen)
            losses.append(torch.stack(per_run, 1))
    return torch.cat(pooled), torch.cat(experts), torch.cat(losses)


def train_run_head(model, corpus: Corpus, windows: int = 2000, steps: int = 3000, tolerance: float = 0.005,
                   lr: float = 1e-3, seed: int = 0) -> dict:
    """Trains only the run head. Everything else stays exactly as it was."""
    pooled, experts, losses = collect(model, corpus, windows)
    targets = best_run_targets(losses, tolerance) - 1
    count = len(targets)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).to(targets.device)
    cut = max(1, count // 10)
    held, fit = order[:cut], order[cut:]
    share = torch.bincount(targets[fit], minlength=model.cfg.max_runs).float().clamp(min=1)
    weights = share.sum() / share / model.cfg.max_runs

    optimizer = torch.optim.AdamW(model.run_head.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        rows = fit[torch.randint(len(fit), (min(256, len(fit)),)).to(fit.device)]
        logits = model.run_logits(pooled[rows])[torch.arange(len(rows)), experts[rows]]
        loss = F.cross_entropy(logits, targets[rows], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    model.run_head_ready.fill_(True)
    with torch.no_grad():
        picks = model.run_logits(pooled[held])[torch.arange(len(held)), experts[held]].argmax(-1)
    return {"sequences": count, "target_counts": torch.bincount(targets + 1, minlength=model.cfg.max_runs + 1)[1:].tolist(),
            "holdout_accuracy": round(float((picks == targets[held]).float().mean()), 4),
            "always_most_common": round(float(torch.bincount(targets[held]).max() / len(held)), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    device = pick_device(args.device)
    path = Path("runs") / args.name / "last.pt"
    model, optimizer, step, tcfg, state = load(path, device)
    if bool(model.run_head_ready):
        print("run head already trained", flush=True)
        return
    corpus = Corpus(args.data, model.cfg.seq_len, seed=args.seed)
    report = train_run_head(model, corpus, seed=args.seed)
    print(json.dumps(report), flush=True)
    (path.parent / "run_head.json").write_text(json.dumps(report, indent=2))
    save(path, model, optimizer, step, tcfg, corpus, state["extra"])


if __name__ == "__main__":
    main()
