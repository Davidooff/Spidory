"""Stage 3: raise run targets for active experts that gain from it, keep only what helps.

Each round trains two branches with identical steps and seeds, differing only in the targets.
Comparing them removes the bias of focused training. With --control-only the targets never
change: that path produces the Plain and Constant runs.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RoundConfig
from .data import DATASETS, Corpus
from .model import update_targets  # noqa: F401  (part of this module's public surface)
from .train import heldout_loss, load, pick_device, save, set_rng_state, train_steps


@torch.no_grad()
def measure_activation(model, corpus: Corpus, split: str, windows: int) -> dict[str, torch.Tensor]:
    """Share of each dataset's sequences routed to each expert."""
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for name in DATASETS:
        batch = corpus.fixed_windows(name, split, windows).to(device)
        choices = torch.cat([model.decide(row[:, :-1])[0] for row in batch.split(32)])
        result[name] = torch.bincount(choices, minlength=model.cfg.n_experts).float().cpu() / windows
    return result


@torch.no_grad()
def loss_table(model, corpus: Corpus, split: str, windows: int) -> dict[str, torch.Tensor]:
    """For each dataset: mean loss of (expert, k), or NaN where the expert saw no sequences."""
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    tables = {}
    for name in DATASETS:
        batch = corpus.fixed_windows(name, split, windows).to(device)
        total = torch.zeros(cfg.n_experts, cfg.max_runs)
        seen = torch.zeros(cfg.n_experts)
        for row in batch.split(32):
            for k in range(1, cfg.max_runs + 1):
                logits, experts, _ = model.infer(row[:, :-1], k=k)
                each = F.cross_entropy(logits.flatten(0, 1).float(), row[:, 1:].flatten(),
                                       reduction="none").view(row.size(0), -1).mean(1).cpu()
                total[:, k - 1].index_add_(0, experts.cpu(), each)
                if k == 1:
                    seen.index_add_(0, experts.cpu(), torch.ones(len(each)))
        tables[name] = total / seen[:, None]
    return tables


def overall_usage(activation: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(activation.values())).mean(0)


def gain_from_more_runs(table: torch.Tensor, expert: int, target: float) -> float:
    """Relative loss gain from going above the current target's lower run count."""
    current = int(target)
    if current >= table.size(1):
        return 0.0
    base, higher = table[expert, current - 1], table[expert, current:].min()
    if base.isnan() or higher.isnan():
        return 0.0
    return float((base - higher) / base)


def choose_group(activation: dict[str, torch.Tensor], table: torch.Tensor, targets: torch.Tensor,
                 dataset: str, lift: float, min_gain: float) -> list[int]:
    """Experts unusually active on this dataset that also gain from more runs."""
    overall = overall_usage(activation)
    lifts = activation[dataset] / overall.clamp(min=1e-6)
    ranked = lifts.argsort(descending=True).tolist()
    group = [e for e in ranked if lifts[e] > lift
             and gain_from_more_runs(table, e, float(targets[e])) >= min_gain]
    return group[: overall.numel() // 2]


def snapshot(model, optimizer) -> dict:
    return copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict()})


def restore(model, optimizer, state: dict) -> None:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def keep_change(changed: dict[str, float], control: dict[str, float], dataset: str, tolerance: float,
                min_gain: float = 0.002) -> bool:
    """Needs a real gain over control: tiny differences are noise, not evidence."""
    helps = changed[dataset] < control[dataset] * (1 - min_gain)
    safe = all(changed[n] <= control[n] * (1 + tolerance) for n in changed if n != dataset)
    return helps and safe


def run_rounds(model, optimizer, corpus: Corpus, tcfg, rcfg: RoundConfig, start: int, seed: int,
               control_only: bool, log, resume: tuple[int, int] = (0, 0), on_round=None) -> int:
    """Runs from `resume` = (pass, dataset index). Returns the final step."""
    step = start
    schedule = [(p, i) for p in range(rcfg.passes) for i in range(len(DATASETS))]
    for number, index in schedule[schedule.index(resume) if resume in schedule else len(schedule):]:
        dataset = DATASETS[index]
        round_seed = seed + 1000 * (number + 1) + index
        stop = step + rcfg.steps
        windows = tcfg.eval_windows

        def branch() -> dict[str, float]:
            torch.manual_seed(round_seed)
            corpus.reseed(round_seed)
            train_steps(model, optimizer, corpus, tcfg, step, stop, dataset, rcfg.focus_share)
            return heldout_loss(model, corpus, "val", windows)

        group: list[int] = []
        if model.cfg.use_experts:
            activation = measure_activation(model, corpus, "val", windows)
            model.expert_usage.copy_(overall_usage(activation).to(model.expert_usage))
            if not control_only:
                table = loss_table(model, corpus, "val", windows)[dataset]
                group = choose_group(activation, table, model.run_targets.cpu(), dataset,
                                     rcfg.lift, rcfg.min_gain)
        record = {"pass": number, "dataset": dataset, "group": group, "kept": False}

        if group:
            before = snapshot(model, optimizer)
            model.run_targets.copy_(update_targets(
                model.run_targets, model.expert_usage, group, rcfg.raise_by,
                model.cfg.target_mean, high=float(model.cfg.max_runs)))
            changed = branch()
            after = snapshot(model, optimizer)
            restore(model, optimizer, before)
            control = branch()
            record.update(changed=changed, control=control)
            if keep_change(changed, control, dataset, rcfg.others_tolerance, rcfg.min_gain):
                restore(model, optimizer, after)
                record["kept"] = True
        else:
            record["control"] = branch()

        step = stop
        record["step"] = step
        record["targets"] = [round(x, 3) for x in model.run_targets.tolist()]
        log(record)
        if on_round:
            following = number + (index + 1) // len(DATASETS), (index + 1) % len(DATASETS)
            on_round(step, following)
    return step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--warm", required=True, help="stage 2 run to start from")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--passes", type=int, default=RoundConfig.passes)
    parser.add_argument("--round-steps", type=int, default=RoundConfig.steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "last.pt"
    resumed = ckpt.exists()
    model, optimizer, start, tcfg, state = load(ckpt if resumed else Path("runs") / args.warm / "last.pt", device)
    rcfg = replace(RoundConfig(), steps=args.round_steps, passes=args.passes)
    corpus = Corpus(args.data, model.cfg.seq_len, seed=args.seed)
    resume = (0, 0)
    if resumed:
        set_rng_state(state["rng"], corpus)
        resume = tuple(state["extra"]["next_round"])
        print(f"{args.name}: resuming at pass {resume[0]}, dataset {resume[1]}, step {start}", flush=True)

    def log(row: dict) -> None:
        print(json.dumps(row), flush=True)
        with open(run / "rounds.jsonl", "a") as sink:
            sink.write(json.dumps(row) + "\n")

    def on_round(step: int, following: tuple[int, int]) -> None:
        save(ckpt, model, optimizer, step, tcfg, corpus, {"next_round": list(following)})

    step = run_rounds(model, optimizer, corpus, tcfg, rcfg, start, args.seed, args.control_only,
                      log, resume, on_round)
    print(json.dumps({"final_step": step}), flush=True)


if __name__ == "__main__":
    main()
