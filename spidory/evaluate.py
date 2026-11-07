"""Compare finished runs on the test split, which rounds never saw."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .data import DATASETS, Corpus
from .flops import average_tflops, tflops_by_k
from .train import heldout_loss, load, pick_device

K_GRID = (1, 2, 3, 4)
WIN_MARGIN = 0.005  # a dataset only counts as won with at least this relative gain
MEAN_GRID = (2.0, 2.5, 3.0)


def mutual_information(counts: torch.Tensor) -> float:
    """In bits, between dataset (rows) and expert (columns)."""
    joint = counts / counts.sum()
    rows, cols = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    mask = joint > 0
    return float((joint[mask] * (joint[mask] / (rows @ cols)[mask]).log2()).sum())


def shuffled_information(choices: torch.Tensor, labels: torch.Tensor, n_experts: int, tries: int = 50) -> float:
    """What mutual information looks like by chance at this sample size."""
    found = []
    for seed in range(tries):
        order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        counts = torch.zeros(int(labels.max()) + 1, n_experts)
        counts.index_put_((labels[order], choices), torch.ones(len(labels)), accumulate=True)
        found.append(mutual_information(counts))
    return sum(found) / tries


@torch.no_grad()
def decisions(model, corpus: Corpus, windows: int, mean_runs: float | None = None, learned: bool = False) -> dict:
    model.eval()
    device = next(model.parameters()).device
    experts, runs, labels = [], [], []
    per_dataset = {}
    for index, name in enumerate(DATASETS):
        batch = corpus.fixed_windows(name, "test", windows).to(device)
        chosen = [model.decide(row[:, :-1], mean_runs=mean_runs, learned=learned) for row in batch.split(32)]
        e, r = torch.cat([c[0] for c in chosen]).cpu(), torch.cat([c[1] for c in chosen]).cpu()
        experts.append(e), runs.append(r), labels.append(torch.full_like(e, index))
        per_dataset[name] = round(r.float().mean().item(), 3)
    experts, runs, labels = torch.cat(experts), torch.cat(runs), torch.cat(labels)
    result = {"mean_runs": round(runs.float().mean().item(), 3), "mean_runs_by_dataset": per_dataset,
              "run_counts": torch.bincount(runs, minlength=model.cfg.max_runs + 1)[1:].tolist()}
    if model.cfg.use_experts:
        n = model.cfg.n_experts
        counts = torch.zeros(len(DATASETS), n)
        counts.index_put_((labels, experts), torch.ones(len(labels)), accumulate=True)
        result.update(activation={name: counts[i].int().tolist() for i, name in enumerate(DATASETS)},
                      mutual_info_bits=round(mutual_information(counts), 4),
                      mutual_info_shuffled=round(shuffled_information(experts, labels, n), 4),
                      mean_runs_by_expert=[round(runs[experts == e].float().mean().item(), 3)
                                           if (experts == e).any() else None for e in range(n)])
    return result


@torch.no_grad()
def maths_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> float:
    """Exact-answer rate with greedy generation on the real inference path."""
    device = next(model.parameters()).device
    eod, correct = tok.token_to_id("[EOD]"), 0
    for item in exam[:questions]:
        prompt = tok.encode(item["prompt"]).ids
        ids, text = list(prompt), ""
        for _ in range(16):
            window = torch.tensor([ids[-model.cfg.seq_len:]], device=device)
            next_id = int(model.infer(window, **choice)[0][0, -1].argmax())
            if next_id == eod:
                break
            ids.append(next_id)
            text = tok.decode(ids[len(prompt):])
            if "\n" in text:
                break
        correct += text.split("\n")[0].strip() == item["answer"]
    return correct / questions


def accuracy_by_level(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> dict:
    """Exact-answer rate overall and per difficulty level."""
    items = exam[:questions]
    levels = sorted({item["level"] for item in items})
    found = {f"level_{n}": round(maths_accuracy(model, tok, [i for i in items if i["level"] == n],
                                                sum(i["level"] == n for i in items), **choice), 4)
             for n in levels}
    counts = [sum(i["level"] == n for i in items) for n in levels]
    found["all"] = round(sum(found[f"level_{n}"] * c for n, c in zip(levels, counts)) / len(items), 4)
    return found


@torch.no_grad()
def cycle_loss(model, corpus: Corpus, windows: int) -> dict:
    """Loss per dataset with the training cycle: runs 1, 2, 3, 4, 1, ... across tokens."""
    from .train import cycle_ks
    import torch.nn.functional as F

    device = next(model.parameters()).device
    losses, by_runs = {}, torch.zeros(model.cfg.max_runs)
    counts = torch.zeros(model.cfg.max_runs)
    for name in DATASETS:
        total = 0.0
        batch = corpus.fixed_windows(name, "test", windows).to(device)
        for row in batch.split(32):
            ks = cycle_ks(row.size(0), row.size(1) - 1, model.cfg.max_runs, device)
            each = F.cross_entropy(model.infer_tokens(row[:, :-1], ks).flatten(0, 1), row[:, 1:].flatten(),
                                   reduction="none").cpu()
            total += float(each.sum())
            flat = ks.flatten().cpu() - 1
            by_runs.index_add_(0, flat, each)
            counts.index_add_(0, flat, torch.ones_like(each))
        losses[name] = round(total / (windows * model.cfg.seq_len), 4)
    return {"loss": losses, "mean_loss": round(sum(losses.values()) / len(losses), 4),
            "loss_by_token_runs": [round(float(v), 4) for v in by_runs / counts.clamp(min=1)]}


def evaluate(run: str, corpus_root: str, windows: int, questions: int, device: torch.device) -> dict:
    model, _, step, _, _ = load(Path("runs") / run / "last.pt", device)
    corpus = Corpus(corpus_root, model.cfg.seq_len)
    tok = Tokenizer.from_file(str(Path(corpus_root) / "tokenizer.json"))
    exam = json.loads((Path(corpus_root) / "exam.json").read_text())
    result = {"run": run, "step": step, "params": model.num_params(), "by_k": {}, "by_mean": {},
              "targets": [round(x, 3) for x in model.run_targets.tolist()]}

    def score(**choice) -> dict:
        losses = heldout_loss(model, corpus, "test", windows, **choice)
        return {"loss": {k: round(v, 4) for k, v in losses.items()},
                "mean_loss": round(sum(losses.values()) / len(losses), 4),
                "maths_accuracy": (levels := accuracy_by_level(model, tok, exam, questions, **choice))["all"],
                "maths_levels": levels}

    cost = tflops_by_k(model.cfg)
    result["cycle"] = cycle_loss(model, corpus, windows)
    for k in K_GRID:
        result["by_k"][str(k)] = score(k=k) | {"tflops_per_token": cost[k]}
    for mean in MEAN_GRID:
        chosen = decisions(model, corpus, windows, mean)
        result["by_mean"][str(mean)] = score(mean_runs=mean) | chosen | {
            "tflops_per_token": average_tflops(cost, chosen["run_counts"])}
    if model.cfg.use_experts and bool(model.run_head_ready):
        chosen = decisions(model, corpus, windows, learned=True)
        result["learned"] = score(learned=True) | chosen | {
            "tflops_per_token": average_tflops(cost, chosen["run_counts"])}
    return result


def constant_at(result: dict, mean: float) -> dict[str, float]:
    """A constant model at a whole mean uses that k; at 2.5 it is the mean of k=2 and k=3."""
    if mean == int(mean):
        return result["by_k"][str(int(mean))]["loss"]
    low, high = result["by_k"][str(int(mean))]["loss"], result["by_k"][str(int(mean) + 1)]["loss"]
    return {n: (low[n] + high[n]) / 2 for n in DATASETS}


def average(losses: dict[str, float]) -> float:
    return sum(losses.values()) / len(losses)


def verdict(plain: dict, constant: dict, variable: dict) -> dict:
    """The success rules, as numbers and pass or fail."""
    one = plain["by_k"]["1"]["mean_loss"]
    best = min(plain["by_k"][k]["mean_loss"] for k in ("2", "3", "4"))
    loops = (one - best) / one

    wins = {}
    for k in ("1", "2", "3", "4"):
        p, c = plain["by_k"][k], constant["by_k"][k]
        wins[k] = {"datasets_won": sum(c["loss"][n] < p["loss"][n] * (1 - WIN_MARGIN) for n in DATASETS),
                   "maths_loss_gain": round((p["loss"]["maths"] - c["loss"]["maths"]) / p["loss"]["maths"], 4),
                   "maths_acc_gain": round(c["maths_accuracy"] - p["maths_accuracy"], 4)}
    experts_pass = any(w["datasets_won"] >= 3 or w["maths_loss_gain"] >= 0.01 for k, w in wins.items() if k != "1")

    gains = {}
    for mean in MEAN_GRID:
        base = average(constant_at(constant, mean))
        gains[str(mean)] = round((base - variable["by_mean"][str(mean)]["mean_loss"]) / base, 4)

    learned = variable.get("learned")
    learned_gain = None
    if learned:
        # Constant at the same average compute, by straight-line blend of its two nearest run counts.
        runs = min(learned["mean_runs"], 4.0)
        low = min(int(runs), 3)
        share = runs - low
        at = lambda k: constant["by_k"][str(k)]["mean_loss"]
        base = at(low) * (1 - share) + at(low + 1) * share
        learned_gain = round((base - learned["mean_loss"]) / base, 4)

    info = variable["by_mean"]["2.5"]
    return {"learned_gain": learned_gain, "learned_pass": bool(learned_gain is not None and learned_gain >= 0.01),
            "learned_mean_runs": learned["mean_runs"] if learned else None,
            "learned_run_counts": learned["run_counts"] if learned else None,
            "loops_gain": round(loops, 4), "loops_pass": loops >= 0.01,
            "experts_by_k": wins, "experts_pass": experts_pass,
            "variable_gain_by_mean": gains, "variable_pass": all(g >= 0.01 for g in gains.values()),
            "mutual_info_bits": info["mutual_info_bits"], "mutual_info_shuffled": info["mutual_info_shuffled"],
            "specialised_pass": info["mutual_info_bits"] >= 3 * info["mutual_info_shuffled"]}


def combine(reports: list[dict]) -> dict:
    """A rule holds only if every seed passes. One seed can never confirm a result."""
    names = ("loops_pass", "experts_pass", "variable_pass", "learned_pass", "specialised_pass")
    summary = {name: all(r["rules"][name] for r in reports) for name in names}
    summary["confirmed"] = len(reports) >= 2
    summary["seeds"] = [r["seed"] for r in reports]
    summary["loops_gain"] = [r["rules"]["loops_gain"] for r in reports]
    summary["variable_gain_by_mean"] = [r["rules"]["variable_gain_by_mean"] for r in reports]
    return summary


def loop_check(run: str, corpus_root: str, windows: int, questions: int, device: torch.device) -> bool:
    """Early check on held-out data: does the plain model gain from more than one run?

    Passes on 1% lower loss, or on 2 points more maths accuracy, the task loops should help most.
    """
    model, *_ = load(Path("runs") / run / "last.pt", device)
    corpus = Corpus(corpus_root, model.cfg.seq_len)
    tok = Tokenizer.from_file(str(Path(corpus_root) / "tokenizer.json"))
    exam = json.loads((Path(corpus_root) / "exam.json").read_text())
    by_k = {k: heldout_loss(model, corpus, "val", windows, k=k) for k in K_GRID}
    losses = {k: average(v) for k, v in by_k.items()}
    accuracy = {k: accuracy_by_level(model, tok, exam, questions, k=k) for k in K_GRID}
    overall = {k: v["all"] for k, v in accuracy.items()}
    gain = (losses[1] - min(losses[k] for k in K_GRID[1:])) / losses[1]
    points = max(overall[k] for k in K_GRID[1:]) - overall[1]
    # Judged per dataset, so one overfit dataset cannot hide a real gain elsewhere.
    per_dataset = {n: (by_k[1][n] - min(by_k[k][n] for k in K_GRID[1:])) / by_k[1][n] for n in by_k[1]}
    passed = max(per_dataset.values()) >= 0.01 or points >= 0.02
    report = {"val_loss_by_k": {k: round(v, 4) for k, v in losses.items()},
              "dataset_loss_by_k": {k: {n: round(x, 4) for n, x in v.items()} for k, v in by_k.items()},
              "maths_accuracy_by_k": accuracy, "loss_gain": round(gain, 4),
              "accuracy_points": round(points, 4),
              "dataset_gain": {n: round(g, 4) for n, g in per_dataset.items()}, "pass": passed}
    print(json.dumps(report), flush=True)
    Path("runs/loop_check.json").write_text(json.dumps(report, indent=2))
    return passed


def table(results: list[dict]) -> str:
    head = ["run", "params", "k1", "k2", "k3", "k4", "m2", "m2.5", "m3", "acc k3", "acc m3"]
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in results:
        cells = [r["run"], f"{r['params']:,}"]
        cells += [f"{r['by_k'][k]['mean_loss']:.3f}" for k in "1234"]
        cells += [f"{r['by_mean'][str(m)]['mean_loss']:.3f}" for m in MEAN_GRID]
        cells += [f"{r['by_k']['3']['maths_accuracy']:.3f}", f"{r['by_mean']['3.0']['maths_accuracy']:.3f}"]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--loop-check", help="run name: exit 3 if loops do not help")
    parser.add_argument("--single", help="evaluate one run name and write it alone")
    parser.add_argument("--names", nargs="+", help="run names to compare; must include plain and constant")
    parser.add_argument("--data", default="data")
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/results.json")
    args = parser.parse_args()
    device = pick_device(args.device)
    if args.loop_check:
        raise SystemExit(0 if loop_check(args.loop_check, args.data, args.windows, args.questions, device) else 3)

    if args.single:
        result = evaluate(args.single, args.data, args.windows, args.questions, device)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(table([result]))
        return

    if args.names:
        report = []
        for seed in args.seeds:
            runs = {name: evaluate(f"{name}_s{seed}", args.data, args.windows, args.questions, device)
                    for name in args.names}
            print(table(list(runs.values())))
            rules = {name: verdict(runs["plain"], runs["constant"], run)
                     for name, run in runs.items() if name not in ("plain", "constant")}
            print(json.dumps(rules, indent=2), flush=True)
            report.append({"seed": seed, "runs": runs, "rules": rules})
        Path(args.out).write_text(json.dumps({"seeds": report}, indent=2))
        return

    report = []
    for seed in args.seeds:
        runs = [evaluate(f"{name}_s{seed}", args.data, args.windows, args.questions, device)
                for name in ("plain", "constant", "variable")]
        print(table(runs))
        rules = verdict(*runs)
        print(json.dumps(rules, indent=2), flush=True)
        report.append({"seed": seed, "runs": runs, "rules": rules})
    summary = combine(report)
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "seeds": report}, indent=2))


if __name__ == "__main__":
    main()
