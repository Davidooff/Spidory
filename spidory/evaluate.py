"""Compare finished runs on the test split, which rounds never saw."""
from __future__ import annotations

import argparse
import json
import re
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
        # Per-token models choose per token: every token counts as one decision.
        e = torch.cat([c[0].flatten() for c in chosen]).cpu()
        r = torch.cat([c[1].flatten() for c in chosen]).cpu()
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
def generate_text(model, tok: Tokenizer, prompt: str, longest: int, done=None, **choice) -> str:
    """Greedy continuation, until end of document, `longest` tokens, or done(text) is true."""
    device = next(model.parameters()).device
    eod = tok.token_to_id("[EOD]")
    ids = tok.encode(prompt).ids
    start, text = len(ids), ""
    for _ in range(longest):
        window = torch.tensor([ids[-model.cfg.seq_len:]], device=device)
        next_id = int(model.infer(window, **choice)[0][0, -1].argmax())
        if next_id == eod:
            break
        ids.append(next_id)
        text = tok.decode(ids[start:])
        if done is not None and done(text):
            break
    return text


def compose_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> float:
    """Share of "write a problem" tasks done as asked, checked by computing the reply."""
    from .mixed_data import grade_compose

    def done(text):  # a story problem ends with its answer line; the others with one line
        return bool(re.search(r"A: -?\d+\s*$", text)) if "A: " in text else text.count("\n") >= 2

    items = exam[:questions]
    return sum(grade_compose(i, generate_text(model, tok, i["prompt"], 120, done, **choice)) for i in items) / max(1, len(items))


def word_problem_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> float:
    """GSM8K or SVAMP: the stated answer, or the last number written, must match."""
    from .mixed_data import last_number

    def done(text):
        return "answer is" in text.lower() and text.rstrip().endswith(".") or "\nQ:" in text

    items = exam[:questions]
    hits = 0
    for item in items:
        text = generate_text(model, tok, item["prompt"], 160, done, **choice).split("\nQ:")[0]
        found = last_number(text)
        hits += found is not None and found.rstrip("0").rstrip(".") == item["answer"].rstrip("0").rstrip(".")
    return hits / max(1, len(items))


@torch.no_grad()
def explain_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, longest: int = 160,
                     **choice) -> float:
    """Exact-answer rate when the model explains its steps first and ends with "The answer is N"."""
    from .mixed_data import answer_of

    device = next(model.parameters()).device
    eod, correct = tok.token_to_id("[EOD]"), 0
    for item in exam[:questions]:
        prompt = tok.encode(item["prompt"]).ids
        ids, found = list(prompt), None
        for _ in range(longest):
            window = torch.tensor([ids[-model.cfg.seq_len:]], device=device)
            next_id = int(model.infer(window, **choice)[0][0, -1].argmax())
            if next_id == eod:
                break
            ids.append(next_id)
            text = tok.decode(ids[len(prompt):])
            if "\n" in text or text.rstrip().endswith("."):
                found = answer_of(text)
                if found is not None or "\n" in text:
                    break
        correct += found == item["answer"]
    return correct / max(1, min(questions, len(exam)))


@torch.no_grad()
def mixed_topic_loss(model, corpus: Corpus, windows: int, chunk: int = 32, **choice) -> dict:
    """Loss inside mixed-topic test texts, per topic of the predicted token."""
    import numpy as np
    import torch.nn.functional as F

    from .mixed_data import TOPICS

    model.eval()
    device = next(model.parameters()).device
    stream = corpus.tokens["mixed", "test"]
    labels = np.memmap(corpus.root / "mixed.test.labels", dtype=np.uint8, mode="r")
    starts = np.linspace(0, len(stream) - corpus.seq_len - 2, windows).astype(np.int64)
    total, count = torch.zeros(len(TOPICS)), torch.zeros(len(TOPICS))
    for begin in range(0, windows, chunk):
        part = starts[begin:begin + chunk]
        rows = torch.from_numpy(np.stack([stream[s:s + corpus.seq_len + 1] for s in part]).astype(np.int64)).to(device)
        topic = torch.from_numpy(np.stack([labels[s + 1:s + corpus.seq_len + 1] for s in part]).astype(np.int64))
        logits = model.infer(rows[:, :-1], **choice)[0]
        loss = F.cross_entropy(logits.flatten(0, 1).float(), rows[:, 1:].flatten(), reduction="none").cpu()
        topic = topic.flatten()
        for name, index in TOPICS.items():
            mask = topic == index
            total[index] += loss[mask].sum()
            count[index] += mask.sum()
    return {name: round(float(total[i] / count[i].clamp(min=1)), 4) for name, i in TOPICS.items()}


def extra_scores(model, corpus: Corpus, tok: Tokenizer, windows: int, questions: int, **choice) -> dict:
    """Scores on the mixed-topic and explained-maths sets, when this corpus has them."""
    root, found = Path(corpus.root), {}
    if "mixed" in DATASETS and (root / "mixed.test.labels").exists():
        found["mixed_topics"] = mixed_topic_loss(model, corpus, windows, **choice)
        exam = json.loads((root / "mixed_exam.json").read_text())
        found["mixed_accuracy"] = round(maths_accuracy(model, tok, exam, min(questions, len(exam)), **choice), 4)
    if "explain" in DATASETS and (root / "explain_exam.json").exists():
        exam = json.loads((root / "explain_exam.json").read_text())
        found["explain_accuracy"] = round(explain_accuracy(model, tok, exam, min(questions, 120), **choice), 4)
    if "compose" in DATASETS and (root / "compose_exam.json").exists():
        exam = json.loads((root / "compose_exam.json").read_text())
        found["compose_accuracy"] = round(compose_accuracy(model, tok, exam, min(questions, 150), **choice), 4)
    for name in ("gsm8k", "svamp"):
        if "wordmath" in DATASETS and (root / f"{name}_exam.json").exists():
            exam = json.loads((root / f"{name}_exam.json").read_text())
            found[f"{name}_accuracy"] = round(word_problem_accuracy(model, tok, exam, min(questions, 150), **choice), 4)
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
                "maths_levels": levels} | extra_scores(model, corpus, tok, windows, questions, **choice)

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
        result["policy"] = policy_test(model, corpus, windows)
    return result


@torch.no_grad()
def sequence_losses(model, corpus: Corpus, split: str, windows: int, chunk: int = 32, **choice) -> dict:
    """Mean next-token loss of each window, per dataset, on the real inference path."""
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device
    found = {}
    for name in DATASETS:
        batch = corpus.fixed_windows(name, split, windows).to(device)
        parts = []
        for row in batch.split(chunk):
            logits = model.infer(row[:, :-1], **choice)[0]
            parts.append(F.cross_entropy(logits.flatten(0, 1).float(), row[:, 1:].flatten(),
                                         reduction="none").view(row.size(0), -1).mean(1).cpu())
        found[name] = torch.cat(parts)
    return found


def paired_ci(ours: dict, other: dict, boots: int = 2000, seed: int = 0) -> dict:
    """Difference in mean loss (ours minus other), datasets weighted equally, with a 95% interval.

    Sequences are resampled with replacement within each dataset, keeping each pair together.
    Negative means ours has lower loss.
    """
    generator = torch.Generator().manual_seed(seed)
    diffs = {n: ours[n] - other[n] for n in ours}
    point = sum(d.mean() for d in diffs.values()) / len(diffs)
    samples = torch.zeros(boots)
    for d in diffs.values():
        idx = torch.randint(len(d), (boots, len(d)), generator=generator)
        samples += d[idx].mean(1)
    samples /= len(diffs)
    low, high = torch.quantile(samples, torch.tensor([0.025, 0.975])).tolist()
    return {"difference": round(float(point), 5), "ci_low": round(low, 5), "ci_high": round(high, 5),
            "ours_better": high < 0}


def policy_test(model, corpus: Corpus, windows: int) -> dict:
    """Does the neuron's per-input choice beat fixed run counts?

    The fixed choices are picked on validation data, then everything is compared on test data:
    against the best single run count, and against the best run count per dataset. Beating the
    per-dataset choice is what shows that choosing per input, not per dataset, helps.
    """
    val = {k: sequence_losses(model, corpus, "val", windows, k=k) for k in K_GRID}
    test = {k: sequence_losses(model, corpus, "test", windows, k=k) for k in K_GRID}
    learned = sequence_losses(model, corpus, "test", windows, learned=True)
    val_mean = {k: sum(float(v.mean()) for v in val[k].values()) / len(DATASETS) for k in K_GRID}
    best = min(K_GRID, key=lambda k: val_mean[k])
    per_dataset = {n: min(K_GRID, key=lambda k: float(val[k][n].mean())) for n in DATASETS}
    fixed = test[best]
    by_dataset = {n: test[per_dataset[n]][n] for n in DATASETS}
    cost = tflops_by_k(model.cfg)
    chosen = decisions(model, corpus, windows, learned=True)
    learned_cost = average_tflops(cost, chosen["run_counts"])
    fixed_cost = cost[best]
    per_dataset_cost = sum(cost[k] for k in per_dataset.values()) / len(DATASETS)
    vs_fixed, vs_dataset = paired_ci(learned, fixed), paired_ci(learned, by_dataset)
    return {"best_fixed_k": best, "per_dataset_k": per_dataset,
            "per_dataset_mean_k": sum(per_dataset.values()) / len(DATASETS),
            "learned_mean_runs": chosen["mean_runs"],
            "tflops": {"learned": learned_cost, "best_fixed": fixed_cost, "per_dataset": per_dataset_cost},
            "vs_best_fixed": vs_fixed | {"at_no_more_compute": learned_cost <= fixed_cost * 1.001},
            "vs_per_dataset": vs_dataset | {"at_no_more_compute": learned_cost <= per_dataset_cost * 1.001},
            "note": "A win counts only with ours_better and at_no_more_compute. The mix variant always "
                    "computes every run, and its fixed-run baselines decode states it never saw alone."}


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
    majority = len(DATASETS) // 2 + 1
    experts_pass = any(w["datasets_won"] >= majority or w["maths_loss_gain"] >= 0.01 for k, w in wins.items() if k != "1")

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
    policy = variable.get("policy") or {}
    return {"policy": policy,
            "beats_best_fixed": bool(policy.get("vs_best_fixed", {}).get("ours_better")
                                     and policy.get("vs_best_fixed", {}).get("at_no_more_compute")),
            "beats_per_dataset": bool(policy.get("vs_per_dataset", {}).get("ours_better")
                                      and policy.get("vs_per_dataset", {}).get("at_no_more_compute")),
            "learned_gain": learned_gain, "learned_pass": bool(learned_gain is not None and learned_gain >= 0.01),
            "learned_mean_runs": learned["mean_runs"] if learned else None,
            "learned_run_counts": learned["run_counts"] if learned else None,
            "loops_gain": round(loops, 4), "loops_pass": loops >= 0.01,
            "experts_by_k": wins, "experts_pass": experts_pass,
            "variable_gain_by_mean": gains, "variable_pass": all(g >= 0.01 for g in gains.values()),
            "mutual_info_bits": info["mutual_info_bits"], "mutual_info_shuffled": info["mutual_info_shuffled"],
            "specialised_pass": info["mutual_info_bits"] >= 3 * info["mutual_info_shuffled"]}


def combine(reports: list[dict]) -> dict:
    """A rule holds only if every seed passes. One seed can never confirm a result."""
    names = ("loops_pass", "experts_pass", "variable_pass", "learned_pass", "specialised_pass",
             "beats_best_fixed", "beats_per_dataset")
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
