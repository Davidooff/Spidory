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


def shuffled_information(choices: torch.Tensor, labels: torch.Tensor, n_experts: int, tries: int = 50,
                         groups: torch.Tensor | None = None) -> float:
    """What mutual information looks like by chance at this sample size.

    With `groups` (the window each decision came from), labels are shuffled between windows, not
    between tokens: tokens of one window are strongly alike, so shuffling them one by one would
    make chance look too small.
    """
    found = []
    for seed in range(tries):
        generator = torch.Generator().manual_seed(seed)
        if groups is None:
            shuffled = labels[torch.randperm(len(labels), generator=generator)]
        else:
            per_group = torch.zeros(int(groups.max()) + 1, dtype=labels.dtype).scatter_(0, groups, labels)
            shuffled = per_group[torch.randperm(len(per_group), generator=generator)][groups]
        counts = torch.zeros(int(labels.max()) + 1, n_experts)
        counts.index_put_((shuffled, choices), torch.ones(len(labels)), accumulate=True)
        found.append(mutual_information(counts))
    return sum(found) / tries


@torch.no_grad()
def decisions(model, corpus: Corpus, windows: int, mean_runs: float | None = None, learned: bool = False) -> dict:
    model.eval()
    device = next(model.parameters()).device
    experts, runs, paid, labels, groups = [], [], [], [], []
    per_dataset = {}
    for index, name in enumerate(DATASETS):
        batch = corpus.fixed_windows(name, "test", windows).to(device)
        chosen = [model.decide(row[:, :-1], mean_runs=mean_runs, learned=learned) for row in batch.split(32)]
        # Per-token models choose per token: every token counts as one decision. Compute is
        # charged honestly: a token must run as many times as any later token in its window,
        # because later tokens attend to its state at their own runs.
        ks = [c[1] for c in chosen]
        charged = [k.flip(-1).cummax(-1).values.flip(-1) if k.dim() == 2 else k for k in ks]
        e = torch.cat([c[0].flatten() for c in chosen]).cpu()
        r = torch.cat([k.flatten() for k in ks]).cpu()
        experts.append(e), runs.append(r), labels.append(torch.full_like(e, index))
        paid.append(torch.cat([k.flatten() for k in charged]).cpu())
        per_window = e.numel() // windows
        groups.append(torch.arange(windows).repeat_interleave(per_window) + index * windows)
        per_dataset[name] = round(r.float().mean().item(), 3)
    experts, runs, paid, labels, groups = (torch.cat(x) for x in (experts, runs, paid, labels, groups))
    result = {"mean_runs": round(runs.float().mean().item(), 3), "mean_runs_by_dataset": per_dataset,
              "mean_runs_paid": round(paid.float().mean().item(), 3),
              "run_counts": torch.bincount(paid, minlength=model.cfg.max_runs + 1)[1:].tolist()}
    if model.cfg.use_experts:
        n = model.cfg.n_experts
        counts = torch.zeros(len(DATASETS), n)
        counts.index_put_((labels, experts), torch.ones(len(labels)), accumulate=True)
        result.update(activation={name: counts[i].int().tolist() for i, name in enumerate(DATASETS)},
                      mutual_info_bits=round(mutual_information(counts), 4),
                      mutual_info_shuffled=round(shuffled_information(experts, labels, n, groups=groups), 4),
                      mean_runs_by_expert=[round(runs[experts == e].float().mean().item(), 3)
                                           if (experts == e).any() else None for e in range(n)])
    return result


BATCH = 64  # exam items generated together; only items with prompts of one length share a batch


@torch.no_grad()
def maths_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> float:
    """Exact-answer rate with greedy generation on the real inference path."""
    return sum(ok for _, ok in answer_items("maths", model, tok, exam[:questions], **choice)) / questions


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
def generate_batch(model, tok: Tokenizer, prompts: list[str], longest: int, done=None,
                   batch: int = BATCH, **choice) -> list[tuple[str, bool]]:
    """Greedy continuations, each until end of document, `longest` tokens, or done(i, text).

    Returns (text, stopped by done) per prompt. Prompts of the same token length run as one
    batch: they gain one token per step, so every row stays exactly as long as it would alone,
    with the same window, the same first tokens for routing and the same noise. No padding is
    needed, so each row computes what a batch of one computes, up to rounding in the matrix
    products.
    """
    device = next(model.parameters()).device
    eod = tok.token_to_id("[EOD]")
    encoded = [tok.encode(p).ids for p in prompts]
    found: list[tuple[str, bool]] = [("", False)] * len(prompts)
    by_length: dict[int, list[int]] = {}
    for i, ids in enumerate(encoded):
        by_length.setdefault(len(ids), []).append(i)
    for indices in by_length.values():
        for begin in range(0, len(indices), batch):
            live = indices[begin:begin + batch]
            ids = {i: list(encoded[i]) for i in live}
            start = len(encoded[live[0]])
            texts = {i: "" for i in live}
            for _ in range(longest):
                if not live:
                    break
                window = torch.tensor([ids[i][-model.cfg.seq_len:] for i in live], device=device)
                next_ids = model.infer(window, **choice)[0][:, -1].argmax(-1).tolist()
                still = []
                for i, next_id in zip(live, next_ids):
                    if next_id == eod:
                        continue
                    ids[i].append(next_id)
                    texts[i] = tok.decode(ids[i][start:])
                    if done is not None and done(i, texts[i]):
                        found[i] = (texts[i], True)
                        continue
                    still.append(i)
                live = still
            for i, text in texts.items():
                if not found[i][1]:
                    found[i] = (text, False)
    return found


@torch.no_grad()
def generate_text(model, tok: Tokenizer, prompt: str, longest: int, done=None, **choice) -> str:
    """Greedy continuation of one prompt, until end of document, `longest` tokens, or done(text)."""
    check = None if done is None else (lambda _, text: done(text))
    return generate_batch(model, tok, [prompt], longest, check, **choice)[0][0]


def compose_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> float:
    """Share of "write a problem" tasks done as asked, checked by computing the reply."""
    items = exam[:questions]
    return sum(ok for _, ok in answer_items("compose", model, tok, items, **choice)) / max(1, len(items))


def compose_done(item: dict):
    """When a reply is complete: a story ends after its answer line, a question after its
    "Answer:" line, an expression after its one line. A whole line is needed, because a number
    of several tokens arrives one token at a time."""
    if item["kind"] == "story":
        return lambda text: bool(re.search(r"\nA: [^\n]*\n", text))
    if item["kind"] == "question":
        return lambda text: bool(re.search(r"Answer: [^\n]*\n", text))
    return lambda text: "\n" in text


def word_problem_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, **choice) -> float:
    """GSM8K or SVAMP: the stated answer, or the last number written, must match."""
    items = exam[:questions]
    return sum(ok for _, ok in answer_items("word", model, tok, items, **choice)) / max(1, len(items))


def explain_accuracy(model, tok: Tokenizer, exam: list[dict], questions: int, longest: int = 220,
                     **choice) -> float:
    """Exact-answer rate when the model explains its steps first and ends with "The answer is N"."""
    hits = sum(ok for _, ok in answer_items("explain", model, tok, exam[:questions], longest=longest, **choice))
    return hits / max(1, min(questions, len(exam)))


def _word_done(text: str) -> bool:
    return "answer is" in text.lower() and text.rstrip().endswith(".") or "\nQ:" in text


def _explain_done(text: str) -> bool:
    """A line break, or a finished sentence that states the answer."""
    from .mixed_data import answer_of

    return "\n" in text or (text.rstrip().endswith(".") and answer_of(text) is not None)


LONGEST = {"maths": 16, "explain": 220, "compose": 180, "word": 300}


@torch.no_grad()
def answer_items(kind: str, model, tok: Tokenizer, items: list[dict], longest: int | None = None,
                 **choice) -> list[tuple[str, bool]]:
    """Exam items: the model's greedy reply to each and whether it is right.

    kind is "maths" (the first line must equal the answer, also used for the mixed-topic exam),
    "explain" (the sentence ending "the answer is N"), "compose" (the reply must do what was
    asked, checked by computing) or "word" (GSM8K: the stated or last number must match). The
    accuracy functions above and the playground both score through here.
    """
    from .mixed_data import answer_of, grade_compose, last_number

    if kind not in LONGEST:
        raise ValueError(f"unknown exam kind {kind!r}")
    if kind == "maths":
        done = lambda i, text: "\n" in text
    elif kind == "explain":
        done = lambda i, text: _explain_done(text)
    elif kind == "compose":
        checks = [compose_done(item) for item in items]
        done = lambda i, text: checks[i](text)
    else:
        done = lambda i, text: _word_done(text)
    replies = generate_batch(model, tok, [item["prompt"] for item in items], longest or LONGEST[kind],
                             done, **choice)
    found = []
    for item, (text, stopped) in zip(items, replies):
        if kind == "maths":
            ok = text.split("\n")[0].strip() == item["answer"]
        elif kind == "explain":
            # As before: the answer counts only where the reply stopped by stating it.
            ok = stopped and answer_of(text) == item["answer"]
        elif kind == "compose":
            ok = grade_compose(item, text)
        else:
            text = text.split("\nQ:")[0]
            number = last_number(text)
            try:
                ok = number is not None and float(number) == float(item["answer"])
            except ValueError:
                ok = False
        found.append((text, bool(ok)))
    return found


def answer_item(kind: str, model, tok: Tokenizer, item: dict, longest: int | None = None,
                **choice) -> tuple[str, bool]:
    """One exam item; see answer_items."""
    return answer_items(kind, model, tok, [item], longest, **choice)[0]


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


def extra_scores(model, corpus: Corpus, tok: Tokenizer, windows: int, questions: int,
                 generate: bool = True, **choice) -> dict:
    """Scores on the mixed-topic and explained-maths sets, when this corpus has them.

    The long generation exams (explained maths, writing a problem, GSM8K) run only when
    `generate` is true, since without a cache they cost tens of minutes per setting.
    """
    root, found = Path(corpus.root), {}
    if "mixed" in DATASETS and (root / "mixed.test.labels").exists():
        found["mixed_topics"] = mixed_topic_loss(model, corpus, windows, **choice)
        exam = json.loads((root / "mixed_exam.json").read_text())
        found["mixed_accuracy"] = round(maths_accuracy(model, tok, exam, min(questions, len(exam)), **choice), 4)
    if not generate:
        return found
    if "explain" in DATASETS and (root / "explain_exam.json").exists():
        exam = json.loads((root / "explain_exam.json").read_text())
        found["explain_accuracy"] = round(explain_accuracy(model, tok, exam, min(questions, 180), **choice), 4)
    if "compose" in DATASETS and (root / "compose_exam.json").exists():
        exam = json.loads((root / "compose_exam.json").read_text())
        found["compose_accuracy"] = round(compose_accuracy(model, tok, exam, min(questions, 150), **choice), 4)
    for name in ("gsm8k",):
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
                "maths_levels": levels} | extra_scores(model, corpus, tok, windows, questions,
                                                       generate=choice in GENERATE_AT, **choice)

    cost = tflops_by_k(model.cfg)
    result["cycle"] = cycle_loss(model, corpus, windows)
    for k in K_GRID:
        if k <= model.cfg.max_runs:  # a model without loops is scored at one run only
            result["by_k"][str(k)] = score(k=k) | {"tflops_per_token": cost[k]}
    for mean in MEAN_GRID if model.cfg.max_runs >= max(MEAN_GRID) else ():
        chosen = decisions(model, corpus, windows, mean)
        result["by_mean"][str(mean)] = score(mean_runs=mean) | chosen | {
            "tflops_per_token": average_tflops(cost, chosen["run_counts"])}
    if model.cfg.use_experts and bool(model.run_head_ready):
        chosen = decisions(model, corpus, windows, learned=True)
        result["learned"] = score(learned=True) | chosen | {
            "tflops_per_token": average_tflops(cost, chosen["run_counts"])}
        result["policy"] = policy_test(model, corpus, windows)
    return result


GENERATE_AT = ({"k": 1}, {"k": 3}, {"learned": True})  # long exams only here

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
        get = lambda table, key, field: f"{table[key][field]:.3f}" if key in table else "—"
        cells += [get(r["by_k"], k, "mean_loss") for k in "1234"]
        cells += [get(r["by_mean"], str(m), "mean_loss") for m in MEAN_GRID]
        cells += [get(r["by_k"], "3", "maths_accuracy"), get(r["by_mean"], "3.0", "maths_accuracy")]
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
