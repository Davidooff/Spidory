"""Second dataset: harder multi-step maths and much more QA data.

Writes data_v2/. Story and code stay as they are, and qa.val and qa.test stay SQuAD, so
scores remain comparable. Maths held-out sets are new, because the task changed.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import easy_maths as em
from .data import fetch_parquet
from .easy_maths import Item, _chain, _split_of, render, prompt
from .grow_data import encode_to

LEVELS = (2, 5, 6, 7, 8, 9)
EXAM_PER_LEVEL = 500
TRAIN_TOKENS = 120_000_000
HELD_TOKENS = 590_000
LETTERS = "abcdefghjkmnpqrstuvwxyz"
PLAIN_WORDINGS = ("What is {}?", "Calculate {}.", "Compute {}.", "{} =")


def _variables(rng: random.Random) -> tuple[str, int, str]:
    """Level 8: track values through assignments. Each step uses the previous one."""
    names = rng.sample(LETTERS, rng.randint(3, 5))
    values = {names[0]: rng.randint(1, 15)}
    parts = [f"{names[0]} = {values[names[0]]}"]
    for index, name in enumerate(names[1:], 1):
        previous = names[index - 1]
        op = rng.choice("+-")
        if index >= 2 and rng.random() < 0.4:
            other = rng.choice(names[:index - 1])
            parts.append(f"{name} = {previous} {op} {other}")
            right = values[other]
        else:
            right = rng.randint(1, 15)
            parts.append(f"{name} = {previous} {op} {right}")
        values[name] = em.OPS[op](values[previous], right)
    text = ", ".join(parts)
    return f"{text}; {names[-1]}", values[names[-1]], f"{text}. What is {names[-1]}?"


def _tree(rng: random.Random, depth: int) -> tuple[str, int]:
    if depth == 0 or rng.random() < 0.2:
        value = rng.randint(0, 9)
        return str(value), value
    left, a = _tree(rng, depth - 1)
    right, b = _tree(rng, depth - 1)
    op = rng.choice("+-*")
    return f"({left} {op} {right})", em.OPS[op](a, b)


def _nested(rng: random.Random) -> tuple[str, int]:
    """Level 9: nested brackets, with results kept to a readable size."""
    while True:
        text, value = _tree(rng, rng.randint(2, 3))
        inner = text[1:-1]
        if inner.count("(") >= 2 and abs(value) <= 500:  # one pair could repeat a level 5 question
            return inner, value


def make_item(rng: random.Random, level: int, split: str, pools: dict) -> Item:
    if level in (2, 5):
        families = sorted(f for (lv, f, name) in pools if lv == level and name == split)
        family = rng.choice(families)
        text, value = rng.choice(pools[level, family, split])
        return Item(level, family, text, value, rng.choice(PLAIN_WORDINGS).format(text))
    while True:
        if level in (6, 7):
            low, high = (6, 8) if level == 6 else (9, 12)
            length = rng.randint(low, high)
            text, value = _chain([rng.randint(0, 20) for _ in range(length)],
                                 [rng.choice("+-") for _ in range(length - 1)])
            question = rng.choice(PLAIN_WORDINGS).format(text)
        elif level == 8:
            text, value, question = _variables(rng)
        else:
            text, value = _nested(rng)
            question = rng.choice(PLAIN_WORDINGS).format(text)
        if _split_of(text) == split:
            return Item(level, f"level{level}", text, value, question)


def generate(split: str, rounds: int, seed: int, pools: dict):
    """Yields shuffled items, one of each level per round, without holding them in memory."""
    rng = random.Random(f"v2:{seed}:{split}")
    for start in range(0, rounds, 20_000):
        batch = [make_item(rng, level, split, pools)
                 for _ in range(min(20_000, rounds - start)) for level in LEVELS]
        rng.shuffle(batch)
        yield batch


def build_maths(root: Path, tok: Tokenizer, seed: int, train_tokens: int, held_tokens: int) -> dict[str, int]:
    eod = tok.token_to_id("[EOD]")
    pools = em._pools()
    pilot = [i for batch in generate("train", 400, seed, pools) for i in batch]
    mean = sum(len(e.ids) + 1 for e in tok.encode_batch([render(i) for i in pilot])) / len(pilot)
    counts, exam = {}, []
    for split, target in (("train", train_tokens), ("val", held_tokens), ("test", held_tokens)):
        rounds = int(target / mean / len(LEVELS))
        total = 0
        with open(root / f"maths.{split}.bin", "wb") as sink:
            for batch in generate(split, rounds, seed, pools):
                total += encode_to(sink, tok, [render(i) for i in batch], eod)
                if split == "test" and len(exam) < EXAM_PER_LEVEL * len(LEVELS) * 3:
                    exam.extend(batch)
        counts[split] = total
    picked = {level: [i for i in exam if i.level == level][:EXAM_PER_LEVEL] for level in LEVELS}
    rows = [{"prompt": prompt(item), "answer": str(item.answer), "level": item.level}
            for group in zip(*picked.values()) for item in group]
    (root / "exam.json").write_text(json.dumps(rows))
    return counts


def _squad_held_titles() -> set[str]:
    return set(fetch_parquet("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet")["title"].to_pylist())


def hotpot_docs(held: set[str]):
    """Multi-hop questions, with only the supporting passages, so they fit a window."""
    for name in ("distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"):
        table = fetch_parquet("hotpotqa/hotpot_qa", name)
        docs = []
        for q, a, facts, context in zip(*(table[c].to_pylist() for c in
                                          ("question", "answer", "supporting_facts", "context"))):
            facts = ast.literal_eval(facts) if isinstance(facts, str) else facts
            context = ast.literal_eval(context) if isinstance(context, str) else context
            wanted = list(dict.fromkeys(facts["title"]))
            if held.intersection(wanted):
                continue
            by_title = dict(zip(context["title"], context["sentences"]))
            passages = ["".join(by_title[t]).strip() for t in wanted if t in by_title]
            docs.append("\n".join(passages + [f"Q: {q}", f"A: {a}"]))
        yield docs


def build_qa(root: Path, source: Path, tok: Tokenizer) -> dict[str, int]:
    eod = tok.token_to_id("[EOD]")
    held = _squad_held_titles()
    added = {}
    shutil.copyfile(source / "qa.train.bin", root / "qa.train.bin")
    for split in ("val", "test"):
        shutil.copyfile(source / f"qa.{split}.bin", root / f"qa.{split}.bin")
    with open(root / "qa.train.bin", "ab") as sink:
        added["hotpot"] = sum(encode_to(sink, tok, docs, eod) for docs in hotpot_docs(held))
        table = fetch_parquet("sentence-transformers/natural-questions", "pair/train-00000-of-00001.parquet")
        added["natural_questions"] = encode_to(sink, tok, [f"Q: {q}\nA: {a}" for q, a in
                                                           zip(table["query"].to_pylist(), table["answer"].to_pylist())], eod)
        table = fetch_parquet("pszemraj/simple_wikipedia", "train.parquet")
        added["simple_wikipedia"] = encode_to(sink, tok, [t for n, t in zip(table["title"].to_pylist(), table["text"].to_pylist())
                                                          if n not in held and t.strip()], eod)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_v2")
    parser.add_argument("--source", default="data_large")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-tokens", type=int, default=TRAIN_TOKENS)
    parser.add_argument("--held-tokens", type=int, default=HELD_TOKENS)
    args = parser.parse_args()
    root, source = Path(args.root), Path(args.source)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "tokenizer.json", root / "tokenizer.json")
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    maths = build_maths(root, tok, args.seed, args.train_tokens, args.held_tokens)
    print("maths", maths, flush=True)
    added = build_qa(root, source, tok)
    print("qa added", added, flush=True)
    stats = json.loads((source / "stats.json").read_text())
    stats["maths"] = maths
    stats["qa"]["train"] += sum(added.values())
    stats["qa_sources_added"] = added
    (root / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
