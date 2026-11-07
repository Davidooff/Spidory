"""Generate the easy maths dataset: five levels, from one operation to several sequential steps.

Every expression belongs to exactly one split, so held-out questions are never seen in training.
Run with `python -m spidory.easy_maths --root data_large`. Only the maths files, exam.json and the
"maths" entry of stats.json change; the tokenizer is reused, never retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import random
import shutil
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tokenizers import Tokenizer

SPLITS = ("train", "val", "test")
LEVELS = (1, 2, 3, 4, 5)
WORDINGS = ("What is {}?", "Calculate {}.", "Compute {}.", "{} =")
HELD_SHARE = 0.05  # Of the unique expressions, for each of val and test.
TRAIN_TOKENS = 60_000_000
HELD_TOKENS = 590_000
EXAM_PER_LEVEL = 600
PILOT_ITEMS = 2000
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
CHAIN_MAX = 20  # Largest number in the chains of levels 3 and 4.
LONG_CHAINS = (4, 5)  # Too many of these to list, so level 4 is sampled instead of enumerated.


class Item(NamedTuple):
    level: int
    family: str  # Sub-type sampled equally within a level, such as one operator.
    expression: str
    answer: int
    question: str


def render(item: Item) -> str:
    return f"Q: {item.question}\nA: {item.answer}"


def prompt(item: Item) -> str:
    return f"Q: {item.question}\nA:"


def _hash(expression: str) -> int:
    return int.from_bytes(hashlib.sha256(expression.encode()).digest()[:8], "big")


def _split_of(expression: str) -> str:
    """Split for expressions too many to list, using the same hash as the listed ones."""
    share = _hash(expression) / 2 ** 64
    return "val" if share < HELD_SHARE else "test" if share < 2 * HELD_SHARE else "train"


def _chain(numbers: list[int], ops: list[str]) -> tuple[str, int]:
    """Text and left-to-right value of a chain of numbers."""
    text, value = str(numbers[0]), numbers[0]
    for op, number in zip(ops, numbers[1:]):
        text += f" {op} {number}"
        value = OPS[op](value, number)
    return text, value


def _level_1() -> dict[str, list[tuple[str, int]]]:
    return {op: [_chain([a, b], [op]) for a in range(10) for b in range(10)] for op in "+-*"}


def _level_2() -> dict[str, list[tuple[str, int]]]:
    # Two 1-digit numbers would be a level 1 expression, so each expression has one level.
    wide = [(a, b) for a in range(100) for b in range(100) if a >= 10 or b >= 10]
    factor = [(a, b) for a, b in wide if a < 10 or b < 10]
    return {"+": [_chain([a, b], ["+"]) for a, b in wide],
            "-": [_chain([a, b], ["-"]) for a, b in wide],
            "*": [_chain([a, b], ["*"]) for a, b in factor]}


def _level_3() -> dict[str, list[tuple[str, int]]]:
    span = range(CHAIN_MAX + 1)
    return {"chain3": [_chain([a, b, c], [x, y])
                       for a in span for b in span for c in span for x in "+-" for y in "+-"]}


def _level_5() -> dict[str, list[tuple[str, int]]]:
    found: dict[str, list[tuple[str, int]]] = {}
    for a in range(10):
        for b in range(10):
            for c in range(10):
                for op in "+-":
                    step = OPS[op]
                    forms = {f"({a} {op} {b}) * {c}": step(a, b) * c,
                             f"{a} * ({b} {op} {c})": a * step(b, c),
                             f"{a} {op} {b} * {c}": step(a, b * c),
                             f"{a} * {b} {op} {c}": step(a * b, c)}
                    for name, entry in zip(("left", "right", "late", "early"), forms.items()):
                        found.setdefault(f"{name}{op}", []).append(entry)
    return found


@cache
def _pools() -> dict[tuple[int, str, str], tuple[tuple[str, int], ...]]:
    """(level, family, split) to its expressions. Hash rank keeps every family in every split."""
    pools = {}
    for level, families in ((1, _level_1()), (2, _level_2()), (3, _level_3()), (5, _level_5())):
        for family, entries in families.items():
            ranked = sorted(entries, key=lambda entry: _hash(entry[0]))
            held = max(1, round(len(ranked) * HELD_SHARE))
            pools[level, family, "val"] = tuple(ranked[:held])
            pools[level, family, "test"] = tuple(ranked[held:2 * held])
            pools[level, family, "train"] = tuple(ranked[2 * held:])
    return pools


def _long_chain(rng: random.Random, length: int, split: str) -> tuple[str, int]:
    while True:
        text, value = _chain([rng.randint(0, CHAIN_MAX) for _ in range(length)],
                             [rng.choice("+-") for _ in range(length - 1)])
        if _split_of(text) == split:
            return text, value


def generate_split(split: str, per_level: int, seed: int) -> list[Item]:
    """`per_level` items of each level, shuffled. Train and held-out items may repeat."""
    rng = random.Random(f"{seed}:{split}")
    families: dict[int, list[str]] = {level: [] for level in LEVELS}
    for level, family, name in _pools():
        if name == split:
            families[level].append(family)
    families[4] = [f"chain{length}" for length in LONG_CHAINS]

    items = []
    for level in LEVELS:
        names = families[level]
        for i in range(per_level):
            family = names[i % len(names)]
            if level == 4:
                entry = _long_chain(rng, int(family.removeprefix("chain")), split)
            else:
                entry = rng.choice(_pools()[level, family, split])
            question = rng.choice(WORDINGS).format(entry[0])
            items.append(Item(level, family, entry[0], entry[1], question))
    rng.shuffle(items)
    return items


def build_exam(test: list[Item], per_level: int, seed: int) -> list[dict]:
    """Balanced exam, with every prefix balanced to within one item per level.

    Prompts repeat only where a level has too few unique ones, which happens at level 1.
    """
    rng = random.Random(f"{seed}:exam")
    rows = []
    for level in LEVELS:
        by_family: dict[str, set[Item]] = {}
        for item in test:
            if item.level == level:
                by_family.setdefault(item.family, set()).add(item)
        chosen = []
        for i, family in enumerate(sorted(by_family)):
            unique = sorted(by_family[family])
            rng.shuffle(unique)
            wanted = per_level // len(by_family) + (i < per_level % len(by_family))
            chosen += [unique[n % len(unique)] for n in range(wanted)]
        rng.shuffle(chosen)
        rows.append(chosen)

    exam = []
    for block in zip(*rows, strict=True):
        order = list(block)
        rng.shuffle(order)
        exam += [{"prompt": prompt(i), "answer": str(i.answer), "level": i.level} for i in order]
    return exam


def check_splits(items: dict[str, list[Item]]) -> None:
    seen = {split: {i.expression for i in items[split]} for split in SPLITS}
    if seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"]:
        raise ValueError("an expression appears in two splits")
    for split in SPLITS:
        if {i.level for i in items[split]} != set(LEVELS):
            raise ValueError(f"a level is missing from {split}")


def encode(tok: Tokenizer, items: list[Item], eod: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(items), 50_000):
        docs = [render(i) for i in items[start:start + 50_000]]
        flat = [t for enc in tok.encode_batch(docs) for t in (*enc.ids, eod)]
        chunks.append(np.array(flat, dtype=np.uint16))
    return np.concatenate(chunks)


def _items_for_tokens(tok: Tokenizer, split: str, tokens: int, seed: int) -> int:
    """Items per level that give about `tokens`, measured on a seeded pilot sample."""
    pilot = generate_split(split, PILOT_ITEMS, seed)
    mean = encode(tok, pilot, 0).size / len(pilot)
    return max(1, round(tokens / mean / len(LEVELS)))


def _backup(root: Path) -> None:
    """Keep the first version of each file; later runs never overwrite the backup."""
    names = [f"maths.{split}.bin" for split in SPLITS] + ["exam.json", "stats.json"]
    folder = root / "old_maths"
    folder.mkdir(exist_ok=True)
    for name in names:
        source, target = root / name, folder / name
        if source.exists() and not target.exists():
            part = target.with_name(name + ".part")
            shutil.copy2(source, part)
            os.replace(part, target)


def _write_text(path: Path, text: str) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(text)
    os.replace(part, path)


def build(root: Path, seed: int, train_tokens: int = TRAIN_TOKENS, held_tokens: int = HELD_TOKENS,
          exam_per_level: int = EXAM_PER_LEVEL) -> dict[str, int]:
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    eod = tok.token_to_id("[EOD]")
    targets = {"train": train_tokens, "val": held_tokens, "test": held_tokens}
    items = {s: generate_split(s, _items_for_tokens(tok, s, targets[s], seed), seed) for s in SPLITS}
    check_splits(items)
    exam = build_exam(items["test"], exam_per_level, seed)
    streams = {split: encode(tok, items[split], eod) for split in SPLITS}

    # Nothing is replaced until all outputs exist, so a failed run leaves the old dataset whole.
    _backup(root)
    stats_path = root / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats["maths"] = {split: int(streams[split].size) for split in SPLITS}
    for split, stream in streams.items():
        part = root / f"maths.{split}.bin.part"
        stream.tofile(part)
        os.replace(part, root / f"maths.{split}.bin")
        print(f"maths {split:5s} items={len(items[split]):8d} tokens={stream.size:10d}", flush=True)
    _write_text(root / "exam.json", json.dumps(exam))
    _write_text(stats_path, json.dumps(stats, indent=2))
    return stats["maths"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_large")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-tokens", type=int, default=TRAIN_TOKENS)
    parser.add_argument("--held-tokens", type=int, default=HELD_TOKENS)
    args = parser.parse_args()
    build(Path(args.root), args.seed, args.train_tokens, args.held_tokens)
