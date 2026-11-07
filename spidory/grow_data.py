"""Grow story and code training data. Val, test and the tokenizer stay exactly as they are."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .data import STORY_URL, fetch_parquet

CODE_FILES = ("data/train-00001-of-00004-648b3bede2edf6e6.parquet",
              "data/train-00002-of-00004-1dfd72b171e6b205.parquet",
              "data/train-00003-of-00004-184ab6d0e3c690b1.parquet")
CODE_REPO = "Nan-Do/code-search-net-python"
FIRST_CODE_FILE = "data/train-00000-of-00004-ee77a7de79eb2ab2.parquet"


def encode_to(sink, tok: Tokenizer, docs: list[str], eod: int) -> int:
    total = 0
    for start in range(0, len(docs), 5000):
        for enc in tok.encode_batch(docs[start:start + 5000]):
            ids = np.array(enc.ids + [eod], dtype=np.uint16)
            ids.tofile(sink)
            total += ids.size
    return total


def grow_stories(root: Path, tok: Tokenizer, eod: int, limit_bytes: int) -> int:
    import urllib.request

    tmp = root / "story.train.grow"
    total, left, tail = 0, limit_bytes, ""
    with urllib.request.urlopen(STORY_URL.format("TinyStories-train.txt")) as src, open(tmp, "wb") as sink:
        while left > 0:
            chunk = src.read(min(64 << 20, left))
            if not chunk:
                break
            left -= len(chunk)
            parts = (tail + chunk.decode("utf-8", errors="ignore")).split("<|endoftext|>")
            tail = parts.pop()
            total += encode_to(sink, tok, [p.strip() for p in parts if p.strip()], eod)
    tmp.rename(root / "story.train.bin")
    return total


def held_out_repos() -> set[str]:
    """The same val and test repositories that `data.fetch_code` chose."""
    repos = fetch_parquet(CODE_REPO, FIRST_CODE_FILE)["repo"].to_pylist()
    names = sorted(set(repos))
    random.Random(0).shuffle(names)
    return set(names[: max(2, len(names) // 20)])


def grow_code(root: Path, tok: Tokenizer, eod: int) -> int:
    held, added = held_out_repos(), 0
    with open(root / "code.train.bin", "ab") as sink:
        for name in CODE_FILES:
            table = fetch_parquet(CODE_REPO, name)
            docs = [d for d, r in zip(table["original_string"].to_pylist(), table["repo"].to_pylist())
                    if r not in held]
            added += encode_to(sink, tok, docs, eod)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--story-bytes", type=int, default=1_500_000_000)
    args = parser.parse_args()
    root = Path(args.root)
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    eod = tok.token_to_id("[EOD]")
    stats = json.loads((root / "stats.json").read_text())
    stats["story"]["train"] = grow_stories(root, tok, eod, args.story_bytes)
    print("story", stats["story"]["train"], flush=True)
    added = grow_code(root, tok, eod)
    stats["code"]["train"] += added
    print("code added", added, "total", stats["code"]["train"], flush=True)
    (root / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
