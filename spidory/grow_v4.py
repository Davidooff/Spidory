"""Grow every training set for data v4. Val, test and the tokenizer stay exactly as they are.

Sources, all with open licences:
- story: TinyStories V2 (CDLA-Sharing-1.0)
- qa: English Wikipedia, 4 of 41 files (CC-BY-SA-3.0), without SQuAD's held-out titles
- code: CodeSearchNet JavaScript, Java and Go (Apache-2.0); python-code-dataset-500k (MIT)
- maths: more generated items with a new seed; held-out questions are excluded by their hash
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from . import data_v2, easy_maths
from .data import STORY_URL, fetch_parquet
from .data_v2 import _squad_held_titles
from .easy_maths import render
from .grow_data import encode_to

WIKI = [f"20231101.en/train-0000{i}-of-00041.parquet" for i in range(4)]
CODE = {
    "Nan-Do/code-search-net-javascript": ["data/train-00000-of-00002-12a81a211c792ae8.parquet",
                                          "data/train-00001-of-00002-40eee81be3cabccb.parquet"],
    "Nan-Do/code-search-net-java": ["data/train-00000-of-00004-63114f67d19982c3.parquet",
                                    "data/train-00001-of-00004-dec5abb4b58c3245.parquet",
                                    "data/train-00002-of-00004-bda0da6730e1d6c6.parquet",
                                    "data/train-00003-of-00004-69402f93844e8a0d.parquet"],
    "Nan-Do/code-search-net-go": ["data/train-00000-of-00002-15a35db0141873f1.parquet",
                                  "data/train-00001-of-00002-5689caf3e5993ece.parquet"],
}
PYTHON_500K = ("jtatman/python-code-dataset-500k", ["data/train-00000-of-00002.parquet",
                                                    "data/train-00001-of-00002.parquet"])


def append_stories(root: Path, tok: Tokenizer, eod: int) -> int:
    import urllib.request

    total, tail = 0, ""
    with urllib.request.urlopen(STORY_URL.format("TinyStoriesV2-GPT4-train.txt")) as src, \
            open(root / "story.train.bin", "ab") as sink:
        while chunk := src.read(64 << 20):
            parts = (tail + chunk.decode("utf-8", errors="ignore")).split("<|endoftext|>")
            tail = parts.pop()
            total += encode_to(sink, tok, [p.strip() for p in parts if p.strip()], eod)
    return total


def append_wikipedia(root: Path, tok: Tokenizer, eod: int) -> int:
    held, total = _squad_held_titles(), 0
    with open(root / "qa.train.bin", "ab") as sink:
        for name in WIKI:
            table = fetch_parquet("wikimedia/wikipedia", name)
            docs = [t for title, t in zip(table["title"].to_pylist(), table["text"].to_pylist())
                    if title not in held and t.strip()]
            total += encode_to(sink, tok, docs, eod)
    return total


def append_code(root: Path, tok: Tokenizer, eod: int) -> int:
    total = 0
    with open(root / "code.train.bin", "ab") as sink:
        for repo, files in CODE.items():
            for name in files:
                total += encode_to(sink, tok, fetch_parquet(repo, name)["original_string"].to_pylist(), eod)
        repo, files = PYTHON_500K
        for name in files:
            table = fetch_parquet(repo, name)
            docs = [f"{i}\n{o}" for i, o in zip(table["instruction"].to_pylist(), table["output"].to_pylist())]
            total += encode_to(sink, tok, docs, eod)
    return total


def append_maths(root: Path, tok: Tokenizer, eod: int, tokens: int, seed: int) -> int:
    pools = easy_maths._pools()
    pilot = [i for batch in data_v2.generate("train", 400, seed, pools) for i in batch]
    mean = sum(len(e.ids) + 1 for e in tok.encode_batch([render(i) for i in pilot])) / len(pilot)
    rounds, total = int(tokens / mean / len(data_v2.LEVELS)), 0
    with open(root / "maths.train.bin", "ab") as sink:
        for batch in data_v2.generate("train", rounds, seed, pools):
            total += encode_to(sink, tok, [render(i) for i in batch], eod)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--maths-tokens", type=int, default=120_000_000)
    args = parser.parse_args()
    root = Path(args.root)
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    eod = tok.token_to_id("[EOD]")
    stats = json.loads((root / "stats.json").read_text())
    added = {}
    for name, grow in (("maths", lambda: append_maths(root, tok, eod, args.maths_tokens, seed=1)),
                       ("code", lambda: append_code(root, tok, eod)),
                       ("qa", lambda: append_wikipedia(root, tok, eod)),
                       ("story", lambda: append_stories(root, tok, eod))):
        added[name] = grow()
        stats[name]["train"] += added[name]
        print(name, "added", added[name], "total", stats[name]["train"], flush=True)
    stats["v4_added"] = added
    (root / "stats.json").write_text(json.dumps(stats, indent=2))
    (root / "V4_DONE").write_text(json.dumps(added))


if __name__ == "__main__":
    main()
