"""Data v5: drop every training document that shares a passage with validation or test data.

Works on the token files directly, so every source is covered at once, whatever its titles:
for each dataset, every n-token window of its val and test streams is hashed, and any training
document (documents end with [EOD]) containing one of those windows is removed. Story uses
24-token windows, because short phrases repeat across children's stories; QA and code use 16.
Maths is already split by question, so it is left as it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

WINDOWS = {"story": 24, "qa": 16, "code": 16}
BASE = np.uint64(1_000_003)


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """A 64-bit hash of every window of n tokens: one value per start position."""
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    values = ids.astype(np.uint64) + np.uint64(1)
    count = len(ids) - n + 1
    out = np.zeros(count, dtype=np.uint64)
    power = np.uint64(1)
    with np.errstate(over="ignore"):
        for j in range(n):
            out += values[j:j + count] * power
            power *= BASE
    return out


def held_hashes(streams: list[np.ndarray], n: int) -> np.ndarray:
    return np.unique(np.concatenate([ngram_hashes(s, n) for s in streams]))


def leaking_documents(train: np.ndarray, held: np.ndarray, n: int, eod: int, chunk: int = 20_000_000) -> np.ndarray:
    """Indexes of training documents that contain any held-out window."""
    doc_of = np.cumsum(train == eod) - (train == eod)  # document index of each token
    bad = []
    for start in range(0, len(train), chunk):
        piece = train[start:start + chunk + n - 1]
        hit = np.isin(ngram_hashes(piece, n), held)
        if hit.any():
            bad.append(np.unique(doc_of[start + np.nonzero(hit)[0]]))
    return np.unique(np.concatenate(bad)) if bad else np.zeros(0, dtype=np.int64)


def drop_documents(train: np.ndarray, bad: np.ndarray, eod: int) -> np.ndarray:
    doc_of = np.cumsum(train == eod) - (train == eod)
    return train[~np.isin(doc_of, bad)]


def clean(root: Path, eod: int) -> dict:
    report = {}
    for name, n in WINDOWS.items():
        train = np.fromfile(root / f"{name}.train.bin", dtype=np.uint16)
        held = held_hashes([np.fromfile(root / f"{name}.{s}.bin", dtype=np.uint16) for s in ("val", "test")], n)
        bad = leaking_documents(train, held, n, eod)
        kept = drop_documents(train, bad, eod)
        documents = int((train == eod).sum())
        tmp = root / f"{name}.train.bin.tmp"
        kept.tofile(tmp)
        tmp.rename(root / f"{name}.train.bin")
        report[name] = {"documents": documents, "dropped_documents": int(len(bad)),
                        "tokens_before": int(len(train)), "tokens_after": int(len(kept))}
        print(name, report[name], flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    root = Path(args.root)
    eod = Tokenizer.from_file(str(root / "tokenizer.json")).token_to_id("[EOD]")
    report = clean(root, eod)
    stats = json.loads((root / "stats.json").read_text())
    for name, found in report.items():
        stats[name]["train"] = found["tokens_after"]
    stats["v5_cleaning"] = report
    (root / "stats.json").write_text(json.dumps(stats, indent=2))
    (root / "V5_DONE").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
