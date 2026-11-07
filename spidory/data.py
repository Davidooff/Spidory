"""Download, tokenize and memory-map the four sources, and sample batches.

Dataset names exist here for measurement and round scheduling only. Training batches
returned by `Corpus.train_batch` are plain token tensors, so names cannot reach the loss.
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

DATASETS = ("story", "maths", "code", "qa")
SPLITS = ("train", "val", "test")
MATH_URL = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"
MATH_MODULES = (
    "arithmetic__add_or_sub",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__mixed",
)
STORY_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{}"
UNK = "[UNK]"


def _fetch_prefix(url: str, path: Path, max_bytes: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url) as src, open(tmp, "wb") as dst:
        left = max_bytes
        while left > 0:
            chunk = src.read(min(1 << 20, left))
            if not chunk:
                break
            dst.write(chunk)
            left -= len(chunk)
    tmp.rename(path)


def fetch_stories(raw: Path, max_bytes: int) -> dict[str, list[str]]:
    _fetch_prefix(STORY_URL.format("TinyStories-train.txt"), raw / "story_train.txt", max_bytes)
    _fetch_prefix(STORY_URL.format("TinyStories-valid.txt"), raw / "story_valid.txt", max_bytes)

    def read(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]

    held = read(raw / "story_valid.txt")
    return {"train": read(raw / "story_train.txt"), "val": held[::2], "test": held[1::2]}


def fetch_maths(raw: Path) -> None:
    """Stream the archive and keep only the wanted files, so 2.3 GB is never stored."""
    out = raw / "maths"
    wanted = {f"mathematics_dataset-v1.0/{folder}/{m}.txt"
              for folder in ("train-easy", "interpolate") for m in MATH_MODULES}
    if out.exists() and len(list(out.glob("*.txt"))) == len(wanted):
        return
    out.mkdir(parents=True, exist_ok=True)
    left = set(wanted)
    with urllib.request.urlopen(MATH_URL) as stream, tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            if member.name in left:
                parts = member.name.split("/")
                data = tar.extractfile(member).read()
                (out / f"{parts[1]}__{parts[2]}").write_bytes(data)
                left.remove(member.name)
                if not left:
                    break
    if left:
        raise RuntimeError(f"maths files missing from archive: {sorted(left)}")


def read_maths(raw: Path) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    def pairs(folder: str) -> list[tuple[str, str]]:
        found = []
        for module in MATH_MODULES:
            lines = (raw / "maths" / f"{folder}__{module}.txt").read_text().splitlines()
            found += [(q.strip(), a.strip()) for q, a in zip(lines[::2], lines[1::2])]
        return found

    render = lambda q, a: f"Q: {q}\nA: {a}"
    rng = random.Random(0)
    train, held = pairs("train-easy"), pairs("interpolate")
    rng.shuffle(train)
    rng.shuffle(held)
    half = len(held) // 2
    val, test = held[:half], held[half:]
    docs = {"train": [render(q, a) for q, a in train],
            "val": [render(q, a) for q, a in val],
            "test": [render(q, a) for q, a in test]}
    exam = [{"prompt": f"Q: {q}\nA:", "answer": a} for q, a in test[:2000]]
    return docs, exam


def fetch_parquet(repo: str, filename: str) -> "object":
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset"))


def fetch_code() -> dict[str, list[str]]:
    table = fetch_parquet("Nan-Do/code-search-net-python",
                          "data/train-00000-of-00004-ee77a7de79eb2ab2.parquet")
    docs = table["original_string"].to_pylist()
    repos = table["repo"].to_pylist()
    # Split by repository so held-out code does not come from the training repositories.
    names = sorted(set(repos))
    random.Random(0).shuffle(names)
    cut = max(2, len(names) // 20)
    val_repos, test_repos = set(names[:cut // 2]), set(names[cut // 2:cut])
    split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for doc, repo in zip(docs, repos):
        key = "val" if repo in val_repos else "test" if repo in test_repos else "train"
        split[key].append(doc)
    return split


def fetch_qa() -> dict[str, list[str]]:
    def render(table) -> list[tuple[str, str]]:
        found = []
        for title, ctx, q, ans in zip(table["title"].to_pylist(), table["context"].to_pylist(),
                                      table["question"].to_pylist(), table["answers"].to_pylist()):
            found.append((title, f"{ctx}\nQ: {q}\nA: {ans['text'][0]}"))
        return found

    train = render(fetch_parquet("rajpurkar/squad", "plain_text/train-00000-of-00001.parquet"))
    held = render(fetch_parquet("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet"))
    titles = sorted({t for t, _ in held})
    val_titles = set(titles[::2])
    random.Random(0).shuffle(train)
    return {"train": [d for _, d in train],
            "val": [d for t, d in held if t in val_titles],
            "test": [d for t, d in held if t not in val_titles]}


def _cap(docs: list[str], max_chars: int) -> list[str]:
    kept, total = [], 0
    for doc in docs:
        if total >= max_chars:
            break
        kept.append(doc)
        total += len(doc)
    return kept


def train_tokenizer(sample: list[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(sample, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[UNK, "[EOD]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def prepare(root: Path, vocab_size: int, max_chars: int, held_chars: int) -> None:
    raw = root / "raw"
    fetch_maths(raw)
    maths, exam = read_maths(raw)
    sources = {"story": fetch_stories(raw, max_chars + (1 << 20)), "maths": maths,
               "code": fetch_code(), "qa": fetch_qa()}
    capped = {name: {split: _cap(docs, max_chars if split == "train" else held_chars)
                     for split, docs in by_split.items()}
              for name, by_split in sources.items()}

    # Equal text from each source, so no dataset dominates the shared vocabulary.
    sample = [d for name in DATASETS for d in _cap(capped[name]["train"], 3_000_000)]
    tok = train_tokenizer(sample, vocab_size)
    tok.save(str(root / "tokenizer.json"))
    eod = tok.token_to_id("[EOD]")

    stats: dict[str, dict[str, int]] = {}
    for name in DATASETS:
        stats[name] = {}
        for split in SPLITS:
            ids: list[np.ndarray] = []
            docs = capped[name][split]
            for start in range(0, len(docs), 2000):
                for enc in tok.encode_batch(docs[start:start + 2000]):
                    ids.append(np.array(enc.ids + [eod], dtype=np.uint16))
            merged = np.concatenate(ids)
            merged.tofile(root / f"{name}.{split}.bin")
            stats[name][split] = int(merged.size)
            print(f"{name:6s} {split:5s} docs={len(docs):8d} tokens={merged.size:10d}", flush=True)
    (root / "exam.json").write_text(json.dumps(exam))
    (root / "stats.json").write_text(json.dumps(stats, indent=2))


class Corpus:
    """Memory-mapped token streams, sampled as fixed-length windows."""

    def __init__(self, root: str | Path, seq_len: int, seed: int = 0):
        self.root = Path(root)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.tokens = {(n, s): np.memmap(self.root / f"{n}.{s}.bin", dtype=np.uint16, mode="r")
                       for n in DATASETS for s in SPLITS}

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _windows(self, stream: np.ndarray, starts: np.ndarray) -> torch.Tensor:
        rows = np.stack([stream[s:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(rows)

    def train_batch(self, size: int, focus: str | None = None, focus_share: float = 0.5) -> torch.Tensor:
        """Mixed batch. With `focus`, that share of rows comes from one dataset."""
        names = [DATASETS[i] for i in self.rng.integers(len(DATASETS), size=size)]
        if focus is not None:
            names = [focus if self.rng.random() < focus_share else n for n in names]
        rows = []
        for name in names:
            stream = self.tokens[name, "train"]
            start = self.rng.integers(len(stream) - self.seq_len - 1, size=1)
            rows.append(self._windows(stream, start))
        return torch.cat(rows)

    def fixed_windows(self, name: str, split: str, count: int) -> torch.Tensor:
        """Evenly spaced windows, identical on every call, so comparisons are fair."""
        stream = self.tokens[name, split]
        starts = np.linspace(0, len(stream) - self.seq_len - 2, count).astype(np.int64)
        return self._windows(stream, starts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-chars", type=int, default=16_000_000)
    parser.add_argument("--held-chars", type=int, default=1_500_000)
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    prepare(Path(args.root), args.vocab_size, args.max_chars, args.held_chars)
