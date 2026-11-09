"""Exams for the playground: the same held-out tasks the evaluation uses, with a verdict per reply.

The v4 exam files were written on the rented machines and never copied back, so they are built
again here, once, with the same code and seed: `build_explain`, `build_mixed` and `build_compose`
from `spidory.mixed_data`, and the GSM8K test split. They are written to a cache folder under
runs/, never into the data folder.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from tokenizers import Tokenizer

# name -> (how it is graded, label for the page, file in the cache or the data folder)
EXAMS = {
    "maths": ("maths", "Maths, levels 2 and 5-9 (first line must be the answer)", "exam.json"),
    "mixed": ("maths", "Problem inside a story or passage (mixed topics)", "mixed_exam.json"),
    "explain": ("explain", "Maths explained in words (\"the answer is N\")", "explain_exam.json"),
    "compose": ("compose", "Write a problem as asked (checked by computing)", "compose_exam.json"),
    "gsm8k": ("word", "GSM8K test word problems", "gsm8k_exam.json"),
}
SOURCES = ("story", "maths", "qa")
HELD_TOKENS = 600_000  # as in the remote build: the mixed exam's random state depends on it


def build(data: Path, cache: Path, templates: Path, seed: int = 0, held_tokens: int = HELD_TOKENS,
          names: tuple[str, ...] = ("explain", "mixed", "compose", "gsm8k")) -> None:
    """Writes the wanted exam files into `cache`. Training sets are built empty: only the
    held-out splits, which the exams come from, are generated again."""
    from .. import mixed_data as md

    work = cache / "build"
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        for split in ("train", "val", "test"):
            link = work / f"{name}.{split}.bin"
            if not link.exists():
                # Training text is never read with a training target of 0, so the test file stands
                # in for it, which keeps memory low.
                link.symlink_to((data / f"{name}.{'test' if split == 'train' else split}.bin").resolve())
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    found = {}
    if "explain" in names:
        md.build_explain(work, tok, seed, 0, held_tokens)
        found["explain_exam.json"] = work / "explain_exam.json"
    if "mixed" in names or "compose" in names:
        group, _ = md.load_templates(templates)
        if "mixed" in names:
            md.build_mixed(work, tok, group, seed, 0, held_tokens)
            found["mixed_exam.json"] = work / "mixed_exam.json"
        if "compose" in names:
            md.build_compose(work, tok, group, seed, 0, held_tokens)
            found["compose_exam.json"] = work / "compose_exam.json"
    for file, path in found.items():
        (cache / file).write_text(path.read_text())
    if "gsm8k" in names:
        from ..data import fetch_parquet

        rows = fetch_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet").to_pylist()
        (cache / "gsm8k_exam.json").write_text(json.dumps(
            [{"prompt": f"Q: {r['question']}\nA:", "answer": md.gsm8k_reply(r["answer"])[1]} for r in rows]))


class Exams:
    """Loads each exam on first use, building the missing ones once."""

    def __init__(self, data: Path, cache: Path, templates: Path):
        self.data, self.cache, self.templates = data, cache, templates
        self.loaded: dict[str, list[dict]] = {}
        self.lock = threading.Lock()

    def _path(self, name: str) -> Path:
        file = EXAMS[name][2]
        return self.data / file if name == "maths" else self.cache / file

    def get(self, name: str) -> list[dict]:
        if name not in EXAMS:
            raise ValueError(f"unknown exam {name!r}")
        with self.lock:
            if name not in self.loaded:
                path = self._path(name)
                if not path.exists():
                    self.cache.mkdir(parents=True, exist_ok=True)
                    build(self.data, self.cache, self.templates, names=(name,))
                self.loaded[name] = json.loads(path.read_text())
            return self.loaded[name]

    def describe(self) -> list[dict]:
        """Names and labels, with sizes for the exams already on disk."""
        found = []
        for name, (kind, label, _) in EXAMS.items():
            path = self._path(name)
            size = len(self.loaded[name]) if name in self.loaded else (
                len(json.loads(path.read_text())) if path.exists() else None)
            found.append({"name": name, "kind": kind, "label": label, "size": size})
        return found
