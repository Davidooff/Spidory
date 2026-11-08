"""Mixed-topic data and maths with explanations, for per-token routing.

Two new datasets, written next to story, maths and qa:

- explain: the maths levels 2 and 5-9, with the question followed by "Explain step by step."
  and an answer that works through every operation before "The answer is N." Questions use
  the same hash split as the maths dataset, so no test expression of either set is trained on.
- mixed: one document holds several topics. Half the tokens are joins of pieces from story,
  maths and qa documents. The other half are insertions: a story or an encyclopedia-style
  passage with a small computation inside, from templates written by a language model and
  filled here, so every answer is computed, not written. Templates are split by id, so test
  passages come from templates never trained on.

Each mixed split also gets a labels file with one byte per token: which topic it belongs to
(0 story, 1 maths, 2 qa, 255 end of document), so loss can be measured per piece.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import easy_maths as em
from .data_v2 import LEVELS, generate

TOPICS = {"story": 0, "maths": 1, "qa": 2}
END = 255
NAMES = ("Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Lucy", "Jack", "Emma", "Leo", "Zoe",
         "Tim", "Sara", "Finn", "Ruby", "Noah", "Ella", "Owen", "Ivy", "Kai", "Nora", "Eli", "Maya")
PLACEHOLDERS = {"name", "name2", "a", "b", "c", "d"}
OPS = {ast.Add: ("+", operator.add), ast.Sub: ("-", operator.sub), ast.Mult: ("*", operator.mul)}
EXPLAIN = "Explain step by step."


# ----- explanations -----

def explain_expression(text: str) -> tuple[list[str], int]:
    """Every operation in evaluation order, as "x op y = z", and the value."""
    steps: list[str] = []

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = walk(node.left), walk(node.right)
            symbol, fn = OPS[type(node.op)]
            value = fn(left, right)
            steps.append(f"{left} {symbol} {right} = {value}")
            return value
        raise ValueError(f"unsupported expression: {text}")

    value = walk(ast.parse(text, mode="eval").body)
    return steps, value


def explain_variables(expression: str) -> tuple[list[str], int]:
    """Level 8, "a = 7, b = a + 5; b": each assignment with its values put in."""
    assignments, target = expression.split(";")
    values: dict[str, int] = {}
    steps = []
    for part in assignments.split(","):
        name, formula = (x.strip() for x in part.split("="))
        filled = re.sub(r"[a-z]", lambda m: str(values[m.group()]), formula)
        inner, value = explain_expression(filled)
        values[name] = value
        steps.append(f"{name} = {value}" if not inner else f"{name} = {filled} = {value}")
    return steps, values[target.strip()]


def explain_item(item: em.Item) -> str:
    steps, value = (explain_variables(item.expression) if item.level == 8
                    else explain_expression(item.expression))
    if value != item.answer:
        raise ValueError(f"explanation disagrees: {item.expression} gives {value}, not {item.answer}")
    body = ". ".join(steps) + "." if steps else f"It is {value}."
    return f"Q: {item.question} {EXPLAIN}\nA: {body} The answer is {value}."


def explain_prompt(item: em.Item) -> str:
    return f"Q: {item.question} {EXPLAIN}\nA:"


def answer_of(text: str) -> str | None:
    """The number after "The answer is", or None."""
    found = re.search(r"The answer is (-?\d+)", text)
    return found.group(1) if found else None


# ----- templates -----

def _split_of_id(template_id: str) -> str:
    share = int.from_bytes(hashlib.sha256(template_id.encode()).digest()[:8], "big") / 2 ** 64
    return "val" if share < 0.1 else "test" if share < 0.2 else "train"


def _formula_names(formula: str) -> set[str]:
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.Name, ast.Load, ast.Constant,
                                 ast.Add, ast.Sub, ast.Mult)):
            raise ValueError(f"formula may use only + - * and brackets: {formula}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, int):
            raise ValueError(f"formula constants must be whole numbers: {formula}")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def check_template(t: dict, highest: int = 5000, tries: int = 300) -> None:
    """Raises ValueError unless the template is usable and always gives a whole answer 0..highest."""
    for key in ("id", "kind", "context", "question", "answer", "vars"):
        if key not in t:
            raise ValueError(f"{t.get('id')}: missing {key}")
    if t["kind"] not in ("story", "fact"):
        raise ValueError(f"{t['id']}: kind must be story or fact")
    used = set(re.findall(r"\{(\w+)\}", t["context"] + t["question"]))
    if not used <= PLACEHOLDERS or re.sub(r"\{\w+\}", "", t["context"] + t["question"]).count("{"):
        raise ValueError(f"{t['id']}: unknown placeholders {used - PLACEHOLDERS}")
    numbers = set(re.findall(r"\{(\w+)\}", t["context"])) & {"a", "b", "c", "d"}
    names = _formula_names(t["answer"])
    if not names or not names <= numbers or set(t["vars"]) != numbers:
        raise ValueError(f"{t['id']}: formula, context numbers and vars disagree")
    for low, high in t["vars"].values():
        if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high):
            raise ValueError(f"{t['id']}: bad range")
    # Every corner of the ranges, then random values. The formula is checked above to hold only
    # numbers, names and + - *, so evaluating it is safe.
    keys = sorted(t["vars"])
    corners = [{k: t["vars"][k][(bits >> i) & 1] for i, k in enumerate(keys)} for bits in range(2 ** len(keys))]
    rng = random.Random(t["id"])
    code = compile(ast.parse(t["answer"], mode="eval"), "<formula>", "eval")
    for values in corners + [{k: rng.randint(*t["vars"][k]) for k in keys} for _ in range(tries)]:
        value = eval(code, {"__builtins__": {}}, values)  # noqa: S307
        if not 0 <= value <= highest:
            raise ValueError(f"{t['id']}: answer {value} out of range for {values}")


def load_templates(folder: Path) -> tuple[list[dict], list[str]]:
    """Valid templates from every .jsonl file, and the reasons others were dropped."""
    found, dropped, seen = [], [], set()
    for path in sorted(folder.glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                t = json.loads(line)
                check_template(t)
                if t["id"] in seen:
                    raise ValueError(f"{t['id']}: duplicate id")
                seen.add(t["id"])
                found.append(t)
            except (ValueError, SyntaxError, KeyError, TypeError) as error:
                dropped.append(f"{path.name}:{number}: {error}")
    return found, dropped


def fill(t: dict, rng: random.Random, explain: bool) -> tuple[str, str, int]:
    """(context, question and answer, value) for one random filling of a template."""
    values = {k: rng.randint(*v) for k, v in t["vars"].items()}
    name, name2 = rng.sample(NAMES, 2)
    words = {"name": name, "name2": name2, **{k: str(v) for k, v in values.items()}}
    context = re.sub(r"\{(\w+)\}", lambda m: words[m.group(1)], t["context"])
    question = re.sub(r"\{(\w+)\}", lambda m: words[m.group(1)], t["question"])
    filled = re.sub(r"\b[a-d]\b", lambda m: str(values[m.group()]), t["answer"])
    steps, value = explain_expression(filled)
    if explain:
        body = ". ".join(steps) + "." if steps else f"It is {value}."
        tail = f"\nQ: {question} {EXPLAIN}\nA: {body} The answer is {value}."
    else:
        tail = f"\nQ: {question}\nA: {value}"
    return context, tail, value


# ----- writing -----

class Writer:
    """Token ids and one topic label per token, written as two parallel files."""

    def __init__(self, root: Path, name: str, split: str, labelled: bool):
        self.ids = open(root / f"{name}.{split}.bin", "wb")
        self.labels = open(root / f"{name}.{split}.labels", "wb") if labelled else None
        self.count = 0

    def add(self, pieces: list[tuple[list[int], int]], eod: int) -> None:
        ids = [i for piece, _ in pieces for i in piece] + [eod]
        np.array(ids, dtype=np.uint16).tofile(self.ids)
        if self.labels is not None:
            labels = [label for piece, label in pieces for _ in piece] + [END]
            np.array(labels, dtype=np.uint8).tofile(self.labels)
        self.count += len(ids)

    def close(self) -> None:
        self.ids.close()
        if self.labels is not None:
            self.labels.close()


def build_explain(root: Path, tok: Tokenizer, seed: int, train_tokens: int, held_tokens: int,
                  exam_per_level: int = 100) -> dict[str, int]:
    eod = tok.token_to_id("[EOD]")
    pools = em._pools()
    counts = {}
    for split, target in (("train", train_tokens), ("val", held_tokens), ("test", held_tokens)):
        writer = Writer(root, "explain", split, labelled=False)
        rounds = 0
        while writer.count < target:
            for batch in generate(split, 2_000, seed + 1000 * (rounds + 1), pools):
                for enc, item in zip(tok.encode_batch([explain_item(i) for i in batch]), batch):
                    if writer.count >= target:
                        break
                    writer.add([(enc.ids, 1)], eod)
                if writer.count >= target:
                    break
            rounds += 1
        writer.close()
        counts[split] = writer.count
    # The exam: test-split questions only, one round gives one question of each level.
    exam = [i for batch in generate("test", exam_per_level, seed + 7, pools) for i in batch]
    picked = {lv: [i for i in exam if i.level == lv][:exam_per_level] for lv in LEVELS}
    rows = [{"prompt": explain_prompt(i), "answer": str(i.answer), "level": i.level}
            for group in zip(*picked.values()) for i in group]
    (root / "explain_exam.json").write_text(json.dumps(rows))
    return counts


def _doc_starts(stream: np.ndarray, eod: int) -> np.ndarray:
    return np.concatenate([[0], np.flatnonzero(stream == eod)[:-1] + 1])


def build_mixed(root: Path, tok: Tokenizer, templates: list[dict], seed: int, train_tokens: int,
                held_tokens: int, exam_size: int = 600) -> dict[str, int]:
    eod = tok.token_to_id("[EOD]")
    gap = tok.encode("\n\n").ids
    streams = {(n, s): np.memmap(root / f"{n}.{s}.bin", dtype=np.uint16, mode="r")
               for n in TOPICS for s in ("train", "val", "test")}
    counts, exam = {}, []
    for split, target in (("train", train_tokens), ("val", held_tokens), ("test", held_tokens)):
        rng = random.Random(f"mixed:{seed}:{split}")
        starts = {n: _doc_starts(np.asarray(streams[n, split]), eod) for n in TOPICS}
        group = [t for t in templates if _split_of_id(t["id"]) == split]
        if not group:
            raise ValueError(f"no templates for the {split} split")
        writer = Writer(root, "mixed", split, labelled=True)

        def piece(name: str, longest: int) -> list[int]:
            stream = streams[name, split]
            begin = int(starts[name][rng.randrange(len(starts[name]))])
            chunk = np.asarray(stream[begin:begin + longest])
            cut = np.flatnonzero(chunk == eod)
            return chunk[: cut[0] if len(cut) else len(chunk)].tolist()

        while writer.count < target:
            if rng.random() < 0.5:  # a join of pieces from different topics
                order = [rng.choice(list(TOPICS)) for _ in range(rng.randint(2, 4))]
                if len(set(order)) == 1:
                    order[-1] = rng.choice([n for n in TOPICS if n != order[0]])
                pieces = []
                for name in order:
                    ids = piece(name, rng.randint(24, 128))
                    if ids:
                        pieces += [(ids, TOPICS[name]), (gap, TOPICS[name])]
                writer.add(pieces[:-1], eod)
            else:  # an insertion: a passage with a computation inside
                t = rng.choice(group)
                context, tail, value = fill(t, rng, explain=rng.random() < 0.3)
                host = TOPICS["story"] if t["kind"] == "story" else TOPICS["qa"]
                pieces = [(tok.encode(context).ids, host), (tok.encode(tail).ids, TOPICS["maths"])]
                if rng.random() < 0.3:  # sometimes the topic changes first
                    lead = rng.choice([n for n in TOPICS if TOPICS[n] != host])
                    pieces = [(piece(lead, rng.randint(24, 96)), TOPICS[lead]), (gap, TOPICS[lead])] + pieces
                writer.add(pieces, eod)
        writer.close()
        counts[split] = writer.count
        if split == "test":
            exam_rng = random.Random(f"mixed-exam:{seed}")
            for index in range(exam_size):
                t = group[index % len(group)]
                text, value = exam_prompt(t, exam_rng)
                lead = ""
                if index % 2:  # half of the exam starts with a piece of another topic
                    name = exam_rng.choice(["story", "maths"] if t["kind"] == "fact" else ["qa", "maths"])
                    lead = tok.decode(piece(name, 64)) + "\n\n"
                exam.append({"prompt": lead + text, "answer": str(value), "kind": t["kind"], "lead": bool(lead)})
    (root / "mixed_exam.json").write_text(json.dumps(exam))
    return counts


def exam_prompt(t: dict, rng: random.Random) -> tuple[str, int]:
    """An exam prompt (passage, question, "A:") and its answer, from one fresh filling."""
    context, tail, value = fill(t, rng, explain=False)
    return context + tail[: tail.rindex("A:") + 2], value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="folder with story, maths and qa splits and tokenizer.json")
    parser.add_argument("--templates", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explain-tokens", type=int, default=150_000_000)
    parser.add_argument("--mixed-tokens", type=int, default=300_000_000)
    parser.add_argument("--held-tokens", type=int, default=600_000)
    args = parser.parse_args()
    root = Path(args.root)
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    templates, dropped = load_templates(Path(args.templates))
    print(f"{len(templates)} templates, {len(dropped)} dropped", flush=True)
    for reason in dropped:
        print("  dropped", reason)
    stats = {"templates": len(templates), "dropped": dropped,
             "explain": build_explain(root, tok, args.seed, args.explain_tokens, args.held_tokens),
             "mixed": build_mixed(root, tok, templates, args.seed, args.mixed_tokens, args.held_tokens)}
    (root / "mixed_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped"}), flush=True)


if __name__ == "__main__":
    main()
