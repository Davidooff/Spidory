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



# ----- explanations in words -----

WORD_OPS = {"+": "plus", "-": "minus", "*": "times"}
SAY = {
    "+": ("{x} plus {y} is {z}", "adding {y} to {x} gives {z}", "{x} and {y} make {z}",
          "if we add {y} to {x}, we get {z}"),
    "-": ("{x} minus {y} is {z}", "taking {y} away from {x} leaves {z}", "subtracting {y} from {x} gives {z}",
          "{x} take away {y} is {z}"),
    "*": ("{x} times {y} is {z}", "multiplying {x} by {y} gives {z}", "{y} groups of {x} make {z}",
          "{x} multiplied by {y} is {z}"),
}
RUNNING = {  # a step that continues from the previous result
    "+": ("adding {y} gives {z}", "we add {y} and get {z}", "with {y} more we have {z}"),
    "-": ("taking away {y} leaves {z}", "we subtract {y} and get {z}", "with {y} less we have {z}"),
    "*": ("multiplying by {y} gives {z}", "we multiply by {y} and get {z}"),
}
NEXT = ("Then", "Next,", "After that,", "Now")
LAST = ("Finally,", "Last,", "At the end,")
LEFT_TO_RIGHT = ("There are only additions and subtractions, so we go from left to right.",
                 "We work from left to right, one number at a time.",
                 "We take the numbers in order, from left to right.")
BRACKETS = ("The part in brackets comes first:", "We start with the brackets:",
            "First we work out what is inside the brackets:")
PRECEDENCE = ("Multiplication comes before addition and subtraction, so first",
              "We multiply before we add or subtract:", "The multiplication goes first:")
CLOSE = ("So the answer is {z}.", "That means the answer is {z}.", "The answer is {z}.")


def _steps_with_reasons(text: str) -> tuple[list[tuple[int, str, int, int, str | None, bool]], int]:
    """(left, op, right, value, reason, continues) for every operation, in evaluation order.

    reason is "brackets" for a step written inside brackets, "precedence" for a multiplication
    done before the addition or subtraction around it, else None. continues is True when the
    step's left side is the result of the step just before it.
    """
    found = []

    def walk(node, parent_op=None):
        if isinstance(node, ast.Constant):
            return node.value, False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value, _ = walk(node.operand)
            return -value, False
        left, left_is_step = walk(node.left, node.op)
        before = len(found)
        right, _ = walk(node.right, node.op)
        symbol, fn = OPS[type(node.op)]
        value = fn(left, right)
        start, end = node.col_offset, node.end_col_offset
        bracketed = start > 0 and text[start - 1] == "(" and end < len(text) and text[end] == ")"
        reason = ("brackets" if bracketed else
                  "precedence" if symbol == "*" and isinstance(parent_op, (ast.Add, ast.Sub)) else None)
        continues = left_is_step and len(found) == before  # nothing was computed on the right
        found.append((left, symbol, right, value, reason, continues))
        return value, True

    value, _ = walk(ast.parse(text, mode="eval").body)
    return found, value


def _say(rng: random.Random, step) -> str:
    left, symbol, right, value, _, continues = step
    bank = RUNNING[symbol] if continues and rng.random() < 0.6 else SAY[symbol]
    return rng.choice(bank).format(x=left, y=right, z=value)


def words_for_expression(text: str, rng: random.Random) -> tuple[str, int]:
    """An explanation in sentences, with why each step comes when it does."""
    steps, value = _steps_with_reasons(text)
    if not steps:
        return f"It is just {value}.", value
    if len(steps) == 1:
        line = _say(rng, steps[0])
        return line[0].upper() + line[1:] + ".", value
    sentences = []
    if not any(s[4] for s in steps) and all(s[1] in "+-" for s in steps):
        sentences.append(rng.choice(LEFT_TO_RIGHT))
    for index, step in enumerate(steps):
        line = _say(rng, step)
        if step[4] == "brackets" and not any(s[4] == "brackets" for s in steps[:index]):
            sentences.append(f"{rng.choice(BRACKETS)} {line}.")
        elif step[4] == "precedence":
            lead = rng.choice(PRECEDENCE)
            sentences.append(f"{lead} {line}." if lead.endswith(("first", ":")) else f"{lead}, {line}.")
        elif index == 0:
            sentences.append(f"First, {line}.")
        else:
            lead = rng.choice(LAST if index == len(steps) - 1 else NEXT)
            sentences.append(f"{lead} {line}.")
    return " ".join(sentences), value


def words_for_variables(expression: str, rng: random.Random) -> tuple[str, int]:
    """Level 8, "a = 7, b = a + 5; b": each letter in order, in sentences."""
    assignments, target = expression.split(";")
    values: dict[str, int] = {}
    sentences = [rng.choice(("We work out each letter in order.", "We find the letters one by one.",
                             "Let us go through the letters in order."))]
    for part in assignments.split(","):
        name, formula = (x.strip() for x in part.split("="))
        if formula.lstrip("-").isdigit():
            values[name] = int(formula)
            sentences.append(rng.choice(("{n} is {v}.", "We are told {n} is {v}.", "{n} starts at {v}.")).format(
                n=name, v=formula))
            continue
        spoken = " ".join(WORD_OPS.get(token, token) for token in formula.split())
        known = [t for t in formula.split() if t.isalpha()]
        filled = re.sub(r"[a-z]", lambda m: str(values[m.group()]), formula)
        steps, value = explain_expression(filled)
        values[name] = value
        given = " and ".join(f"{t} is {values[t]}" for t in dict.fromkeys(known))
        left, symbol, right = filled.split()
        sentences.append(rng.choice((
            "{n} is {s}. Since {g}, {calc}, so {n} is {v}.",
            "{n} is {s}, and {g}, so {n} is {calc_short}.",
            "For {n} we take {s}: {calc}. So {n} is {v}.",
        )).format(n=name, s=spoken, g=given, v=value,
                  calc=SAY[symbol][0].format(x=left, y=right, z=value),
                  calc_short=f"{left} {WORD_OPS[symbol]} {right}, which is {value}"))
    return " ".join(sentences), values[target.strip()]


def explain_item(item: em.Item, rng: random.Random | None = None) -> str:
    """The question, then an explanation in words, then "the answer is N"."""
    rng = rng or random.Random(item.question)
    body, value = (words_for_variables(item.expression, rng) if item.level == 8
                   else words_for_expression(item.expression, rng))
    if value != item.answer:
        raise ValueError(f"explanation disagrees: {item.expression} gives {value}, not {item.answer}")
    return f"Q: {item.question} {EXPLAIN}\nA: {body} {rng.choice(CLOSE).format(z=value)}"


def explain_prompt(item: em.Item) -> str:
    return f"Q: {item.question} {EXPLAIN}\nA:"


def answer_of(text: str) -> str | None:
    """The number after "The answer is", or None."""
    found = re.search(r"[Tt]he answer is (-?\d+)", text)
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
    if explain:
        body, value = words_for_expression(filled, rng)
        tail = f"\nQ: {question} {EXPLAIN}\nA: {body} {rng.choice(CLOSE).format(z=value)}"
    else:
        _, value = explain_expression(filled)
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
        rng = random.Random(f"explain-words:{seed}:{split}")
        rounds = 0
        while writer.count < target:
            for batch in generate(split, 2_000, seed + 1000 * (rounds + 1), pools):
                for enc, item in zip(tok.encode_batch([explain_item(i, rng) for i in batch]), batch):
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


# ----- compose: the model writes the problem -----

COMPOSE_STORY = (
    "Write a short story problem that uses the numbers {nums} and has the answer {ans}.",
    "Make up a story with a question. Use the numbers {nums}. The answer must be {ans}.",
    "Create a word problem for children with the numbers {nums}, where the answer is {ans}.",
)
COMPOSE_FACT = (
    "Write a short encyclopedia-style passage with a question. Use the numbers {nums}. The answer must be {ans}.",
    "Make up a passage about an imaginary place with the numbers {nums}, and a question whose answer is {ans}.",
)
COMPOSE_SUM = {
    "+": ("Make up a sum of {n} numbers that equals {t}.", "Write {n} numbers that add up to {t}.",
          "Write an addition with {n} numbers whose result is {t}."),
    "-": ("Write a subtraction whose result is {t}.", "Make up a subtraction that gives {t}."),
    "*": ("Write a multiplication of two numbers that equals {t}.", "Make up a product of two numbers equal to {t}."),
    "mix": ("Write an expression with {n} numbers, using both addition and subtraction, that equals {t}.",
            "Make up a calculation with {n} numbers, with at least one plus and one minus, giving {t}."),
    "brackets": ("Write an expression with brackets that equals {t}.",
                 "Make up a calculation that uses brackets and gives {t}."),
}
COMPOSE_QUESTION = ("Make up a maths question whose answer is {t}.", "Write a question that has the answer {t}.",
                    "Invent a short arithmetic question with the answer {t}.")


def _numbers_text(values: list[int]) -> str:
    return ", ".join(map(str, values[:-1])) + " and " + str(values[-1]) if len(values) > 1 else str(values[0])


def compose_story_item(t: dict, rng: random.Random) -> dict:
    """A request to write a problem with given numbers and answer, and a correct reply."""
    context, tail, value = fill(t, rng, explain=False)
    numbers = [int(n) for n in re.findall(r"\d+", context)]
    used = sorted(set(numbers), key=numbers.index)
    bank = COMPOSE_STORY if t["kind"] == "story" else COMPOSE_FACT
    request = rng.choice(bank).format(nums=_numbers_text(used), ans=value)
    return {"kind": "story", "prompt": f"Task: {request}\nProblem:", "reply": f" {context}{tail}",
            "numbers": used, "answer": value}


def _expression(rng: random.Random, kind: str) -> tuple[str, int, int]:
    if kind == "+":
        n = rng.randint(2, 4)
        parts = [rng.randint(1, 30) for _ in range(n)]
        return " + ".join(map(str, parts)), sum(parts), n
    if kind == "-":
        a = rng.randint(5, 60)
        b = rng.randint(1, a)
        return f"{a} - {b}", a - b, 2
    if kind == "*":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return f"{a} * {b}", a * b, 2
    if kind == "mix":
        while True:
            n = rng.randint(3, 5)
            ops = [rng.choice("+-") for _ in range(n - 1)]
            if "+" in ops and "-" in ops:
                break
        text = str(rng.randint(1, 20)) + "".join(f" {op} {rng.randint(1, 20)}" for op in ops)
        return text, explain_expression(text)[1], n
    inner_op, outer_op = rng.choice("+-"), rng.choice("*+-")
    text = f"({rng.randint(1, 15)} {inner_op} {rng.randint(1, 9)}) {outer_op} {rng.randint(2, 9)}"
    return text, explain_expression(text)[1], 3


def compose_expression_item(rng: random.Random) -> dict:
    kind = rng.choice(list(COMPOSE_SUM))
    text, value, n = _expression(rng, kind)
    request = rng.choice(COMPOSE_SUM[kind]).format(n=n, t=value)
    return {"kind": kind, "prompt": f"Task: {request}\nAnswer:", "reply": f" {text} = {value}",
            "count": n, "answer": value}


def compose_question_item(item: em.Item, rng: random.Random) -> dict:
    request = rng.choice(COMPOSE_QUESTION).format(t=item.answer)
    return {"kind": "question", "prompt": f"Task: {request}\nQuestion:",
            "reply": f" {item.question}\nAnswer: {item.answer}", "answer": item.answer}


def grade_compose(item: dict, text: str) -> bool:
    """Whether a model's reply does what the task asked. Checked by computing, not by matching."""
    text = text.strip()
    if item["kind"] == "story":
        numbers = [int(n) for n in re.findall(r"\d+", text)]
        last = re.findall(r"A: (-?\d+)", text)
        return bool(last) and int(last[-1]) == item["answer"] and all(n in numbers for n in item["numbers"])
    if item["kind"] == "question":
        line = text.split("\n")[0].strip()
        try:
            letters = re.fullmatch(r"(.*=.*)\. What is ([a-z])\?", line)
            if letters:  # "a = 7, b = a + 5. What is b?"
                return explain_variables(f"{letters.group(1)}; {letters.group(2)}")[1] == item["answer"]
            expression = re.sub(r"^(What is|Calculate|Compute)\s+", "", line).rstrip("?.= ").strip()
            return explain_expression(expression)[1] == item["answer"]
        except (ValueError, SyntaxError, KeyError):
            return False
    line = text.split("\n")[0]
    if "=" not in line:
        return False
    left, right = line.rsplit("=", 1)
    try:
        value = explain_expression(left.strip())[1]
    except (ValueError, SyntaxError):
        return False
    if right.strip() != str(item["answer"]) or value != item["answer"]:
        return False
    numbers = re.findall(r"\d+", left)
    ops = set(re.findall(r"[+*-]", left))
    return {
        "+": ops == {"+"} and len(numbers) == item.get("count", len(numbers)),
        "-": ops == {"-"},
        "*": ops == {"*"} and len(numbers) == 2,
        "mix": {"+", "-"} <= ops and len(numbers) == item.get("count", len(numbers)),
        "brackets": "(" in left and ")" in left,
    }[item["kind"]]


def build_compose(root: Path, tok: Tokenizer, templates: list[dict], seed: int, train_tokens: int,
                  held_tokens: int, exam_size: int = 300) -> dict[str, int]:
    """Tasks that ask the model to write a problem: a story problem, an expression, a question."""
    eod = tok.token_to_id("[EOD]")
    pools = em._pools()
    counts = {}
    for split, target in (("train", train_tokens), ("val", held_tokens), ("test", held_tokens)):
        rng = random.Random(f"compose:{seed}:{split}")
        group = [t for t in templates if _split_of_id(t["id"]) == split]
        questions = (i for batch in generate(split, 10 ** 9, seed + 31, pools) for i in batch)
        writer = Writer(root, "compose", split, labelled=False)
        while writer.count < target:
            roll = rng.random()
            item = (compose_story_item(rng.choice(group), rng) if roll < 0.5 else
                    compose_expression_item(rng) if roll < 0.8 else compose_question_item(next(questions), rng))
            writer.add([(tok.encode(item["prompt"] + item["reply"]).ids, 1)], eod)
        writer.close()
        counts[split] = writer.count
    rng = random.Random(f"compose-exam:{seed}")
    group = [t for t in templates if _split_of_id(t["id"]) == "test"]
    questions = (i for batch in generate("test", 10 ** 9, seed + 37, pools) for i in batch)
    exam = []
    for index in range(exam_size):
        item = (compose_story_item(group[index % len(group)], rng) if index % 3 == 0 else
                compose_expression_item(rng) if index % 3 == 1 else compose_question_item(next(questions), rng))
        exam.append({k: v for k, v in item.items() if k != "reply"})
    (root / "compose_exam.json").write_text(json.dumps(exam))
    return counts


# ----- word problems from public datasets -----

def _windows(text: str, size: int = 10) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def gsm8k_reply(answer: str) -> tuple[str, str]:
    """GSM8K's solution with its calculator notes removed, ending "The answer is N", and N."""
    body, final = answer.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    final = final.strip().replace(",", "")
    return f"{body}\nThe answer is {final}.", final


def build_wordmath(root: Path, tok: Tokenizer, seed: int) -> dict[str, int]:
    """Orca-Math 200k and GSM8K train for training; GSM8K test and SVAMP only as exams.

    Training problems that share a 10-word passage with any exam question are dropped.
    """
    from .data import fetch_parquet

    gsm_train = fetch_parquet("openai/gsm8k", "main/train-00000-of-00001.parquet").to_pylist()
    gsm_test = fetch_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet").to_pylist()
    orca = fetch_parquet("microsoft/orca-math-word-problems-200k", "data/train-00000-of-00001.parquet").to_pylist()
    svamp = (fetch_parquet("ChilleD/SVAMP", "data/train-00000-of-00001.parquet").to_pylist()
             + fetch_parquet("ChilleD/SVAMP", "data/test-00000-of-00001.parquet").to_pylist())
    exams = {
        "gsm8k_exam.json": [{"prompt": f"Q: {r['question']}\nA:", "answer": gsm8k_reply(r["answer"])[1]}
                            for r in gsm_test],
        "svamp_exam.json": [{"prompt": f"Q: {r['Body'].strip()} {r['Question'].strip()}\nA:",
                             "answer": str(int(float(r["Answer"]))) if float(r["Answer"]).is_integer()
                             else str(r["Answer"])} for r in svamp],
    }
    held = set()
    for rows in exams.values():
        for r in rows:
            held |= _windows(r["prompt"])
    docs, dropped = [], 0
    for r in gsm_train:
        if _windows(r["question"]) & held:
            dropped += 1
            continue
        docs.append(f"Q: {r['question']}\nA: {gsm8k_reply(r['answer'])[0]}")
    for r in orca:
        if _windows(r["question"]) & held:
            dropped += 1
            continue
        docs.append(f"Q: {r['question'].strip()}\nA: {r['answer'].strip()}")
    random.Random(seed).shuffle(docs)
    cut = max(1, len(docs) // 100)
    eod = tok.token_to_id("[EOD]")
    counts = {"dropped_for_overlap": dropped}
    for split, part in (("val", docs[:cut]), ("test", docs[cut:2 * cut]), ("train", docs[2 * cut:])):
        writer = Writer(root, "wordmath", split, labelled=False)
        for start in range(0, len(part), 2000):
            for enc in tok.encode_batch(part[start:start + 2000]):
                writer.add([(enc.ids, 1)], eod)
        writer.close()
        counts[split] = writer.count
    for name, rows in exams.items():
        (root / name).write_text(json.dumps(rows))
    return counts


def last_number(text: str) -> str | None:
    """The stated answer: after "the answer is", else the last number written."""
    found = answer_of(text)
    if found is not None:
        return found
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="folder with story, maths and qa splits and tokenizer.json")
    parser.add_argument("--templates", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explain-tokens", type=int, default=300_000_000)
    parser.add_argument("--mixed-tokens", type=int, default=600_000_000)
    parser.add_argument("--compose-tokens", type=int, default=200_000_000)
    parser.add_argument("--held-tokens", type=int, default=600_000)
    parser.add_argument("--no-wordmath", action="store_true", help="skip the public word-problem sets")
    args = parser.parse_args()
    root = Path(args.root)
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    templates, dropped = load_templates(Path(args.templates))
    print(f"{len(templates)} templates, {len(dropped)} dropped", flush=True)
    for reason in dropped:
        print("  dropped", reason)
    stats = {"templates": len(templates), "dropped": dropped,
             "explain": build_explain(root, tok, args.seed, args.explain_tokens, args.held_tokens),
             "mixed": build_mixed(root, tok, templates, args.seed, args.mixed_tokens, args.held_tokens),
             "compose": build_compose(root, tok, templates, args.seed, args.compose_tokens, args.held_tokens)}
    if not args.no_wordmath:
        stats["wordmath"] = build_wordmath(root, tok, args.seed)
    (root / "mixed_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped"}), flush=True)


if __name__ == "__main__":
    main()
