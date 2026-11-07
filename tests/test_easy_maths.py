import json
import re

import numpy as np
import pytest

from spidory import data, easy_maths
from spidory.easy_maths import LEVELS, SPLITS, generate_split, render

SEED = 3
PER_LEVEL = 400
WORDING = re.compile(r"^(?:What is |Calculate |Compute )?(.+?)(?:\?|\.| =)$")


@pytest.fixture(scope="module")
def items():
    return {split: generate_split(split, PER_LEVEL, SEED) for split in SPLITS}


def parse(item):
    """Read the expression and answer back from the text alone."""
    match = re.fullmatch(r"Q: (.+)\nA: (-?\d+)", render(item))
    assert match, render(item)
    return WORDING.fullmatch(match[1])[1], int(match[2])


def evaluate(expression):
    assert re.fullmatch(r"[\d +\-*()]+", expression)
    return eval(expression)  # Python's own precedence, unrelated to the generator's arithmetic.


def test_answers_are_correct_and_text_matches_format(items):
    for split in SPLITS:
        for item in items[split]:
            expression, answer = parse(item)
            assert expression == item.expression
            assert answer == evaluate(expression)


def test_splits_do_not_overlap(items):
    seen = {split: {parse(i)[0] for i in items[split]} for split in SPLITS}
    assert not seen["train"] & seen["val"]
    assert not seen["train"] & seen["test"]
    assert not seen["val"] & seen["test"]


def test_full_expression_pools_are_disjoint_and_about_five_percent():
    by_split = {split: set() for split in SPLITS}
    for (_, _, split), entries in easy_maths._pools().items():
        assert entries
        assert not by_split[split] & {text for text, _ in entries}
        by_split[split] |= {text for text, _ in entries}
    assert not by_split["train"] & by_split["val"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["val"] & by_split["test"]
    total = sum(len(found) for found in by_split.values())
    assert abs(len(by_split["test"]) / total - 0.05) < 0.002
    assert abs(len(by_split["val"]) / total - 0.05) < 0.002


def test_every_level_is_in_every_split_in_equal_numbers(items):
    for split in SPLITS:
        counts = [sum(i.level == level for i in items[split]) for level in LEVELS]
        assert counts == [PER_LEVEL] * len(LEVELS)


def test_levels_have_the_requested_shape(items):
    for item in items["train"] + items["val"]:
        numbers = [int(n) for n in re.findall(r"\d+", item.expression)]
        operators = re.findall(r" ([+\-*]) ", item.expression)
        if item.level == 1:
            assert len(numbers) == 2 and max(numbers) <= 9
        elif item.level == 2:
            assert len(numbers) == 2 and 10 <= max(numbers) <= 99
            assert operators != ["*"] or min(numbers) <= 9
        elif item.level == 3:
            assert len(numbers) == 3 and max(numbers) <= 20 and "*" not in operators
        elif item.level == 4:
            assert len(numbers) in (4, 5) and max(numbers) <= 20 and "*" not in operators
        else:
            assert len(numbers) == 3 and max(numbers) <= 9 and "*" in operators


def test_precedence_examples():
    pool = {text: value for (level, _, _), entries in easy_maths._pools().items() if level == 5
            for text, value in entries}
    assert pool["(3 + 4) * 2"] == 14
    assert pool["2 + 3 * 4"] == 14
    assert pool["2 * 3 - 4"] == 2
    assert pool["2 * (3 - 4)"] == -2


def test_generation_is_deterministic_for_one_seed(items):
    assert generate_split("val", PER_LEVEL, SEED) == items["val"]
    assert generate_split("val", PER_LEVEL, SEED + 1) != items["val"]


def test_exam_is_balanced_and_comes_from_test(items):
    exam = easy_maths.build_exam(items["test"], 60, SEED)
    assert len(exam) == 300
    assert all(set(row) == {"prompt", "answer", "level"} for row in exam)
    assert [sum(r["level"] == level for r in exam) for level in LEVELS] == [60] * 5
    assert [sum(r["level"] == level for r in exam[:100]) for level in LEVELS] == [20] * 5
    assert {r["prompt"] for r in exam} <= {easy_maths.prompt(i) for i in items["test"]}
    assert len({r["level"] for r in exam[:5]}) == 5


def test_build_writes_outputs_keeps_other_stats_and_backs_up_once(tmp_path):
    sample = [render(i) for i in generate_split("train", 200, SEED)]
    tok = data.train_tokenizer(sample, 300)
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "stats.json").write_text(json.dumps({"story": {"train": 1}, "maths": {"train": 2}}))
    (tmp_path / "maths.train.bin").write_bytes(b"old")

    stats = easy_maths.build(tmp_path, SEED, 6000, 3000, 10)
    easy_maths.build(tmp_path, SEED, 6000, 3000, 10)

    assert (tmp_path / "old_maths" / "maths.train.bin").read_bytes() == b"old"
    saved = json.loads((tmp_path / "stats.json").read_text())
    assert saved["story"] == {"train": 1} and saved["maths"] == stats
    backup = json.loads((tmp_path / "old_maths" / "stats.json").read_text())
    assert backup["maths"] == {"train": 2}
    for split in SPLITS:
        stream = np.fromfile(tmp_path / f"maths.{split}.bin", dtype=np.uint16)
        assert stream.size == stats[split]
        assert stream[-1] == tok.token_to_id("[EOD]")
    assert len(json.loads((tmp_path / "exam.json").read_text())) == 50
