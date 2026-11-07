import random
import re

import pytest

from spidory import data_v2, easy_maths


@pytest.fixture(scope="module")
def pools():
    return easy_maths._pools()


@pytest.fixture(scope="module")
def items(pools):
    return {split: [i for batch in data_v2.generate(split, 300, 0, pools) for i in batch]
            for split in ("train", "val", "test")}


def test_every_level_in_every_split(items):
    for found in items.values():
        assert {i.level for i in found} == set(data_v2.LEVELS)


def test_chain_and_nested_answers_match_independent_eval(items):
    for item in items["train"]:
        if item.level in (2, 5, 6, 7, 9):
            assert eval(item.expression) == item.answer


def test_chains_have_the_right_length(items):
    for item in items["train"]:
        numbers = len(re.findall(r"\d+", item.expression))
        if item.level == 6:
            assert 6 <= numbers <= 8
        if item.level == 7:
            assert 9 <= numbers <= 12


def test_variable_answers_match_independent_run(items):
    checked = 0
    for item in items["train"]:
        if item.level == 8:
            code, name = item.expression.split("; ")
            scope = {}
            exec(code.replace(", ", "\n"), {}, scope)
            assert scope[name] == item.answer
            assert item.question.endswith(f"What is {name}?")
            checked += 1
    assert checked == 300


def test_nesting_is_real(items):
    assert all(item.expression.count("(") >= 2 for item in items["train"] if item.level == 9)
    assert all(abs(item.answer) <= 500 for item in items["train"] if item.level == 9)


def test_splits_do_not_overlap(items):
    seen = {split: {i.expression for i in found} for split, found in items.items()}
    assert not seen["train"] & seen["val"]
    assert not seen["train"] & seen["test"]
    assert not seen["val"] & seen["test"]


def test_generation_repeats_for_one_seed(pools):
    first = [i for batch in data_v2.generate("val", 50, 3, pools) for i in batch]
    again = [i for batch in data_v2.generate("val", 50, 3, pools) for i in batch]
    assert first == again


def test_format(items):
    for item in items["val"][:200]:
        assert re.fullmatch(r"Q: .+\nA: -?\d+", easy_maths.render(item))


def test_levels_are_equal(items):
    counts = {level: sum(i.level == level for i in items["train"]) for level in data_v2.LEVELS}
    assert set(counts.values()) == {300}
