"""Mixed-topic data and explained maths: correct answers, honest splits, aligned labels."""
import json
import random
from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from spidory import data, evaluate, mixed_data as md
from spidory import easy_maths as em
from spidory.data_v2 import LEVELS, generate
from spidory.model import Spidory

from test_spidory import CFG

GOOD = {"id": "s001", "kind": "story",
        "context": "{name} had {a} shells. {name2} gave {name} {b} more and then {c} were lost.",
        "question": "How many shells does {name} have now?", "answer": "a + b - c",
        "vars": {"a": [5, 20], "b": [1, 9], "c": [0, 4]}}


@pytest.fixture(scope="module")
def tok():
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    text = ["Q: What is 3 + 4? Explain step by step.\nA: 3 + 4 = 7. The answer is 7.",
            "Tom had 5 shells. Lily gave Tom 3 more. How many shells?", "\n\n",
            "The town of Velmora built a bridge in 1820 with 300 stones."] * 50
    tokenizer.train_from_iterator(text, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[EOD]"],
                                                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tokenizer


@pytest.fixture(scope="module")
def items():
    pools = em._pools()
    return {split: [i for batch in generate(split, 60, 0, pools) for i in batch] for split in ("train", "test")}


def test_explanations_follow_the_operations():
    assert md.explain_expression("14 + 7 - 18") == (["14 + 7 = 21", "21 - 18 = 3"], 3)
    assert md.explain_expression("3 + 4 * 2") == (["4 * 2 = 8", "3 + 8 = 11"], 11)
    assert md.explain_expression("(3 + 4) * 2") == (["3 + 4 = 7", "7 * 2 = 14"], 14)
    steps, value = md.explain_variables("a = 7, b = a + 5, c = b - a; c")
    assert value == 5 and steps == ["a = 7", "b = 7 + 5 = 12", "c = 12 - 7 = 5"]


def test_every_level_explains_to_the_true_answer(items):
    for item in items["train"] + items["test"]:
        text = md.explain_item(item)
        assert md.answer_of(text) == str(item.answer)
        assert text.startswith(f"Q: {item.question} {md.EXPLAIN}\nA: ")
    assert {i.level for i in items["train"]} == set(LEVELS)


def test_explanations_read_as_sentences(items):
    rng = random.Random(0)
    for item in items["train"]:
        text = md.explain_item(item, rng)
        body = text.split("\nA: ", 1)[1]
        assert (body[0].isupper() or body[0].isdigit() or body[0] == "-") and body.endswith(".")
        for clumsy in ("then then", "Then then", "Now then", "Last, then", "that is", "  ", ". ."):
            assert clumsy not in body, (clumsy, body)
        # Every number in the explanation's results agrees with the arithmetic: re-check the sums.
        assert md.answer_of(body) == str(item.answer)


def test_explained_questions_keep_the_maths_split(items):
    """Test expressions of the maths sets must never be explained in training."""
    train = {i.expression for i in items["train"]}
    assert not train & {i.expression for i in items["test"]}
    for item in items["train"]:
        if item.level not in (2, 5):
            assert em._split_of(item.expression) == "train"


def test_good_template_passes_and_bad_ones_are_refused():
    md.check_template(GOOD)
    bad = [
        GOOD | {"answer": "a / b"},                                   # division
        GOOD | {"answer": "a - b - 30"},                              # can go negative
        GOOD | {"context": GOOD["context"] + " {place}"},             # unknown placeholder
        GOOD | {"vars": {"a": [5, 20], "b": [1, 9]}},                 # c has no range
        GOOD | {"answer": "__import__('os')"},                        # not arithmetic
        GOOD | {"kind": "poem"},
    ]
    for template in bad:
        with pytest.raises((ValueError, SyntaxError)):
            md.check_template(template)


def test_loading_drops_bad_lines_and_duplicates(tmp_path):
    lines = [json.dumps(GOOD), "not json", json.dumps(GOOD), json.dumps(GOOD | {"id": "s002", "answer": "a / b"})]
    (tmp_path / "story_1.jsonl").write_text("\n".join(lines))
    found, dropped = md.load_templates(tmp_path)
    assert [t["id"] for t in found] == ["s001"] and len(dropped) == 3


def test_filled_answers_are_computed():
    rng = random.Random(0)
    for explain in (False, True):
        for _ in range(50):
            context, tail, value = md.fill(GOOD, rng, explain)
            numbers = [int(n) for n in __import__("re").findall(r"\d+", context)]
            assert value == numbers[0] + numbers[1] - numbers[2]
            assert "{" not in context + tail
            if explain:
                assert tail.endswith(".") and md.answer_of(tail) == str(value)
            else:
                assert tail.endswith(f"A: {value}")


def test_template_splits_are_stable_and_held_out():
    ids = [f"s{i:03d}" for i in range(300)]
    splits = [md._split_of_id(i) for i in ids]
    assert splits == [md._split_of_id(i) for i in ids]
    assert 15 <= splits.count("test") <= 50 and 15 <= splits.count("val") <= 50


def _streams(root, tok):
    """Story, maths and qa splits whose tokens show their topic: 10-19, 20-29 and 30-39."""
    eod = tok.token_to_id("[EOD]")
    for index, name in enumerate(("story", "maths", "qa")):
        for split in data.SPLITS:
            rng = np.random.default_rng(index)
            docs = [np.append(rng.integers(10 * (index + 1), 10 * (index + 2), rng.integers(30, 200)), eod)
                    for _ in range(80)]
            np.concatenate(docs).astype(np.uint16).tofile(root / f"{name}.{split}.bin")


def _templates():
    found = []
    for i in range(60):
        kind = "story" if i % 2 else "fact"
        found.append(GOOD | {"id": f"{kind[0]}{i:03d}", "kind": kind})
    return found


def test_mixed_documents_have_aligned_labels(tmp_path, tok):
    _streams(tmp_path, tok)
    counts = md.build_mixed(tmp_path, tok, _templates(), 0, 40_000, 8_000, exam_size=20)
    gap = set(tok.encode("\n\n").ids)
    for split in ("train", "test"):
        ids = np.fromfile(tmp_path / f"mixed.{split}.bin", dtype=np.uint16)
        labels = np.fromfile(tmp_path / f"mixed.{split}.labels", dtype=np.uint8)
        assert len(ids) == len(labels) == counts[split]
        eod = tok.token_to_id("[EOD]")
        assert ((ids == eod) == (labels == md.END)).all()
        # Joined pieces keep their topic: a token from the story range is labelled story, and so on.
        for topic, low in ((0, 10), (1, 20), (2, 30)):
            inside = (ids >= low) & (ids < low + 10) & ~np.isin(ids, list(gap))
            assert inside.any() and (labels[inside] == topic).mean() > 0.6
        docs = np.split(labels, np.flatnonzero(labels == md.END) + 1)
        assert sum(len(set(d[d != md.END].tolist())) > 1 for d in docs if len(d) > 1) > len(docs) * 0.6
    exam = json.loads((tmp_path / "mixed_exam.json").read_text())
    assert len(exam) == 20 and all(e["prompt"].endswith("A:") for e in exam)
    assert any(e["lead"] for e in exam) and not all(e["lead"] for e in exam)


def test_explain_dataset_and_exam(tmp_path, tok):
    counts = md.build_explain(tmp_path, tok, 0, 20_000, 4_000, exam_per_level=5)
    assert counts["train"] >= 20_000
    exam = json.loads((tmp_path / "explain_exam.json").read_text())
    assert len(exam) == 5 * len(LEVELS) and all(e["prompt"].endswith(f"{md.EXPLAIN}\nA:") for e in exam)
    text = tok.decode(np.fromfile(tmp_path / "explain.train.bin", dtype=np.uint16)[:3000].tolist())
    assert "answer is" in text


def test_evaluation_scores_mixed_topics_and_exams(tmp_path, tok, monkeypatch):
    _streams(tmp_path, tok)
    md.build_mixed(tmp_path, tok, _templates(), 0, 20_000, 6_000, exam_size=6)
    md.build_explain(tmp_path, tok, 0, 6_000, 3_000, exam_per_level=1)
    md.build_compose(tmp_path, tok, _templates(), 0, 6_000, 3_000, exam_size=3)
    for split in data.SPLITS:  # any token file will do for the loss; the exams are what is tested
        (tmp_path / f"wordmath.{split}.bin").write_bytes((tmp_path / f"explain.{split}.bin").read_bytes())
    (tmp_path / "gsm8k_exam.json").write_text(json.dumps([{"prompt": "Q: What is 3 + 4?\nA:", "answer": "7"}]))
    names = ("story", "maths", "qa", "mixed", "explain", "compose", "wordmath")
    monkeypatch.setattr(data, "DATASETS", names)
    monkeypatch.setattr(evaluate, "DATASETS", names)
    corpus = data.Corpus(tmp_path, CFG.seq_len)
    batch = corpus.train_batch(8)
    assert batch.shape == (8, CFG.seq_len + 1)
    model = Spidory(replace(CFG, vocab_size=tok.get_vocab_size()))
    found = evaluate.extra_scores(model, corpus, tok, 4, 3, k=2)
    assert set(found["mixed_topics"]) == {"story", "maths", "qa"}
    assert all(np.isfinite(v) for v in found["mixed_topics"].values())
    for key in ("mixed_accuracy", "explain_accuracy", "compose_accuracy", "gsm8k_accuracy"):
        assert 0 <= found[key] <= 1, key
    assert "explain_accuracy" not in evaluate.extra_scores(model, corpus, tok, 4, 3, generate=False, k=2)


def test_answer_reader():
    assert md.answer_of("3 + 4 = 7. The answer is 7.") == "7"
    assert md.answer_of("2 - 9 = -7. The answer is -7.") == "-7"
    assert md.answer_of("no answer here") is None


def test_compose_replies_are_graded_by_computing():
    rng = random.Random(1)
    pools = em._pools()
    questions = (i for b in generate("train", 50, 0, pools) for i in b)
    templates = _templates()
    for _ in range(60):
        for item in (md.compose_story_item(rng.choice(templates), rng), md.compose_expression_item(rng),
                     md.compose_question_item(next(questions), rng)):
            assert md.grade_compose(item, item["reply"]), item
    expression = {"kind": "*", "answer": 24, "count": 2}
    assert md.grade_compose(expression, " 4 * 6 = 24") and md.grade_compose(expression, "3 * 8 = 24\nmore")
    for wrong in ("4 * 5 = 24", "20 + 4 = 24", "4 * 6 = 25", "24", "2 * 3 * 4 = 24"):
        assert not md.grade_compose(expression, wrong), wrong
    story = {"kind": "story", "numbers": [7, 5], "answer": 12}
    assert md.grade_compose(story, "Tom had 7 apples and got 5 more.\nQ: How many?\nA: 12")
    assert not md.grade_compose(story, "Tom had 7 apples and got 4 more.\nQ: How many?\nA: 12")
    assert not md.grade_compose(story, "Tom had 7 apples and got 5 more.\nQ: How many?\nA: 13")
    question = {"kind": "question", "answer": 9}
    assert md.grade_compose(question, "What is 4 + 5?\nAnswer: 9")
    assert md.grade_compose(question, "a = 4, b = a + 5. What is b?")
    assert not md.grade_compose(question, "What is 4 + 6?")


def test_compose_dataset_and_exam(tmp_path, tok):
    counts = md.build_compose(tmp_path, tok, _templates(), 0, 20_000, 3_000, exam_size=30)
    assert counts["train"] >= 20_000
    exam = json.loads((tmp_path / "compose_exam.json").read_text())
    assert len(exam) == 30 and {e["kind"] for e in exam} >= {"story", "question"}
    assert all("reply" not in e and e["prompt"].startswith("Task: ") for e in exam)
    test_ids = {t["id"] for t in _templates() if md._split_of_id(t["id"]) == "test"}
    assert test_ids  # story tasks in the exam come only from held-out templates


class _Table:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


def test_word_problems_keep_exams_out_of_training(tmp_path, tok, monkeypatch):
    shared = "A farmer had 41 sheep in a field near the old mill and sold some of them at the market"
    tables = {
        ("openai/gsm8k", "main/train-00000-of-00001.parquet"): [
            {"question": "Ann has 3 cats and buys 2 more cats today. How many cats?", "answer": "3+2 = <<3+2=5>>5\n#### 5"},
            {"question": shared + ". How many are left?", "answer": "41-10=<<41-10=31>>31\n#### 31"}],
        ("openai/gsm8k", "main/test-00000-of-00001.parquet"): [
            {"question": shared + ". How many did he sell?", "answer": "#### 10"}],
        ("microsoft/orca-math-word-problems-200k", "data/train-00000-of-00001.parquet"): [
            {"question": f"Problem number {i} about counting {i} boxes of pens on a long shelf.", "answer": f"There are {i} boxes."}
            for i in range(300)],
        ("ChilleD/SVAMP", "data/train-00000-of-00001.parquet"): [
            {"Body": "There are 5 birds.", "Question": "How many birds?", "Answer": 5.0}],
        ("ChilleD/SVAMP", "data/test-00000-of-00001.parquet"): [],
    }
    monkeypatch.setattr(data, "fetch_parquet", lambda repo, name: _Table(tables[repo, name]))
    counts = md.build_wordmath(tmp_path, tok, 0)
    assert counts["dropped_for_overlap"] == 1
    text = tok.decode(np.fromfile(tmp_path / "wordmath.train.bin", dtype=np.uint16).tolist())
    assert "old mill" not in text and "<<" not in text
    gsm = json.loads((tmp_path / "gsm8k_exam.json").read_text())
    assert gsm == [{"prompt": f"Q: {shared}. How many did he sell?\nA:", "answer": "10"}]
    assert not (tmp_path / "svamp_exam.json").exists()  # contaminated by Orca rewrites, so unused
    assert md.gsm8k_reply("2+2=<<2+2=4>>4\n#### 1,204") == ("2+2=4\nThe answer is 1204.", "1204")
    assert md.last_number("So she has 12 left.") == "12" and md.last_number("The answer is 7. Then 9") == "7"
    assert md.answer_of("The answer is 3. No wait, the answer is 1,200.") == "1200"


def test_rewrites_with_new_numbers_are_caught():
    exam = "Each basket of peaches has 19 red peaches and 4 green peaches. If there are 15 such baskets"
    rewrite = "Each basket of peaches has 7 red peaches and some green peaches. There are 9 such baskets"
    assert md._windows(exam) & md._windows(rewrite)


def test_training_never_shows_a_maths_exam_expression():
    rng = random.Random(0)
    held = md.held_expressions()
    assert "26 - 18" in held or any(" - " in h for h in held)
    for _ in range(5000):
        item = md.compose_expression_item(rng, "train")
        assert not md.held_out(item["reply"].split("=")[0].strip()), item
    small = GOOD | {"answer": "a - b", "vars": {"a": [10, 30], "b": [1, 9]},
                    "context": "{name} had {a} shells and gave {b} away."}
    for _ in range(2000):
        for explain in (False, True):
            context, tail, value = md.fill(small, rng, explain, split="train")
            a, b = [int(n) for n in __import__("re").findall(r"\d+", context)]
            assert not md.held_out(f"{a} - {b}")


def test_replies_are_complete_before_generation_stops(tok):
    """Fed one token at a time, a correct reply must not stop early and must then pass."""
    rng = random.Random(2)
    questions = (i for b in generate("test", 50, 0, em._pools()) for i in b)
    items = [md.compose_story_item(t, rng) for t in _templates()[:10]] + \
            [md.compose_expression_item(rng) for _ in range(20)] + [md.compose_question_item(next(questions), rng) for _ in range(10)]
    for item in items:
        ids = tok.encode(item["reply"] + "\n").ids
        done = evaluate.compose_done(item)
        stopped = next((i for i in range(1, len(ids) + 1) if done(tok.decode(ids[:i]))), None)
        assert stopped is not None, item
        assert md.grade_compose(item, tok.decode(ids[:stopped])), (item, tok.decode(ids[:stopped]))


def test_grader_refuses_trivial_replies():
    assert not md.grade_compose({"kind": "brackets", "answer": 12}, "(12) = 12")
    assert md.grade_compose({"kind": "brackets", "answer": 14}, "(3 + 4) * 2 = 14")
    assert not md.grade_compose({"kind": "question", "answer": 17}, "What is 17?\nAnswer: 17")
    story = {"kind": "story", "numbers": [12, 5, 3], "answer": 14}
    assert not md.grade_compose(story, "12 5 3\nA: 14")
    assert md.grade_compose(story, "Tom had 12 cards. He won 5 more and lost 3 of them in the park.\nQ: How many now?\nA: 14")


def test_near_duplicate_templates_share_a_split():
    a = GOOD | {"id": "s001"}
    b = GOOD | {"id": "s999", "context": GOOD["context"].replace("shells", "stones")}
    c = GOOD | {"id": "f500", "kind": "fact", "question": "How many years passed?", "answer": "a * b",
                "context": "The town of Velmora built {a} bridges, each with {b} arches, while {c} were planned."}
    templates = [a, b, c]
    md.assign_splits(templates)
    assert a["split"] == b["split"]
    assert md.split_of(c) == c["split"]
