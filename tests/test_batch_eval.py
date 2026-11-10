"""Exams generated in batches give the same replies and verdicts as one item at a time, and
models without loops train and score at one run only."""
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from spidory import evaluate
from spidory.flops import tflops_by_k
from spidory.model import Spidory
from spidory.train import sample_ks

from test_spidory import CFG, TCFG, wake_experts

ALPHABET = "0123456789\n. abcdefghij"


class Tok:
    """Every id decodes to one character, so replies hit line breaks, full stops and numbers."""

    def encode(self, text):
        return type("E", (), {"ids": [ord(c) % 60 + 2 for c in text]})()

    def decode(self, ids):
        return "".join(ALPHABET[i % len(ALPHABET)] for i in ids)

    def token_to_id(self, name):
        return 1


def old_answer(kind, model, tok, item, longest=None, **choice):
    """The evaluation code before batching (commit 65867bf), one item at a time."""
    from spidory.mixed_data import answer_of, grade_compose, last_number

    def generate(prompt, top, done):
        eod = tok.token_to_id("[EOD]")
        ids = tok.encode(prompt).ids
        start, text = len(ids), ""
        for _ in range(top):
            window = torch.tensor([ids[-model.cfg.seq_len:]])
            next_id = int(model.infer(window, **choice)[0][0, -1].argmax())
            if next_id == eod:
                break
            ids.append(next_id)
            text = tok.decode(ids[start:])
            if done is not None and done(text):
                break
        return text

    if kind == "compose":
        text = generate(item["prompt"], longest or 180, evaluate.compose_done(item))
        return text, grade_compose(item, text)
    if kind == "word":
        text = generate(item["prompt"], longest or 300, evaluate._word_done).split("\nQ:")[0]
        found = last_number(text)
        try:
            return text, found is not None and float(found) == float(item["answer"])
        except ValueError:
            return text, False
    eod = tok.token_to_id("[EOD]")
    prompt = tok.encode(item["prompt"]).ids
    ids, text, found = list(prompt), "", None
    for _ in range(longest or (16 if kind == "maths" else 220)):
        window = torch.tensor([ids[-model.cfg.seq_len:]])
        next_id = int(model.infer(window, **choice)[0][0, -1].argmax())
        if next_id == eod:
            break
        ids.append(next_id)
        text = tok.decode(ids[len(prompt):])
        if kind == "maths":
            if "\n" in text:
                break
        elif "\n" in text or text.rstrip().endswith("."):
            found = answer_of(text)
            if found is not None or "\n" in text:
                break
    if kind == "maths":
        return text, text.split("\n")[0].strip() == item["answer"]
    return text, found == item["answer"]


def items(count=14):
    # Several prompts per length, some shorter than the routing prefix, some longer than a window.
    lengths = [3, 3, 5, 9, 9, 9, 20, 30, 47, 47, 52, 60, 4, 5][:count]
    found = []
    for n, length in enumerate(lengths):
        prompt = "".join("qwertyuiopasdfghjklzx"[(n * 7 + j) % 21] for j in range(length))
        found.append({"prompt": prompt, "answer": str(n % 10), "kind": ("story", "question", "expression")[n % 3],
                      "numbers": [n, 2], "level": 2 + n % 3})
    return found


MODELS = {
    "plain": replace(CFG, use_experts=False),
    "per_sequence": CFG,
    "per_loop": replace(CFG, route_per_loop=True),
    "per_token": replace(CFG, route_per_token=True, run_mode="joint"),
    "per_token_signal": replace(CFG, route_per_token=True, run_mode="joint", run_signal=True),
    "between": replace(CFG, between_layers=1, run_signal=True),
}


def built(cfg):
    torch.manual_seed(3)
    model = Spidory(cfg)
    if cfg.use_experts:
        wake_experts(model)
        model.run_head_ready.fill_(True)
    return model.eval()


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("kind", ["maths", "explain", "compose", "word"])
def test_batched_replies_match_one_at_a_time(name, kind):
    model = built(MODELS[name])
    choices = [{"k": 1}, {"k": 3}] + ([{"learned": True}, {"mean_runs": 2.5}] if MODELS[name].use_experts else [])
    longest = {"maths": 16, "explain": 30, "compose": 30, "word": 30}[kind]
    for choice in choices:
        old = [old_answer(kind, model, Tok(), item, longest=longest, **choice) for item in items()]
        new = evaluate.answer_items(kind, model, Tok(), items(), longest=longest, batch=4, **choice)
        assert new == old, (name, kind, choice)


def test_batch_rows_compute_what_a_batch_of_one_computes():
    model = built(MODELS["per_sequence"])
    rows = torch.randint(2, 60, (5, 7))
    together = model.infer(rows, k=2)[0]
    for i in range(5):
        assert torch.allclose(together[i], model.infer(rows[i:i + 1], k=2)[0][0], atol=1e-5)


def test_single_item_paths_agree():
    model = built(MODELS["per_token"])
    item = items()[3]
    assert evaluate.answer_item("maths", model, Tok(), item, k=2) == old_answer("maths", model, Tok(), item, k=2)
    # An expression item stops at its first line, as the old generate_text did with that check.
    expression = item | {"kind": "expression"}
    assert evaluate.answer_item("compose", model, Tok(), expression, longest=12, k=2) == \
        old_answer("compose", model, Tok(), expression, longest=12, k=2)


def test_unknown_exam_kind_is_refused_before_generating():
    with pytest.raises(ValueError):
        evaluate.answer_items("nope", built(MODELS["plain"]), Tok(), items(2))


def test_a_model_without_loops_always_runs_once():
    cfg = replace(CFG, use_experts=False, max_runs=1, target_mean=1.0)
    model = Spidory(cfg)
    experts = torch.zeros(4, dtype=torch.long)
    for step in (0, TCFG.stage1_steps, TCFG.stage1_steps + TCFG.stage2_steps, 10_000):
        for _ in range(20):
            assert sample_ks(step, replace(TCFG, cycle_steps=4), model, experts, tokens=CFG.seq_len).max() == 1
    assert torch.isfinite(model.infer(torch.randint(2, 60, (2, 9)), k=1)[0]).all()


def test_the_deep_model_costs_what_plain_costs_at_three_runs():
    plain = replace(CFG, use_experts=False, n_core=10)
    deep = replace(plain, n_core=30, max_runs=1)
    assert tflops_by_k(deep)[1] == pytest.approx(tflops_by_k(plain)[3])


def test_train_cli_builds_the_deep_model(tmp_path, monkeypatch):
    from spidory import train

    seen = {}
    monkeypatch.setattr(train, "Spidory", lambda cfg: seen.setdefault("cfg", cfg) and (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(train, "Corpus", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train", "--name", "x", "--plain", "--max-runs", "1", "--n-core", "30",
                                     "--device", "cpu"])
    with pytest.raises(SystemExit):
        train.main()
    cfg = seen["cfg"]
    assert (cfg.n_core, cfg.max_runs, cfg.target_mean, cfg.use_experts) == (30, 1, 1.0, False)


def test_score_table_handles_models_without_loops():
    row = {"run": "noloop_s0", "params": 10, "by_mean": {},
           "by_k": {"1": {"mean_loss": 1.5, "maths_accuracy": 0.5}}}
    line = evaluate.table([row]).splitlines()[-1]
    assert "1.500" in line and line.count("—") == 8


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _jobs(tmp_path, *jobs, exists_ok=False):
    """Runs run_jobs.sh in a copy of the scripts folder with a fake Python that logs its calls."""
    (tmp_path / "scripts").mkdir()
    for name in ("run_jobs.sh", "run_mixed.sh"):
        (tmp_path / "scripts" / name).write_text((SCRIPTS / name).read_text())
        (tmp_path / "scripts" / name).chmod(0o755)
    (tmp_path / "runs").mkdir(exist_ok=exists_ok)
    fake = tmp_path / "py"
    fake.write_text('#!/usr/bin/env bash\necho "$@" >> calls.log\n')
    fake.chmod(0o755)
    env = os.environ | {"PY": str(fake)}
    result = subprocess.run(["bash", "scripts/run_jobs.sh", *jobs], cwd=tmp_path, env=env, capture_output=True)
    calls = (tmp_path / "calls.log").read_text().splitlines() if (tmp_path / "calls.log").exists() else []
    return result.returncode, calls, tmp_path / "runs"


def test_jobs_run_in_order_and_mark_the_end_once(tmp_path):
    code, calls, runs = _jobs(tmp_path, "noloop@2", "token@1")
    assert code == 0 and (runs / "DONE").exists() and not (runs / "STOPPED").exists()
    names = [c.split("--name ")[1].split()[0] for c in calls if "--name" in c]
    assert names == ["noloop_warm_s2", "noloop_s2", "token_warm_s1", "token_s1"]
    assert any("--max-runs 1" in c for c in calls if "noloop_warm_s2" in c)
    assert any("--out runs/results_noloop_s2.json" in c for c in calls)
    assert any("--out runs/results_token_s1.json" in c for c in calls)


def test_a_failed_job_is_recorded_and_the_next_job_still_runs(tmp_path):
    code, calls, runs = _jobs(tmp_path, "score@plain_s0", "noloop@1")
    assert code == 0 and not (runs / "DONE").exists()
    assert "score@plain_s0" in (runs / "STOPPED").read_text() and "score@plain_s0" in (runs / "FAILED_JOBS").read_text()
    assert [c.split("--name ")[1].split()[0] for c in calls if "--name" in c] == ["noloop_warm_s1", "noloop_s1"]


def test_a_training_failure_never_writes_stopped_before_the_end(tmp_path):
    (tmp_path / "scripts").mkdir()
    for name in ("run_jobs.sh", "run_mixed.sh"):
        (tmp_path / "scripts" / name).write_text((SCRIPTS / name).read_text())
    (tmp_path / "runs").mkdir()
    fake = tmp_path / "py"
    # Fails the first training and checks that no STOPPED file exists while later jobs run.
    fake.write_text('#!/usr/bin/env bash\necho "$@" >> calls.log\n'
                    '[ -f runs/STOPPED ] && echo seen >> stopped_seen\n'
                    '[[ "$*" == *noloop_warm_s0* ]] && exit 1\nexit 0\n')
    fake.chmod(0o755)
    subprocess.run(["bash", "scripts/run_jobs.sh", "noloop@0", "plain@2"], cwd=tmp_path,
                   env=os.environ | {"PY": str(fake)}, capture_output=True)
    assert not (tmp_path / "stopped_seen").exists()
    assert "noloop@0" in (tmp_path / "runs" / "FAILED_JOBS").read_text()
    assert "plain_s2" in (tmp_path / "calls.log").read_text()


@pytest.mark.parametrize("job", ["plian@2", "@2", "plain@", "plain@x", "token_sigal_once@1", "nothing"])
def test_bad_jobs_are_refused_before_anything_runs(tmp_path, job):
    code, calls, runs = _jobs(tmp_path, "noloop@0", job)
    assert code == 1 and calls == [] and (runs / "STOPPED").exists() and not (runs / "DONE").exists()


def test_no_job_starts_after_the_time_limit(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "LIMIT_REACHED").touch()
    code, calls, runs = _jobs(tmp_path, "noloop@0", exists_ok=True)
    assert code == 1 and calls == [] and not (runs / "DONE").exists()


def test_run_mixed_fails_when_no_version_matches(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_mixed.sh").write_text((SCRIPTS / "run_mixed.sh").read_text())
    (tmp_path / "runs").mkdir()
    result = subprocess.run(["bash", "scripts/run_mixed.sh", "0", "nope"], cwd=tmp_path,
                            env=os.environ | {"PY": "true"}, capture_output=True)
    assert result.returncode == 1 and "nope" in (tmp_path / "runs" / "STOPPED").read_text()
