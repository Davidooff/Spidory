from dataclasses import replace

import pytest
import torch

from spidory.model import Spidory
from spidory.train import make_optimizer, save
from spidory.ui import server

from test_spidory import CFG, TCFG, corpus, wake_experts  # noqa: F401


class FakeTok:
    def encode(self, text):
        return type("E", (), {"ids": [ord(c) % 60 + 2 for c in text][:20]})()

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)

    def token_to_id(self, name):
        return 1


@pytest.fixture
def registry(tmp_path, corpus):  # noqa: F811
    entries = []
    for name, cfg in (("moe", CFG), ("plain", replace(CFG, use_experts=False))):
        model = Spidory(cfg)
        if cfg.use_experts:
            wake_experts(model)
            model.run_head_ready.fill_(True)
        save(tmp_path / name / "last.pt", model, make_optimizer(model, TCFG), 1, TCFG, corpus)
        entries.append({"id": name, "name": name, "note": "", "path": f"{name}/last.pt"})
    return server.Registry(tmp_path, entries)


def test_models_are_described(registry):
    found = {m["id"]: m for m in registry.describe()}
    assert found["moe"]["neuron"] and not found["plain"]["neuron"]
    assert found["plain"]["experts"] == 0 and len(found["moe"]["mflops"]) == server.EXTRA_RUNS


def test_settings_map_to_inference_choices(registry):
    moe, plain = registry.get("moe"), registry.get("plain")
    assert server.setting_args({"kind": "neuron"}, moe) == {"learned": True}
    assert server.setting_args({"kind": "frozen", "value": 3}, plain) == {"k": 3}
    assert server.setting_args({"kind": "mean", "value": 2.5}, moe) == {"mean_runs": 2.5}
    for bad, model in (({"kind": "neuron"}, plain), ({"kind": "mean", "value": 2}, plain),
                       ({"kind": "frozen", "value": 9}, moe)):
        with pytest.raises(ValueError):
            server.setting_args(bad, model)


def test_columns_run_side_by_side_with_frozen_runs(registry):
    events = []
    request = {"prompt": "hello", "max_tokens": 5, "temperature": 0,
               "columns": [{"model": "moe", "setting": {"kind": "frozen", "value": k}} for k in (1, 3)]
               + [{"model": "plain", "setting": {"kind": "neuron"}}]}
    server.run_columns(registry, FakeTok(), request, events.append)
    tokens = [e for e in events if "text" in e]
    assert {e["runs"] for e in tokens if e["col"] == 0} == {1}
    assert {e["runs"] for e in tokens if e["col"] == 1} == {3}
    assert any(e.get("error") for e in events if e.get("col") == 2)
    order = [e["col"] for e in tokens]
    assert order[:2] == [0, 1]  # one token each in turn, not one column after another
    assert events[-1] == {"finished": True}
    assert all(e["neuron"] is not None for e in tokens)


def test_neuron_setting_reports_its_own_choice(registry):
    events = []
    request = {"prompt": "abc", "max_tokens": 3, "columns": [{"model": "moe", "setting": {"kind": "neuron"}}]}
    server.run_columns(registry, FakeTok(), request, events.append)
    for e in (e for e in events if "text" in e):
        assert e["runs"] == max(range(len(e["neuron"])), key=lambda i: e["neuron"][i]) + 1


def test_expert_lock_runs_that_expert_in_every_run(registry):
    moe = registry.get("moe")
    assert server.expert_args({"kind": "router"}, moe) == {}
    assert server.expert_args({"kind": "fixed", "value": 2}, moe) == {"expert": 2}
    with pytest.raises(ValueError):
        server.expert_args({"kind": "fixed", "value": CFG.n_experts}, moe)
    with pytest.raises(ValueError):
        server.expert_args({"kind": "fixed", "value": 0}, registry.get("plain"))
    events = []
    request = {"prompt": "hello", "max_tokens": 4,
               "columns": [{"model": "moe", "setting": {"kind": "frozen", "value": 3},
                            "experts": {"kind": "fixed", "value": e}} for e in range(CFG.n_experts)]}
    server.run_columns(registry, FakeTok(), request, events.append)
    for e in (e for e in events if "text" in e):
        assert e["expert"] == e["col"] and e["path"] == [e["col"]] * 3


def test_locked_expert_differs_from_router_output():
    model = Spidory(CFG).eval()
    wake_experts(model)
    tokens = torch.randint(0, CFG.vocab_size, (1, 40))
    chosen = int(model.infer(tokens, k=2)[1][0])
    other = (chosen + 1) % CFG.n_experts
    same, _, _ = model.infer(tokens, k=2, expert=chosen)
    assert torch.allclose(same, model.infer(tokens, k=2)[0])
    assert not torch.allclose(model.infer(tokens, k=2, expert=other)[0], same)


def test_per_loop_routing_stays_locked():
    model = Spidory(replace(CFG, route_per_loop=True)).eval()
    wake_experts(model)
    model._route_record = []
    model.infer(torch.randint(0, CFG.vocab_size, (1, 40)), k=4, expert=1)
    assert [int(r[0]) for r in model._route_record] == [1, 1, 1, 1]
    assert not model._route_locked


def test_runs_past_training_maximum(registry):
    for cfg in (CFG, replace(CFG, between_layers=1, between_mode="per_gap")):
        model = Spidory(cfg).eval()
        logits, _, ks = model.infer(torch.randint(0, cfg.vocab_size, (1, 20)), k=cfg.max_runs + 2)
        assert int(ks[0]) == cfg.max_runs + 2 and torch.isfinite(logits).all()
    events = []
    request = {"prompt": "hi", "max_tokens": 2,
               "columns": [{"model": "moe", "setting": {"kind": "frozen", "value": server.EXTRA_RUNS}}]}
    server.run_columns(registry, FakeTok(), request, events.append)
    assert all(e["runs"] == server.EXTRA_RUNS and e["mflops"] > 0 for e in events if "text" in e)


def test_discover_finds_runs_in_known_folders(tmp_path, corpus):  # noqa: F811
    model = Spidory(CFG)
    for where in ("remote_v3_s0/plain_s0", "remote_between/neuron_between_shared_s0", "elsewhere/x"):
        save(tmp_path / "runs" / where / "last.pt", model, make_optimizer(model, TCFG), 1, TCFG, corpus)
    found = server.Registry(tmp_path).describe()
    assert [m["id"] for m in found] == ["remote_v3_s0/plain_s0", "remote_between/neuron_between_shared_s0"]
    assert found[0]["group"] == "v3, seed 0" and found[0]["params"] == model.num_params()
    assert len(found[0]["mflops"]) == server.EXTRA_RUNS


def test_per_token_cycle_gives_each_token_its_own_runs(registry):
    moe = registry.get("moe")
    assert server.setting_args({"kind": "cycle", "value": 4}, moe) == {"cycle": 4}
    with pytest.raises(ValueError):
        server.setting_args({"kind": "cycle", "value": 9}, moe)
    events = []
    request = {"prompt": "hello", "max_tokens": 6,
               "columns": [{"model": "moe", "setting": {"kind": "cycle", "value": 4}},
                           {"model": "moe", "setting": {"kind": "cycle", "value": 3}, "experts": {"kind": "fixed", "value": 1}},
                           {"model": "plain", "setting": {"kind": "cycle", "value": 2}}]}
    server.run_columns(registry, FakeTok(), request, events.append)
    tokens = [e for e in events if "text" in e]
    prompt = len(FakeTok().encode("hello").ids)
    for col, top in ((0, 4), (1, 3), (2, 2)):
        runs = [e["runs"] for e in tokens if e["col"] == col]
        assert runs == [(prompt - 1 + i) % top + 1 for i in range(len(runs))]
    assert all(e["expert"] == 1 and e["path"] == [1] * e["runs"] for e in tokens if e["col"] == 1)


# ----- playground v4: new model flags, traces and exams -----

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from spidory.ui import exams as ui_exams  # noqa: E402

from test_mixed_data import _streams, tok  # noqa: E402,F401

TOKEN_CFG = replace(CFG, route_per_token=True, route_per_loop=True, run_signal=True)


@pytest.fixture
def v4_registry(tmp_path, corpus):  # noqa: F811
    entries = []
    for name, cfg in (("token", TOKEN_CFG), ("moe", CFG), ("plain", replace(CFG, use_experts=False))):
        model = Spidory(cfg)
        if cfg.use_experts:
            wake_experts(model)
            model.run_head_ready.fill_(True)
        save(tmp_path / name / "last.pt", model, make_optimizer(model, TCFG), 1, TCFG, corpus)
        entries.append({"id": name, "name": name, "note": "", "path": f"{name}/last.pt"})
    return server.Registry(tmp_path, entries)


def test_v4_models_are_found_first_with_their_features(tmp_path, corpus):  # noqa: F811
    runs = {"remote_v3_s0/plain_s0": CFG,
            "remote_mixed_token/token_signal_s0": TOKEN_CFG,
            "remote_mixed_plain/plain_signal_nonorm_s0": replace(CFG, use_experts=False, run_signal=True,
                                                                 boundary_norm=False)}
    for where, cfg in runs.items():
        model = Spidory(cfg)
        save(tmp_path / "runs" / where / "last.pt", model, make_optimizer(model, TCFG), 1, TCFG, corpus)
    found = {m["id"]: m for m in server.Registry(tmp_path).describe()}
    assert list(found)[-1] == "remote_v3_s0/plain_s0"  # v4 comes first
    token, nonorm = found["remote_mixed_token/token_signal_s0"], found["remote_mixed_plain/plain_signal_nonorm_s0"]
    assert token["group"] == nonorm["group"] == "v4, seed 0"
    assert token["per_token"] and token["run_signal"] and token["features"]["routing"] == "per token"
    assert "chosen per token" in token["note"] and "run number signal" in token["note"]
    assert nonorm["features"]["boundary norm"] == "no" and "no boundary normalization" in nonorm["note"]
    assert found["remote_v3_s0/plain_s0"]["features"]["run signal"] == "no"


def test_trace_gives_loss_runs_and_experts_per_token(v4_registry):
    events = []
    columns = [{"model": "token", "setting": {"kind": "frozen", "value": 2}},
               {"model": "token", "setting": {"kind": "neuron"}},
               {"model": "moe", "setting": {"kind": "frozen", "value": 3}},
               {"model": "plain", "setting": {"kind": "cycle", "value": 3}},
               {"model": "plain", "setting": {"kind": "neuron"}}]
    server.trace_columns(v4_registry, FakeTok(), {"text": "hello there friend", "columns": columns}, events.append)
    pieces = events[0]["pieces"]
    traces = {e["col"]: e for e in events if "tokens" in e}
    assert len(pieces) == len(FakeTok().encode("hello there friend").ids)
    assert any(e.get("error") for e in events if e.get("col") == 4) and events[-1] == {"finished": True}
    ids = torch.tensor([FakeTok().encode("hello there friend").ids])
    logits = v4_registry.get("token").infer(ids, k=2)[0]
    expected = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
    got = torch.tensor([t["loss"] for t in traces[0]["tokens"][1:]])
    assert traces[0]["tokens"][0]["loss"] is None and torch.allclose(got, expected, atol=1e-3)
    for t in traces[0]["tokens"]:
        assert t["runs"] == t["paid"] == 2 and len(t["path"]) == 2
    runs = [t["runs"] for t in traces[1]["tokens"]]  # the neuron chooses per token
    assert [t["paid"] for t in traces[1]["tokens"]] == [max(runs[i:]) for i in range(len(runs))]
    assert all(len(t["path"]) == t["runs"] for t in traces[1]["tokens"])
    assert all(t["runs"] == t["paid"] == 3 and len(set(t["path"])) <= 1 for t in traces[2]["tokens"])
    cycle = traces[3]["tokens"]
    assert [t["runs"] for t in cycle] == [i % 3 + 1 for i in range(len(cycle))]
    assert all(t["paid"] == 3 and t["path"] == [] for t in cycle)
    assert all(traces[c]["mflops"] > 0 for c in traces)


class FakeExams:
    def get(self, name):
        return [{"prompt": f"Q: {i}?\nA:", "answer": str(i), "level": 2} for i in range(5)]


def test_score_streams_one_verdict_per_item_and_column(v4_registry, monkeypatch):
    monkeypatch.setitem(server.EXAMS, "toy", ("maths", "toy", "toy.json"))
    events = []
    columns = [{"model": "moe", "setting": {"kind": "frozen", "value": 2}},
               {"model": "token", "setting": {"kind": "neuron"}},
               {"model": "plain", "setting": {"kind": "cycle", "value": 2}}]
    server.score_columns(v4_registry, FakeTok(), FakeExams(), {"exam": "toy", "indices": [3, 7], "columns": columns},
                         events.append)
    verdicts = [e for e in events if "item" in e]
    assert [(e["item"], e["col"]) for e in verdicts] == [(3, 0), (3, 1), (2, 0), (2, 1)]  # 7 wraps to 2
    assert all(isinstance(e["ok"], bool) and e["answer"] == str(e["item"]) for e in verdicts)
    assert any(e.get("error") for e in events if e.get("col") == 2) and events[-1] == {"finished": True}


def test_exam_answers_match_the_evaluation_scores(v4_registry):
    from spidory import evaluate

    model, exam = v4_registry.get("moe"), FakeExams().get("toy")
    ok = [evaluate.answer_item("maths", model, FakeTok(), item, k=2)[1] for item in exam]
    assert evaluate.maths_accuracy(model, FakeTok(), exam, len(exam), k=2) == sum(ok) / len(exam)


def test_exams_are_built_into_the_cache_only(tmp_path, tok):  # noqa: F811
    data, cache = tmp_path / "data", tmp_path / "cache"
    data.mkdir()
    _streams(data, tok)
    tok.save(str(data / "tokenizer.json"))
    before = {p.name: p.stat().st_mtime_ns for p in data.iterdir()}
    templates = Path(__file__).resolve().parents[1] / "templates"
    ui_exams.build(data, cache, templates, held_tokens=3_000, names=("explain", "mixed", "compose"))
    assert {p.name: p.stat().st_mtime_ns for p in data.iterdir()} == before
    for name in ("explain", "mixed", "compose"):
        rows = json.loads((cache / ui_exams.EXAMS[name][2]).read_text())
        assert rows and all("prompt" in r and "answer" in r for r in rows)
    found = ui_exams.Exams(data, cache, templates)
    assert len(found.get("compose")) == 300 and found.describe()[2]["size"] == len(found.get("explain"))
    with pytest.raises(ValueError):
        found.get("nothing")
