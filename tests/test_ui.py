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
    assert found["plain"]["experts"] == 0 and len(found["moe"]["mflops"]) == CFG.max_runs


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
