"""Released weights: export to safetensors and load back give the same model, bit for bit."""
import json

import torch

from spidory.config import ModelConfig, TrainConfig, to_dict
from spidory.hub import export, load_folder
from spidory.model import Spidory


def _checkpoint(tmp_path, **cfg):
    model = Spidory(ModelConfig(d_model=64, n_heads=4, seq_len=32, vocab_size=128, n_input=1, n_core=2,
                                n_output=1, expert_dim=16, **cfg))
    path = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "optimizer": {}, "step": 7,
                "model_cfg": to_dict(model.cfg), "train_cfg": to_dict(TrainConfig()), "rng": {}, "extra": {}}, path)
    return model.eval(), path


def _same(a, b, k):
    tokens = torch.randint(0, 128, (2, 32), generator=torch.Generator().manual_seed(0))
    return torch.equal(a.infer(tokens, k=k)[0], b.infer(tokens, k=k)[0])


def test_export_round_trip_plain(tmp_path):
    model, path = _checkpoint(tmp_path, use_experts=False)
    loaded = load_folder(export(path, tmp_path / "out"))
    a, b = model.state_dict(), loaded.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert _same(model, loaded, 1) and _same(model, loaded, 3)


def test_export_round_trip_per_token_with_run_signal(tmp_path):
    model, path = _checkpoint(tmp_path, route_per_token=True, route_per_loop=True, run_signal=True)
    loaded = load_folder(export(path, tmp_path / "out"))
    assert _same(model, loaded, 3)


def test_export_keeps_config_and_drops_optimizer(tmp_path):
    _, path = _checkpoint(tmp_path, use_experts=False, max_runs=1)
    out = export(path, tmp_path / "out")
    config = json.loads((out / "config.json").read_text())
    assert config["model_cfg"]["max_runs"] == 1 and config["step"] == 7
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.safetensors"]
