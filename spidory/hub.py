"""Released weights: export a training checkpoint to safetensors, and load a released model.

A released model is a folder with `model.safetensors` (fp32 weights, no optimizer state) and
`config.json` (`model_cfg`, `train_cfg`, `step`). The paper's 30 models live in the Hugging Face
repo `Davidow/spidory`, one folder per model and seed, for example `token_signal_s0`.

    from spidory.hub import load
    model, tok = load("token_signal_s0")
    out = model.infer(tok_ids, k=3)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import Spidory

REPO = "Davidow/spidory"


def full_weights(model: Spidory, weights: dict) -> dict:
    """Fills in parts added after a checkpoint was saved, as the playground does."""
    weights = dict(weights)
    for key, value in model.state_dict().items():
        weights.setdefault(key, value)
    return weights


def export(checkpoint: Path, out: Path) -> Path:
    """Writes `out/model.safetensors` and `out/config.json` from a training checkpoint."""
    from safetensors.torch import save_file

    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    cfg = {"arch": "prenorm", **state["model_cfg"]}
    model = Spidory(ModelConfig(**cfg))
    weights = full_weights(model, state["model"])
    model.load_state_dict(weights)  # the same check loading makes: every key, every shape
    out.mkdir(parents=True, exist_ok=True)
    # Tied weights share storage, which safetensors refuses; loading ties them again.
    save_file({k: v.detach().contiguous().clone() for k, v in weights.items()}, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(
        {"model_cfg": cfg, "train_cfg": state["train_cfg"], "step": state["step"]}, indent=1) + "\n")
    return out


def load_folder(folder: Path) -> Spidory:
    """A released model from a local folder, in eval mode."""
    from safetensors.torch import load_file

    config = json.loads((Path(folder) / "config.json").read_text())
    model = Spidory(ModelConfig(**config["model_cfg"]))
    model.load_state_dict(load_file(str(Path(folder) / "model.safetensors")))
    return model.eval()


def load(name: str, repo: str = REPO, revision: str | None = None):
    """Downloads one released model and the tokenizer. Returns (model, tokenizer)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    files = [hf_hub_download(repo, f"{name}/{f}", revision=revision) for f in ("config.json", "model.safetensors")]
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=revision))
    return load_folder(Path(files[0]).parent), tok
