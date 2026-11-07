"""Local playground: prompt trained Spidory models side by side and watch how they run.

Run: .venv/bin/python -m spidory.ui.server, then open http://127.0.0.1:8765
It only listens on this computer.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..config import ModelConfig
from ..flops import tflops_by_k
from ..model import Spidory

PAGE = Path(__file__).with_name("index.html")
CATALOG = [
    {"id": "big-100m", "name": "Experts 100M", "path": "runs/remote_100m/big_s0/last.pt",
     "note": "v2-100M. Run neuron trained after the model, on a frozen copy (not the joint design)."},
    {"id": "plain-24m-v3", "name": "Plain 24M, data v2", "path": "runs/remote_v3/plain_warm_s0/last.pt",
     "note": "v3 Plain after stage 2. Harder maths and bigger QA. No experts, no neuron."},
    {"id": "plain-24m-v2", "name": "Plain 24M, easy maths", "path": "runs/remote_big/plain_warm_s0/last.pt",
     "note": "v2 Plain after stage 2. Easy maths, old QA. No experts, no neuron."},
]


def load_model(path: Path) -> Spidory:
    """Weights only. mmap keeps the optimizer state on disk, so memory stays low."""
    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    model = Spidory(ModelConfig(**{"arch": "prenorm", **state["model_cfg"]}))
    weights = dict(state["model"])
    for key, value in model.state_dict().items():
        weights.setdefault(key, value)  # parts added after this checkpoint was saved
    model.load_state_dict(weights)
    return model.eval()


class Registry:
    def __init__(self, root: Path, catalog: list[dict]):
        self.root, self.models, self.lock = root, {}, threading.Lock()
        self.entries = [e for e in catalog if (root / e["path"]).exists()]

    def describe(self) -> list[dict]:
        found = []
        for entry in self.entries:
            state = torch.load(self.root / entry["path"], weights_only=True, map_location="cpu", mmap=True)
            cfg = ModelConfig(**{"arch": "prenorm", **state["model_cfg"]})
            ready = bool(state["model"].get("run_head_ready", torch.tensor(False)))
            found.append({**{k: entry[k] for k in ("id", "name", "note")},
                          "experts": cfg.n_experts if cfg.use_experts else 0, "max_runs": cfg.max_runs,
                          "neuron": cfg.use_experts and ready, "width": cfg.d_model,
                          "layers": [cfg.n_input, cfg.n_core, cfg.n_output],
                          "between": {"layers": cfg.between_layers, "mode": cfg.between_mode,
                                      "merge": cfg.between_merge, "boundary_norm": cfg.boundary_norm},
                          "mflops": {k: round(v * 1e6, 1) for k, v in tflops_by_k(cfg).items()}})
        return found

    def get(self, model_id: str) -> Spidory:
        with self.lock:
            if model_id not in self.models:
                entry = next(e for e in self.entries if e["id"] == model_id)
                self.models[model_id] = load_model(self.root / entry["path"])
            return self.models[model_id]


def setting_args(setting: dict, model: Spidory) -> dict:
    """Turns a column's run setting into inference arguments."""
    kind = setting.get("kind", "frozen")
    if kind == "neuron":
        if not (model.cfg.use_experts and bool(model.run_head_ready)):
            raise ValueError("this model has no trained run neuron")
        return {"learned": True}
    if kind == "mean":
        if not model.cfg.use_experts:
            raise ValueError("an average needs experts; use a frozen run count")
        return {"mean_runs": float(setting["value"])}
    runs = int(setting.get("value", 1))
    if not 1 <= runs <= model.cfg.max_runs:
        raise ValueError(f"runs must be 1 to {model.cfg.max_runs}")
    return {"k": runs}


@torch.no_grad()
def step(model: Spidory, ids: list[int], choice: dict, temperature: float, generator: torch.Generator) -> dict:
    """One generated token, with what the model decided on the way."""
    window = torch.tensor([ids[-model.cfg.seq_len:]])
    logits, experts, runs = model.infer(window, **choice)
    probs = F.softmax(logits[0, -1].float(), -1)
    if temperature <= 0:
        token = int(probs.argmax())
    else:
        token = int(torch.multinomial(F.softmax(logits[0, -1].float() / temperature, -1), 1, generator=generator))
    top = probs.topk(5)
    expert = int(experts[0])
    neuron = None
    if model.cfg.use_experts and bool(model.run_head_ready):
        pooled = model._encode(window)[1]
        neuron = [round(float(p), 4) for p in F.softmax(model.run_logits(pooled)[0, max(expert, 0)].float(), -1)]
    return {"token": token, "expert": expert, "runs": int(runs[0]), "neuron": neuron,
            "top": [[int(i), round(float(p), 4)] for p, i in zip(top.values, top.indices)]}


def run_columns(registry: Registry, tok: Tokenizer, request: dict, emit) -> None:
    """Generates for all columns in turn, one token each, so they progress side by side."""
    eod = tok.token_to_id("[EOD]")
    prompt = tok.encode(request["prompt"]).ids or [eod]
    steps = max(1, min(int(request.get("max_tokens", 60)), 300))
    temperature = float(request.get("temperature", 0.0))
    columns = []
    for index, column in enumerate(request["columns"]):
        try:
            model = registry.get(column["model"])
            choice = setting_args(column["setting"], model)
            cost = tflops_by_k(model.cfg)
            columns.append({"index": index, "model": model, "choice": choice, "ids": list(prompt),
                            "text": "", "done": False, "cost": cost,
                            "generator": torch.Generator().manual_seed(int(request.get("seed", 0)))})
        except (ValueError, StopIteration, KeyError) as error:
            emit({"col": index, "error": str(error)})
    for _ in range(steps):
        active = [c for c in columns if not c["done"]]
        if not active:
            break
        for column in active:
            began = time.perf_counter()
            found = step(column["model"], column["ids"], column["choice"], temperature, column["generator"])
            seconds = time.perf_counter() - began
            if found["token"] == eod:
                column["done"] = True
                emit({"col": column["index"], "end": True})
                continue
            column["ids"].append(found["token"])
            full = tok.decode(column["ids"][len(prompt):])
            piece, column["text"] = full[len(column["text"]):], full
            emit({"col": column["index"], "text": piece, "ms": round(seconds * 1000),
                  "mflops": round(column["cost"][found["runs"]] * 1e6, 1),
                  "top": [[tok.decode([i]), p] for i, p in found["top"]],
                  **{k: found[k] for k in ("expert", "runs", "neuron")}})
    emit({"finished": True})


def make_handler(registry: Registry, tok: Tokenizer):
    gpu_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/models":
                self._send(200, json.dumps(registry.describe()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/run":
                self._send(404, b"not found", "text/plain")
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            def emit(event: dict) -> None:
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()

            with gpu_lock:
                try:
                    run_columns(registry, tok, request, emit)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the page pressed Stop

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokenizer", default="data_large/tokenizer.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    registry = Registry(Path("."), CATALOG)
    tok = Tokenizer.from_file(args.tokenizer)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(registry, tok))
    print(f"Spidory playground on http://127.0.0.1:{args.port} with {len(registry.entries)} models", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
