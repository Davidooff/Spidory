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
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..config import ModelConfig
from ..evaluate import answer_item
from ..flops import tflops_by_k
from ..model import Spidory
from .exams import EXAMS, Exams

PAGE = Path(__file__).with_name("index.html")
EXTRA_RUNS = 8  # frozen runs may go past training's max_runs, up to this, to test extrapolation
GROUPS = {  # folder under runs/ -> how the page groups its models, in this order
    "remote_mixed_plain": "v4, seed 0", "remote_mixed_token": "v4, seed 0",
    "remote_mixed_perloop_between": "v4, seed 0",
    "remote_v3_s0": "v3, seed 0", "remote_v3_s1": "v3, seed 1", "remote_between": "Between block, seed 0",
    "remote_100m": "100M (older)", "remote_big": "v2 (older)", "remote_v3": "v3 first try (older)",
    "remote": "v1 (older)",
}
def load_model(path: Path) -> Spidory:
    """Weights only. mmap keeps the optimizer state on disk, so memory stays low."""
    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    model = Spidory(ModelConfig(**{"arch": "prenorm", **state["model_cfg"]}))
    weights = dict(state["model"])
    for key, value in model.state_dict().items():
        weights.setdefault(key, value)  # parts added after this checkpoint was saved
    model.load_state_dict(weights)
    return model.eval()


def discover(root: Path) -> list[dict]:
    """Every run's final checkpoint under runs/<folder>/<run>/last.pt, grouped by folder."""
    order = list(GROUPS)
    found = []
    for path in sorted((root / "runs").glob("*/*/last.pt")):
        folder, run = path.parent.parent.name, path.parent.name
        if folder not in GROUPS:
            continue
        found.append({"id": f"{folder}/{run}", "name": run, "group": GROUPS[folder],
                      "path": path.relative_to(root).as_posix(), "note": ""})
    return sorted(found, key=lambda e: (order.index(e["path"].split("/")[1]), e["name"]))


def features(cfg: ModelConfig, ready: bool) -> dict[str, str]:
    """What a model has, feature by feature, so the page can show which features two columns differ in."""
    routing = ("none" if not cfg.use_experts else "per token" if cfg.route_per_token
               else "per sequence, each run" if cfg.route_per_loop else "per sequence")
    neuron = ("none" if not cfg.use_experts or not ready else
              f"{cfg.neuron}, {cfg.neuron_train}" if cfg.run_mode == "joint" else "post-hoc")
    between = "none"
    if cfg.between_layers:
        between = f"{cfg.between_layers} layers, {cfg.between_mode.replace('_', ' ')}"
        between += ", learned merge" if cfg.between_merge else ""
    return {"experts": f"{cfg.n_experts}, {cfg.active_experts} active" if cfg.use_experts else "none",
            "routing": routing, "neuron": neuron,
            "run signal": "yes" if getattr(cfg, "run_signal", False) else "no",
            "boundary norm": "yes" if cfg.boundary_norm else "no",
            "between block": between, "blocks": cfg.arch}


def describe_cfg(cfg: ModelConfig, ready: bool) -> str:
    """A short plain description of what a model is."""
    if not cfg.use_experts:
        kind = "Plain: no experts, no neuron"
    else:
        kind = f"{cfg.n_experts} experts, {cfg.active_experts} active"
        if cfg.route_per_token:
            kind += ", expert and runs chosen per token"
        elif cfg.route_per_loop:
            kind += ", router picks again each run"
        if ready and cfg.run_mode == "joint":
            kind += f", run neuron ({cfg.neuron}, trained jointly, {cfg.neuron_train})"
        elif ready:
            kind += ", run neuron trained after the model (post-hoc)"
        else:
            kind += ", no trained neuron (constant runs or warm-up)"
    if cfg.between_layers:
        merge = (" with learned merge" + ("" if cfg.boundary_norm else ", no normalization")
                 if cfg.between_merge else "")
        kind += f"; between block {cfg.between_layers} layers, {cfg.between_mode.replace('_', ' ')}{merge}"
    if getattr(cfg, "run_signal", False):
        kind += "; run number signal"
        kind += "" if cfg.boundary_norm else ", no boundary normalization"
    return kind


class Registry:
    """Models found on disk. With no catalog it looks again on every listing, so runs copied
    in while the page is open appear after a reload."""

    def __init__(self, root: Path, catalog: list[dict] | None = None):
        self.root, self.catalog, self.models, self.lock = root, catalog, {}, threading.Lock()
        self._cache: dict[str, tuple[float, dict]] = {}
        self.entries = self._scan()

    def _scan(self) -> list[dict]:
        entries = discover(self.root) if self.catalog is None else self.catalog
        return [e for e in entries if (self.root / e["path"]).exists()]

    def describe(self) -> list[dict]:
        found = []
        for entry in self._scan():
            try:
                found.append(self._describe(entry))
            except Exception:  # noqa: BLE001  a checkpoint still being copied in
                continue
        known = {f["id"] for f in found}
        self.entries = [e for e in self._scan() if e["id"] in known]
        return found

    def _describe(self, entry: dict) -> dict:
        stamp = (self.root / entry["path"]).stat().st_mtime
        cached = self._cache.get(entry["id"])
        if cached and cached[0] == stamp:
            return cached[1]
        state = torch.load(self.root / entry["path"], weights_only=True, map_location="cpu", mmap=True)
        cfg = ModelConfig(**{"arch": "prenorm", **state["model_cfg"]})
        ready = bool(state["model"].get("run_head_ready", torch.tensor(False)))
        with torch.device("meta"):  # counts parameters without allocating them
            params = Spidory(cfg).num_params()
        found = {**{k: entry[k] for k in ("id", "name")}, "group": entry.get("group", "Models"),
                 "note": entry.get("note") or describe_cfg(cfg, ready),
                 "experts": cfg.n_experts if cfg.use_experts else 0, "max_runs": cfg.max_runs,
                 "extra_runs": EXTRA_RUNS, "neuron": cfg.use_experts and ready,
                 "active_experts": cfg.active_experts, "per_loop": cfg.route_per_loop,
                 "per_token": cfg.route_per_token, "run_signal": cfg.run_signal,
                 "boundary_norm": cfg.boundary_norm, "features": features(cfg, ready),
                 "width": cfg.d_model, "params": params,
                 "layers": [cfg.n_input, cfg.n_core, cfg.n_output],
                 "between": {"layers": cfg.between_layers, "mode": cfg.between_mode,
                             "merge": cfg.between_merge, "boundary_norm": cfg.boundary_norm},
                 "mflops": {k: round(v * 1e6, 1) for k, v in tflops_by_k(cfg, EXTRA_RUNS).items()}}
        self._cache[entry["id"]] = (stamp, found)
        self.models.pop(entry["id"], None)  # a replaced checkpoint loads again
        return found

    def get(self, model_id: str) -> Spidory:
        with self.lock:
            if model_id not in self.models:
                entry = next(e for e in self.entries if e["id"] == model_id)
                self.models[model_id] = load_model(self.root / entry["path"])
            return self.models[model_id]


def expert_args(lock: dict | None, model: Spidory) -> dict:
    """A column's expert lock: the router decides, or one expert runs in every loop."""
    if not lock or lock.get("kind", "router") == "router":
        return {}
    if not model.cfg.use_experts:
        raise ValueError("this model has no experts to lock")
    expert = int(lock["value"])
    if not 0 <= expert < model.cfg.n_experts:
        raise ValueError(f"expert must be 0 to {model.cfg.n_experts - 1}")
    return {"expert": expert}


def setting_args(setting: dict, model: Spidory) -> dict:
    """Turns a column's run setting into inference arguments."""
    kind = setting.get("kind", "frozen")
    if kind == "cycle":
        top = int(setting.get("value", model.cfg.max_runs))
        if not 2 <= top <= model.cfg.max_runs:
            raise ValueError(f"a per-token cycle goes up to 2 to {model.cfg.max_runs} runs")
        return {"cycle": top}
    if kind == "neuron":
        if not (model.cfg.use_experts and bool(model.run_head_ready)):
            raise ValueError("this model has no trained run neuron")
        return {"learned": True}
    if kind == "mean":
        if not model.cfg.use_experts:
            raise ValueError("an average needs experts; use a frozen run count")
        return {"mean_runs": float(setting["value"])}
    runs = int(setting.get("value", 1))
    if not 1 <= runs <= EXTRA_RUNS:
        raise ValueError(f"runs must be 1 to {EXTRA_RUNS}")
    return {"k": runs}


@torch.no_grad()
def step(model: Spidory, ids: list[int], choice: dict, temperature: float, generator: torch.Generator) -> dict:
    """One generated token, with what the model decided on the way."""
    window = torch.tensor([ids[-model.cfg.seq_len:]])
    model._route_record = []
    try:
        if "cycle" in choice:
            # Per-token runs, as in training stage 2b: the token at position t gets t % top + 1
            # runs, counted from the start of the text. All tokens pass through every run, since
            # attention needs them, but each keeps the state of its own last run.
            top, expert_lock = choice["cycle"], choice.get("expert")
            start = max(0, len(ids) - model.cfg.seq_len)
            ks = (torch.arange(start, start + window.size(1)) % top + 1)[None]
            logits = model.infer_tokens(window, ks, expert=expert_lock)
            experts = (torch.tensor([expert_lock]) if expert_lock is not None
                       else model.decide(window, k=1)[0])
            runs = ks[:, -1:]
        else:
            logits, experts, runs = model.infer(window, **choice)
        path = [int(r[0].flatten()[-1]) for r in model._route_record]  # the expert in each run, last token
    finally:
        model._route_record = None
    probs = F.softmax(logits[0, -1].float(), -1)
    if temperature <= 0:
        token = int(probs.argmax())
    else:
        token = int(torch.multinomial(F.softmax(logits[0, -1].float() / temperature, -1), 1, generator=generator))
    top = probs.topk(5)
    # Per-token models choose per token: show the choice for the last token, the one predicted from.
    expert, run_count = int(experts[0].flatten()[-1]), int(runs[0].flatten()[-1])
    path = path[:run_count]  # with per-token runs, later runs belong to other tokens
    neuron = None
    if model.cfg.use_experts and bool(model.run_head_ready) and "cycle" not in choice:
        h0, pooled = model._encode(window)
        pooled = model._routing_input(h0, pooled)
        pooled = pooled[:, -1] if model.cfg.route_per_token else pooled
        neuron = [round(float(p), 4) for p in F.softmax(model.run_logits(pooled)[0, max(expert, 0)].float(), -1)]
    return {"token": token, "expert": expert, "runs": run_count, "neuron": neuron, "path": path,
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
            choice = {**setting_args(column["setting"], model), **expert_args(column.get("experts"), model)}
            cost = tflops_by_k(model.cfg, EXTRA_RUNS)
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
            paid = column["choice"].get("cycle", found["runs"])  # a cycle computes every run for all
            emit({"col": column["index"], "text": piece, "ms": round(seconds * 1000),
                  "mflops": round(column["cost"][paid] * 1e6, 1),
                  "top": [[tok.decode([i]), p] for i, p in found["top"]],
                  **{k: found[k] for k in ("expert", "runs", "neuron", "path")}})
    emit({"finished": True})


def _pieces(tok: Tokenizer, ids: list[int]) -> list[str]:
    """The text of each token as it appears in the whole decoded text, spaces included."""
    pieces, before = [], ""
    for end in range(1, len(ids) + 1):
        now = tok.decode(ids[:end])
        pieces.append(now[len(before):])
        before = now
    return pieces


@torch.no_grad()
def trace(model: Spidory, ids: list[int], choice: dict) -> dict:
    """One forward pass over a text: per token, the loss of predicting it, its run count, the
    runs it pays for, and its expert in each run."""
    window = torch.tensor([ids[-model.cfg.seq_len:]])
    length = window.size(1)
    model._route_record = []
    try:
        if "cycle" in choice:
            top = choice["cycle"]
            ks = (torch.arange(length) % top + 1)[None]
            logits = model.infer_tokens(window, ks, expert=choice.get("expert"))
            runs = ks[0]
            paid = torch.full_like(runs, top)  # every token passes every run, as attention needs them
        else:
            logits, _, ks = model.infer(window, **choice)
            runs = ks[0] if ks.dim() == 2 else ks[0].repeat(length)
            # A token runs as often as any later token, because their attention reads its state.
            paid = runs.flip(-1).cummax(-1).values.flip(-1) if ks.dim() == 2 else runs
        record = [r[0] for r in model._route_record]
    finally:
        model._route_record = None
    losses = F.cross_entropy(logits[0, :-1].float(), window[0, 1:], reduction="none")
    tokens = []
    for t in range(length):
        count = int(runs[t])
        path = [int(r[t] if r.dim() else r) for r in record[:count]]
        tokens.append({"loss": None if t == 0 else round(float(losses[t - 1]), 4), "runs": count,
                       "paid": int(paid[t]), "path": path})
    return {"tokens": tokens, "mean_loss": round(float(losses.mean()), 4) if length > 1 else None,
            "mean_paid": round(float(paid.float().mean()), 3)}


def trace_columns(registry: Registry, tok: Tokenizer, request: dict, emit) -> None:
    """Every column's trace of the same text, one event per column."""
    ids = tok.encode(request["text"]).ids
    if len(ids) < 2:
        emit({"error": "the text needs at least two tokens"})
        return
    seq_len = max((registry.get(c["model"]).cfg.seq_len for c in request["columns"]), default=256)
    shown = ids[-seq_len:]
    emit({"pieces": _pieces(tok, shown), "cut": len(ids) - len(shown)})
    for index, column in enumerate(request["columns"]):
        try:
            model = registry.get(column["model"])
            choice = {**setting_args(column["setting"], model), **expert_args(column.get("experts"), model)}
            found = trace(model, shown, choice)
            cost = tflops_by_k(model.cfg, EXTRA_RUNS)
            found["mflops"] = round(sum(cost[t["paid"]] for t in found["tokens"]) / len(shown) * 1e6, 1)
            emit({"col": index, **found})
        except (ValueError, StopIteration, KeyError) as error:
            emit({"col": index, "error": str(error)})
    emit({"finished": True})


def score_columns(registry: Registry, tok: Tokenizer, exams: Exams, request: dict, emit) -> None:
    """Exam items in every column, one verdict per item and column, item by item."""
    name = request["exam"]
    exam = exams.get(name)
    kind = EXAMS[name][0]
    indices = [int(i) % len(exam) for i in request["indices"]][:200]
    columns = []
    for index, column in enumerate(request["columns"]):
        try:
            model = registry.get(column["model"])
            choice = {**setting_args(column["setting"], model), **expert_args(column.get("experts"), model)}
            if "cycle" in choice:
                raise ValueError("exams run with one run setting per reply; pick a frozen count, average or neuron")
            columns.append((index, model, choice))
        except (ValueError, StopIteration, KeyError) as error:
            emit({"col": index, "error": str(error)})
    for item_index in indices:
        item = exam[item_index]
        for index, model, choice in columns:
            began = time.perf_counter()
            text, ok = answer_item(kind, model, tok, item, **choice)
            emit({"col": index, "item": item_index, "text": text, "ok": bool(ok),
                  "answer": str(item.get("answer", "")), "ms": round((time.perf_counter() - began) * 1000)})
    emit({"finished": True})


def make_handler(registry: Registry, tok: Tokenizer, exams: Exams | None = None):
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
            elif self.path == "/api/exams" and exams is not None:
                self._send(200, json.dumps(exams.describe()).encode(), "application/json")
            elif self.path.startswith("/api/exam?") and exams is not None:
                query = parse_qs(urlparse(self.path).query)
                try:
                    with gpu_lock:  # building an exam the first time takes a while
                        exam = exams.get(query["name"][0])
                    index = int(query.get("index", ["0"])[0]) % len(exam)
                    body = {"index": index, "size": len(exam), "item": exam[index]}
                    self._send(200, json.dumps(body).encode(), "application/json")
                except (ValueError, KeyError) as error:
                    self._send(400, json.dumps({"error": str(error)}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            jobs = {"/api/run": lambda request, emit: run_columns(registry, tok, request, emit),
                    "/api/trace": lambda request, emit: trace_columns(registry, tok, request, emit)}
            if exams is not None:
                jobs["/api/score"] = lambda request, emit: score_columns(registry, tok, exams, request, emit)
            if self.path not in jobs:
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
                    jobs[self.path](request, emit)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the page pressed Stop

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokenizer", default="data_v3/tokenizer.json", help="relative to --root")
    parser.add_argument("--root", default=".", help="project folder that holds runs/")
    parser.add_argument("--data", default="data_v3", help="data folder for the maths exam and the "
                        "held-out text the other exams are built from, relative to --root")
    parser.add_argument("--templates", default=str(Path(__file__).resolve().parents[2] / "templates"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    registry = Registry(root)
    tok = Tokenizer.from_file(str(root / args.tokenizer))
    exams = Exams(root / args.data, root / "runs" / "ui_exams", Path(args.templates))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(registry, tok, exams))
    print(f"Spidory playground on http://127.0.0.1:{args.port} with {len(registry.entries)} models", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
