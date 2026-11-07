"""Measure real inference speed of finished models, for dollars per million tokens."""
import json
import time
from pathlib import Path

import torch

from spidory.data import Corpus
from spidory.train import load, pick_device

device = pick_device("auto")
report = {}
for path in sorted(Path("runs").glob("*_s[0-9]/last.pt")):
    name = path.parent.name
    if name.endswith("warm") or "warm" in name:
        continue
    model, *_ = load(path, device)
    batch = Corpus("data", model.cfg.seq_len).fixed_windows("story", "test", 64)[:, :-1].to(device)
    speeds = {}
    for k in range(1, model.cfg.max_runs + 1):
        model.infer(batch, k=k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.time()
        for _ in range(10):
            model.infer(batch, k=k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        speeds[k] = round(10 * batch.numel() / (time.time() - began))
    report[name] = speeds
Path("runs/speed.json").write_text(json.dumps(report, indent=2))
