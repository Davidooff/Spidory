"""Speed and peak memory of each training phase on this GPU, before a long run."""
import time
from dataclasses import replace

import torch

from spidory.config import ModelConfig, TrainConfig
from spidory.data import Corpus
from spidory.model import Spidory
from spidory.train import compile_for_speed, make_optimizer, train_steps

device = torch.device("cuda")
corpus = Corpus("data", 256, seed=0)
base = TrainConfig(batch_size=64, stage1_steps=20, stage2_steps=20, cycle_steps=20, steps=100, lr=6e-3, warmup_lr_steps=5)
cases = [("one run", 0, 20, {}), ("loops", 20, 40, {}), ("cycle", 40, 60, {}),
         ("neuron sampled", 60, 80, {"run_mode": "joint", "neuron_train": "sampled"}),
         ("neuron average", 60, 80, {"run_mode": "joint", "neuron_train": "average"}),
         ("neuron mix", 60, 80, {"run_mode": "joint", "neuron_train": "mix"}),
         ("two experts", 60, 80, {"run_mode": "joint", "active_experts": 2})]
for label, start, stop, extra in cases:
    torch.manual_seed(0)
    model = Spidory(replace(ModelConfig(), **extra)).to(device)
    compile_for_speed(model)
    optimizer = make_optimizer(model, base)
    train_steps(model, optimizer, corpus, base, start, start + 4)  # compile and warm up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.time()
    train_steps(model, optimizer, corpus, base, start + 4, stop)
    torch.cuda.synchronize()
    seconds = (time.time() - began) / (stop - start - 4)
    print(f"{label:15s} {seconds * 1000:7.0f} ms/step  {base.batch_size * 256 / seconds:9.0f} tok/s  "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:5.1f} GB", flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
