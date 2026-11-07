"""Stage 3 with the run neuron trained jointly with the model.

Three ways to train it, all with the two-sided penalty on each expert's average runs:
- "sampled": the model learns from one run count sampled from the neuron. The neuron learns
  from the losses at all run counts, weighted by its own probabilities.
- "average": the model learns from the average loss over 1, 2, 3 and 4 runs. The neuron learns
  as in "sampled".
- "mix": the final state is the neuron-weighted mix of every run's state, so the neuron learns
  only from the main loss.
Only the last run of each path has gradients (TBPTL depth 1), as in stage 2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import balance_penalty, run_penalty


def neuron_objective(probs: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Expected loss under the neuron's choices, relative to each sequence's average.

    Lowering it moves probability toward the run counts with lower loss.
    """
    advantage = losses - losses.mean(1, keepdim=True)
    return (probs * advantage.detach()).sum(1).mean()


def _per_sequence_loss(model, h, ks, experts, targets):
    logits = model._decode(h, ks, experts)
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(),
                           reduction="none").view(targets.shape).mean(1)


def joint_step(model, inputs: torch.Tensor, targets: torch.Tensor, tcfg, tau: float):
    cfg = model.cfg
    size, runs, device = inputs.size(0), cfg.max_runs, inputs.device
    rows = torch.arange(size, device=device)
    h0, pooled = model._encode(inputs)
    x = model._unit(h0)
    router_logits = model.router(pooled)
    gates = model.route(router_logits, tau, True)
    experts = gates.argmax(-1)
    probs = F.softmax(model.run_logits(pooled)[rows, experts].float(), dim=-1)

    # Every run's state and loss, without gradients.
    states = []
    with torch.no_grad():
        state = x.detach()
        for run in range(1, runs + 1):
            state = model._run_once(state, x.detach(), gates.detach())
            states.append(state)  # state after this run, before any between block
            if model.has_between and run < runs:
                state = model._between(state, run, x.detach())
        losses = torch.stack([_per_sequence_loss(model, states[k - 1], torch.full((size,), k, device=device),
                                                 experts, targets) for k in range(1, runs + 1)], 1)

    def start_of(ks):
        """The state before each sequence's last run: x for one run, else the stored state."""
        if runs == 1:
            return x
        before = torch.stack([x.detach()] + states[:-1], 1)[rows, ks - 1]
        return torch.where((ks == 1)[:, None, None], x, model._enter_last(before, ks, x))

    def last_run(ks):
        return model._run_once(start_of(ks), x, gates)

    with torch.no_grad():
        sampled = torch.multinomial(probs, 1).squeeze(1) + 1
        uniform = torch.randint(1, runs + 1, (size,), device=device)
        ks = torch.where(torch.rand(size, device=device) < tcfg.neuron_explore, uniform, sampled)

    parts = {}
    mode = cfg.neuron_train
    if mode == "average":
        main = torch.stack([_per_sequence_loss(model, last_run(torch.full((size,), k, device=device)),
                                               torch.full((size,), k, device=device), experts, targets)
                            for k in range(1, runs + 1)], 1).mean()
    elif mode == "mix":
        fresh = last_run(ks)
        stacked = torch.stack(states, 1).clone()
        stacked[rows, ks - 1] = fresh
        mixed = (probs.to(fresh.dtype)[:, :, None, None] * stacked).sum(1)
        main = _per_sequence_loss(model, mixed, ks, experts, targets).mean()
    else:
        main = _per_sequence_loss(model, last_run(ks), ks, experts, targets).mean()
    loss = main
    parts["ce"] = main.item()

    if mode in ("sampled", "average"):
        neuron = neuron_objective(probs, losses)
        loss = loss + tcfg.neuron_weight * neuron
        parts["neuron"] = neuron.item()
    penalty = run_penalty(probs, gates, model.run_targets)
    balance = balance_penalty(router_logits)
    loss = loss + tcfg.run_penalty * penalty + tcfg.balance_penalty * balance
    counts = torch.arange(1, runs + 1, device=device, dtype=probs.dtype)
    parts.update(run=penalty.item(), balance=balance.item(),
                 mean_runs=float((probs.detach() * counts).sum(1).mean()))
    return loss, parts
