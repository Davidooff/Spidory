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

from .model import routing_losses, run_penalty


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


def _per_token_loss(model, h, ks, experts, targets):
    logits = model._decode(h, ks, experts)
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="none").view(targets.shape)


def joint_step_tokens(model, inputs: torch.Tensor, targets: torch.Tensor, tcfg, tau: float):
    """The joint step with one expert and one run count per token.

    Every run passes all tokens, since attention needs them; a token's output is the state after
    its own last run. The neuron learns from each token's losses at every run count, weighted by
    its own probabilities, as in the per-sequence "sampled" way.

    Approximation: a token's loss at k is measured with every token at k. With per-token run
    counts, the other tokens it attends to sit at their own counts instead. A review measured the
    gap at about 10% of the spread between run counts on a random model.
    """
    cfg = model.cfg
    size, tokens, runs, device = inputs.size(0), inputs.size(1), cfg.max_runs, inputs.device
    h0, _ = model._encode(inputs)
    x = model._unit(h0)
    model._route_tau = tau
    model._loop_router_logits = []
    router_logits = model.router(x)  # (batch, tokens, experts)
    gates = model.route(router_logits, tau, True)
    experts = gates.argmax(-1)
    probs = F.softmax(model.run_probs_for(x, experts).float(), dim=-1)  # (batch, tokens, runs)

    # Each token's loss at every run count, without gradients.
    with torch.no_grad():
        state, losses = x.detach(), []
        for run in range(1, runs + 1):
            state = model._run_once(state, x.detach(), gates.detach(), run)
            losses.append(_per_token_loss(model, state, torch.full_like(experts, run), experts, targets))
            if model.has_between and run < runs:
                state = model._between(state, run, x.detach())
        losses = torch.stack(losses, -1)

    parts = {}
    if cfg.neuron_train == "average":
        # The model learns from every run count: each run's output, from detached inputs.
        state, main = x, 0.0
        for run in range(1, runs + 1):
            entering = state if (model.has_between and run > 1) else state.detach()
            new = model._run_once(entering, x, gates, run)
            main = main + _per_token_loss(model, new, torch.full_like(experts, run), experts, targets).mean()
            state = new.detach()
            if model.has_between and run < runs:
                state = model._between(state, run, x)
        main = main / runs
    else:
        with torch.no_grad():
            sampled = torch.multinomial(probs.reshape(-1, runs), 1).view(size, tokens) + 1
            uniform = torch.randint(1, runs + 1, (size, tokens), device=device)
            ks = torch.where(torch.rand(size, tokens, device=device) < tcfg.neuron_explore, uniform, sampled)
        h = model._loop_tokens(x, gates, ks)
        main = _per_token_loss(model, h, ks, experts, targets).mean()
    loss = main
    parts["ce"] = main.item()
    neuron = neuron_objective(probs.reshape(-1, runs), losses.reshape(-1, runs))
    loss = loss + tcfg.neuron_weight * neuron
    parts["neuron"] = neuron.item()
    penalty = run_penalty(probs, gates, model.run_targets)
    routing, routing_parts = routing_losses(router_logits, gates, model._loop_router_logits, tcfg)
    model._loop_router_logits = None
    loss = loss + tcfg.run_penalty * penalty + routing
    counts = torch.arange(1, runs + 1, device=device, dtype=probs.dtype)
    parts.update(routing_parts)
    parts.update(run=penalty.item(), mean_runs=float((probs.detach() * counts).sum(-1).mean()))
    return loss, parts


def joint_step(model, inputs: torch.Tensor, targets: torch.Tensor, tcfg, tau: float):
    if model.cfg.route_per_token:
        return joint_step_tokens(model, inputs, targets, tcfg, tau)
    cfg = model.cfg
    size, runs, device = inputs.size(0), cfg.max_runs, inputs.device
    rows = torch.arange(size, device=device)
    h0, pooled = model._encode(inputs)
    x = model._unit(h0)
    model._route_tau = tau
    model._loop_router_logits = []
    router_logits = model.router(pooled)
    gates = model.route(router_logits, tau, True)
    experts = gates.argmax(-1)
    probs = F.softmax(model.run_logits(pooled)[rows, experts].float(), dim=-1)

    # Every run's state and loss, without gradients.
    states = []
    with torch.no_grad():
        state = x.detach()
        for run in range(1, runs + 1):
            state = model._run_once(state, x.detach(), gates.detach(), run)
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
        return model._run_once(start_of(ks), x, gates, ks)

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
    routing, routing_parts = routing_losses(router_logits, gates, model._loop_router_logits, tcfg)
    model._loop_router_logits = None
    loss = loss + tcfg.run_penalty * penalty + routing
    counts = torch.arange(1, runs + 1, device=device, dtype=probs.dtype)
    parts.update(routing_parts)
    parts.update(run=penalty.item(),
                 mean_runs=float((probs.detach() * counts).sum(1).mean()))
    return loss, parts
