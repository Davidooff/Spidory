"""The router must actually decide: confident per input, balanced across experts, and able to
send clearly different data to different experts. The first nGPT runs failed all three."""
from dataclasses import replace

import torch
import torch.nn.functional as F

from spidory.config import TrainConfig
from spidory.model import Spidory, balance_penalty, compute_loss, routing_uncertainty

from test_spidory import CFG, TCFG, tokens


def two_kinds(size, seq, vocab, generator):
    """Half the batch uses low tokens, half high tokens, and each kind continues differently."""
    kind = torch.arange(size) % 2
    low, high = 2, vocab // 2
    start = torch.where(kind[:, None] == 0, torch.randint(low, high, (size, 1), generator=generator),
                        torch.randint(high, vocab, (size, 1), generator=generator))
    steps = torch.where(kind[:, None] == 0, 1, 3)
    width = torch.where(kind[:, None] == 0, high - low, vocab - high)
    base = torch.where(kind[:, None] == 0, low, high)
    seqs = base + (start - base + steps * torch.arange(seq)[None]) % width
    return seqs, kind


def test_balance_loss_punishes_collapse_and_sharpening_punishes_indecision():
    one = F.one_hot(torch.zeros(8, dtype=torch.long), 4).float()
    spread = F.one_hot(torch.arange(8) % 4, 4).float()
    confident_collapse = torch.full((8, 4), -5.0)
    confident_collapse[:, 0] = 5.0
    decided = torch.full((8, 4), -5.0)
    decided[torch.arange(8), torch.arange(8) % 4] = 5.0
    undecided = torch.zeros(8, 4)
    # Balance: a confident collapse onto one expert costs 4, a spread assignment the minimum, 1.
    assert float(balance_penalty(confident_collapse, one)) > 3.9
    assert abs(float(balance_penalty(decided, spread)) - 1.0) < 0.05
    # An undecided router looks balanced (1.0) to the balance loss: sharpening is what catches it.
    assert abs(float(balance_penalty(undecided, one)) - 1.0) < 1e-6
    assert float(routing_uncertainty(undecided)) > 0.99 > 0.01 > float(routing_uncertainty(decided))


def test_uncertainty_is_one_when_undecided_and_near_zero_when_decided():
    assert abs(float(routing_uncertainty(torch.zeros(4, 4))) - 1.0) < 1e-6
    assert float(routing_uncertainty(torch.tensor([[20.0, 0, 0, 0]]))) < 0.01


def test_router_learns_to_decide_and_separate_two_kinds_of_data():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    cfg = replace(CFG, n_experts=2, vocab_size=64, seq_len=32, prefix=8, d_model=64, n_core=1)
    model = Spidory(cfg).train()
    tcfg = replace(TCFG, balance_penalty=0.05, router_sharpen=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(400):
        batch, _ = two_kinds(32, cfg.seq_len, cfg.vocab_size, generator)
        out = model(batch[:, :-1], ks=1, tau=max(0.3, 1.0 - step / 300))
        loss, parts = compute_loss(out, batch[:, 1:], tcfg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_weights()
    model.eval()
    batch, kind = two_kinds(64, cfg.seq_len, cfg.vocab_size, generator)
    with torch.no_grad():
        probs = torch.softmax(model.router(model._encode(batch[:, :-1])[1]), -1)
    choice = probs.argmax(-1)
    assert probs.max(-1).values.mean() > 0.7  # it decides
    assert set(choice.tolist()) == {0, 1}  # both experts are used
    for k in (0, 1):  # each kind of data goes mostly to one expert
        share = (choice[kind == k] == choice[kind == k].mode().values).float().mean()
        assert share > 0.8


def runs_of_each_sequence(model, batch, k):
    """The expert chosen at each run, one sequence at a time."""
    found = []
    for i in range(len(batch)):
        model._route_record = []
        model.infer(batch[i:i + 1], k=k)
        found.append([int(r[0]) for r in model._route_record])
    model._route_record = None
    return found


def test_per_loop_routing_can_change_expert_between_runs():
    torch.manual_seed(0)
    model = Spidory(replace(CFG, route_per_loop=True)).eval()
    with torch.no_grad():
        model.router[-1].weight.normal_(std=5.0)
    found = runs_of_each_sequence(model, tokens(12), 4)
    assert all(len(r) == 4 for r in found)  # one routing per run
    assert any(len(set(r)) > 1 for r in found)  # some sequence switches expert between runs


def test_per_loop_routing_trains_and_counts_in_the_balance():
    model = Spidory(replace(CFG, route_per_loop=True)).train()
    batch = tokens(16)
    out = model(batch[:, :-1], ks=3)
    assert len(out["loop_routing"]) == 1  # the last run, which has gradients, re-routed
    loss, parts = compute_loss(out, batch[:, 1:], TCFG)
    loss.backward()
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert "router_uncertainty" in parts


def test_without_per_loop_routing_runs_keep_their_expert():
    torch.manual_seed(0)
    model = Spidory(CFG).eval()
    with torch.no_grad():
        model.router[-1].weight.normal_(std=5.0)
    assert all(len(set(r)) == 1 for r in runs_of_each_sequence(model, tokens(8), 3))
