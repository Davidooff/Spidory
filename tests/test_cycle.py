from dataclasses import replace

import torch

from spidory.config import TrainConfig
from spidory.data import Corpus
from spidory.evaluate import cycle_loss
from spidory.model import Spidory
from spidory.train import cycle_ks, make_optimizer, sample_ks, stage3_start, train_steps

from test_spidory import CFG, TCFG, corpus, tokens, wake_experts  # noqa: F401


def test_cycle_goes_up_by_one_each_token_then_restarts():
    torch.manual_seed(0)
    ks = cycle_ks(5, 9, 4, "cpu")
    for row in ks:
        steps = (row[1:] - row[:-1]) % 4
        assert steps.eq(1).all()  # +1 each next token, 4 back to 1
        assert set(row.tolist()) == {1, 2, 3, 4}


def test_per_token_counts_match_per_sequence_counts_when_equal():
    model = Spidory(CFG).eval()
    wake_experts(model)
    batch = tokens(4)
    with torch.no_grad():
        for k in (1, 3):
            per_sequence = model(batch, ks=k)["logits"]
            per_token = model(batch, ks=torch.full(batch.shape, k))["logits"]
            assert torch.allclose(per_sequence, per_token, atol=1e-4)


def test_finished_tokens_keep_their_last_state():
    model = Spidory(CFG).eval()
    x = model._unit(model._encode(tokens(2))[0])
    ks = torch.ones(2, CFG.seq_len, dtype=torch.long)
    ks[:, 1::2] = 4
    with torch.no_grad():
        out = model._loop_tokens(x, None, ks)
        one_run = model._run_once(x, x, None)
    assert torch.allclose(out[:, 0], one_run[:, 0], atol=1e-5)  # token 0 stopped after run 1
    assert not torch.allclose(out[:, 1], one_run[:, 1], atol=1e-3)  # token 1 ran 4 times


def test_each_token_learns_from_its_own_last_run():
    model = Spidory(CFG).train()
    flags = []
    original = model._run_once
    model._run_once = lambda s, *a: (flags.append(s.requires_grad), original(s, *a))[1]
    batch = tokens(2)
    out = model(batch, ks=cycle_ks(2, CFG.seq_len, 4, "cpu"))
    out["logits"].sum().backward()
    assert flags == [False, False, False, False]  # every run input is detached
    assert model.input_block[0].qkv.weight.grad.abs().sum() > 0  # x still carries gradient


def test_stage_2b_cycles_per_token_and_stage_3_follows():
    cfg = replace(TCFG, stage1_steps=2, stage2_steps=2, cycle_steps=3)
    model = Spidory(CFG)
    experts = torch.zeros(4, dtype=torch.long)
    assert sample_ks(4, cfg, model, experts, 10).shape == (4, 10)
    assert sample_ks(3, cfg, model, experts, 10).shape == (4,)
    assert stage3_start(cfg) == 7 and sample_ks(7, cfg, model, experts, 10).shape == (4,)


def test_cycle_stage_trains_end_to_end_and_is_evaluated(corpus):  # noqa: F811
    cfg = replace(TCFG, stage1_steps=1, stage2_steps=1, cycle_steps=3, eval_windows=4)
    model = Spidory(CFG)
    train_steps(model, make_optimizer(model, cfg), corpus, cfg, 0, 5)
    found = cycle_loss(model, corpus, 4)
    assert set(found["loss"]) == {"story", "maths", "qa"}
    assert len(found["loss_by_token_runs"]) == CFG.max_runs


def test_pieces_give_the_same_update_as_the_whole_batch(corpus):  # noqa: F811
    cfg = replace(TCFG, stage1_steps=0, stage2_steps=0, cycle_steps=5, batch_size=8)
    grads = []
    for pieces in (1, 4):
        torch.manual_seed(0)
        model = Spidory(replace(CFG, use_experts=False))
        torch.manual_seed(1)
        corpus.reseed(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        train_steps(model, optimizer, corpus, replace(cfg, heavy_pieces=pieces), 0, 1)
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)
