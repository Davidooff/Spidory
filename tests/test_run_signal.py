"""Run signal: each run knows its number through per-run mixing weights and vectors."""
from dataclasses import replace

import pytest
import torch

from spidory.joint import joint_step
from spidory.model import Spidory
from spidory.train import cycle_ks, load, make_optimizer, save, train_steps

from test_spidory import CFG, TCFG, corpus, tokens, wake_experts  # noqa: F401

SIGNAL = replace(CFG, run_signal=True)


def pair(cfg):
    torch.manual_seed(0)
    plain = Spidory(replace(cfg, run_signal=False)).eval()
    torch.manual_seed(0)
    signal = Spidory(replace(cfg, run_signal=True)).eval()
    return plain, signal


@pytest.mark.parametrize("cfg", [CFG, replace(CFG, use_experts=False),
                                 replace(CFG, between_layers=1, between_mode="shared")])
def test_starts_exactly_as_dory(cfg):
    """a = b = 1 and zero vectors: the same seed gives the same outputs as without the signal."""
    plain, signal = pair(cfg)
    for name, value in plain.state_dict().items():
        assert torch.equal(value, signal.state_dict()[name]), name
    batch = tokens(3)
    for k in (1, 2, 4):
        assert torch.allclose(plain(batch, ks=k)["logits"], signal(batch, ks=k)["logits"], atol=1e-6)


def test_each_run_reads_its_own_row():
    model = Spidory(replace(SIGNAL, use_experts=False)).eval()
    batch = tokens(2)
    before = {k: model(batch, ks=k)["logits"] for k in (1, 2, 3)}
    with torch.no_grad():
        model.run_scale_input[1].zero_()  # run 2 stops mixing the input back in
    after = {k: model(batch, ks=k)["logits"] for k in (1, 2, 3)}
    assert torch.equal(before[1], after[1])
    assert not torch.allclose(before[2], after[2])
    assert not torch.allclose(before[3], after[3])  # run 3 starts from run 2's changed state


def test_only_the_last_run_learns_its_rows():
    """TBPTL depth 1: the rows of earlier runs get no gradient."""
    model = Spidory(SIGNAL)
    wake_experts(model)
    model(tokens(4), ks=3)["logits"].pow(2).mean().backward()
    for table in (model.run_scale_state, model.run_scale_input, model.run_signal_embed, model.expert_run_embed):
        grads = table.grad.abs().sum(-1)
        assert grads[2] > 0 and grads[0] == 0 and grads[1] == 0 and grads[3] == 0


def test_between_block_learns_its_gap_vector():
    model = Spidory(replace(SIGNAL, between_layers=1, between_mode="shared"))
    model(tokens(4), ks=3)["logits"].pow(2).mean().backward()
    grads = model.between_run_embed.grad.abs().sum(-1)
    assert grads[1] > 0 and grads[0] == 0  # the block before run 3 is gap 2


def test_without_boundary_normalization():
    """The user's version: no normalization at loop boundaries, the run signal does the mixing."""
    model = Spidory(replace(SIGNAL, boundary_norm=False))
    wake_experts(model)
    out = model(tokens(3), ks=4)["logits"]
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert model.run_scale_state.grad[3].abs().sum() > 0


def test_runs_past_the_maximum_reuse_the_last_row():
    model = Spidory(replace(SIGNAL, between_layers=1, between_mode="per_gap")).eval()
    logits, _, ks = model.infer(tokens(2), k=CFG.max_runs + 2)
    assert int(ks[0]) == CFG.max_runs + 2 and torch.isfinite(logits).all()


def test_per_token_cycle_and_grouped_inference_agree():
    model = Spidory(SIGNAL).eval()
    wake_experts(model)
    batch = tokens(5)
    together = model.infer(batch, k=3)[0]
    alone = torch.cat([model.infer(batch[i:i + 1], k=3)[0] for i in range(5)])
    assert torch.allclose(together, alone, atol=1e-5)
    ks = cycle_ks(5, CFG.seq_len, CFG.max_runs, batch.device)
    assert torch.isfinite(model.infer_tokens(batch, ks)).all()


@pytest.mark.parametrize("mode", ["sampled", "average", "mix"])
def test_joint_neuron_trains_with_the_signal(mode):
    model = Spidory(replace(SIGNAL, run_mode="joint", neuron_train=mode))
    wake_experts(model)
    batch = tokens(4)
    loss, _ = joint_step(model, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.run_signal_embed.grad.abs().sum() > 0


def test_trains_through_every_stage_and_reloads(corpus, tmp_path):  # noqa: F811
    cfg = replace(TCFG, stage1_steps=1, stage2_steps=1, cycle_steps=2, steps=6)
    model = Spidory(replace(SIGNAL, run_mode="joint"))
    optimizer = make_optimizer(model, cfg)
    train_steps(model, optimizer, corpus, cfg, 0, 6)
    assert not torch.equal(model.run_scale_input, torch.ones_like(model.run_scale_input))
    save(tmp_path / "last.pt", model, optimizer, 6, cfg, corpus)
    loaded, *_ = load(tmp_path / "last.pt", torch.device("cpu"))
    assert loaded.cfg.run_signal and torch.equal(loaded.run_scale_input, model.run_scale_input)
