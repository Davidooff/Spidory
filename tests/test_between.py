from dataclasses import replace

import pytest
import torch

from spidory.config import ModelConfig
from spidory.flops import tflops_by_k
from spidory.joint import joint_step
from spidory.model import Spidory
from spidory.train import cycle_ks, load, make_optimizer, save

from test_spidory import CFG, TCFG, corpus, tokens, wake_experts  # noqa: F401


def between_model(mode, layers=2, **extra):
    torch.manual_seed(0)
    model = Spidory(replace(CFG, between_layers=layers, between_mode=mode, **extra))
    wake_experts(model)
    return model


def count_between_calls(model):
    calls = []
    original = model._between
    model._between = lambda s, gap, x: (calls.append((gap, torch.is_grad_enabled())), original(s, gap, x))[1]
    return calls


@pytest.mark.parametrize("mode", ["shared", "per_gap"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_between_runs_k_minus_one_times(mode, k):
    model = between_model(mode).eval()
    calls = count_between_calls(model)
    model.infer(tokens(1), k=k)
    assert [gap for gap, _ in calls] == list(range(1, k))


def test_per_gap_has_its_own_blocks_and_shared_reuses_one():
    assert len(between_model("per_gap").between) == CFG.max_runs - 1
    assert len(between_model("shared").between) == 1
    per_gap = between_model("per_gap").eval()
    with torch.no_grad():
        s = per_gap._unit(torch.randn(2, CFG.seq_len, CFG.d_model))
        assert not torch.allclose(per_gap._between(s, 1, s), per_gap._between(s, 2, s), atol=1e-4)
        shared = between_model("shared").eval()
        assert torch.allclose(shared._between(s, 1, s), shared._between(s, 3, s))


def test_no_between_block_changes_nothing():
    torch.manual_seed(0)
    plain = Spidory(CFG).eval()
    torch.manual_seed(0)
    zero = Spidory(replace(CFG, between_layers=0)).eval()
    batch = tokens(2)
    assert not hasattr(zero, "between")
    assert torch.allclose(plain.infer(batch, k=3)[0], zero.infer(batch, k=3)[0])


@pytest.mark.parametrize("mode", ["shared", "per_gap"])
def test_only_the_last_between_block_learns(mode):
    model = between_model(mode).train()
    calls = count_between_calls(model)
    batch = tokens(4)
    out = model(batch[:, :-1], ks=3)
    out["logits"].sum().backward()
    assert calls == [(1, False), (2, True)]  # gap 1 without gradients, gap 2 with them
    if mode == "per_gap":
        assert model.between[0][0].qkv.weight.grad is None
        assert model.between[1][0].qkv.weight.grad.abs().sum() > 0
        assert model.between[2][0].qkv.weight.grad is None
    else:
        assert model.between[0][0].qkv.weight.grad.abs().sum() > 0


def test_between_block_gets_no_gradient_at_one_run():
    model = between_model("shared").train()
    out = model(tokens(2)[:, :-1], ks=1)
    out["logits"].sum().backward()
    assert model.between[0][0].qkv.weight.grad is None


@pytest.mark.parametrize("mode", ["shared", "per_gap"])
def test_grouped_inference_matches_single_sequences(mode):
    model = between_model(mode).eval()
    batch = tokens(6)
    ks = torch.tensor([1, 2, 3, 4, 2, 3])
    with torch.no_grad():
        mixed = model(batch, ks=ks)["logits"]
    for i in range(6):
        assert torch.allclose(mixed[i], model.infer(batch[i:i + 1], k=int(ks[i]))[0][0], atol=1e-4)


def test_per_token_counts_match_per_sequence_counts_with_between():
    model = between_model("per_gap").eval()
    batch = tokens(3)
    with torch.no_grad():
        for k in (2, 4):
            assert torch.allclose(model(batch, ks=k)["logits"],
                                  model(batch, ks=torch.full(batch.shape, k))["logits"], atol=1e-4)


def test_cycle_stage_trains_the_between_blocks():
    model = between_model("per_gap").train()
    batch = tokens(2)
    model(batch, ks=cycle_ks(2, CFG.seq_len, 4, "cpu"))["logits"].sum().backward()
    assert all(block[0].qkv.weight.grad is not None for block in model.between)


@pytest.mark.parametrize("mode", ["shared", "per_gap"])
def test_joint_neuron_step_trains_the_between_block(mode):
    model = between_model(mode, run_mode="joint").train()
    batch = tokens(16)
    loss, _ = joint_step(model, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.between.parameters())


def test_checkpoints_keep_the_between_settings(corpus, tmp_path):  # noqa: F811
    model = between_model("per_gap")
    path = tmp_path / "b.pt"
    save(path, model, make_optimizer(model, TCFG), 1, TCFG, corpus)
    loaded = load(path, torch.device("cpu"))[0]
    assert loaded.cfg.between_layers == 2 and loaded.cfg.between_mode == "per_gap"
    batch = tokens(2)
    assert torch.allclose(loaded.eval().infer(batch, k=3)[0], model.eval().infer(batch, k=3)[0], atol=1e-5)


def test_parameter_counts_for_both_modes():
    base = ModelConfig()
    block = sum(p.numel() for p in Spidory(replace(base, use_experts=False)).core[0].parameters())
    plain = Spidory(replace(base, use_experts=False)).num_params()
    shared = Spidory(replace(base, use_experts=False, between_layers=6)).num_params()
    per_gap = Spidory(replace(base, use_experts=False, between_layers=6, between_mode="per_gap")).num_params()
    assert shared - plain == 6 * block
    assert per_gap - plain == 18 * block


def test_compute_counts_the_between_runs():
    cfg = replace(CFG, between_layers=2)
    with_between, without = tflops_by_k(cfg), tflops_by_k(replace(cfg, between_layers=0))
    assert with_between[1] == pytest.approx(without[1])
    gap = with_between[2] - without[2]
    assert with_between[4] - without[4] == pytest.approx(3 * gap)
