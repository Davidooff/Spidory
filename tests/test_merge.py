from dataclasses import replace

import pytest
import torch

from spidory.joint import joint_step
from spidory.model import Spidory
from spidory.train import cycle_ks

from test_spidory import CFG, TCFG, tokens, wake_experts  # noqa: F401


def merge_model(norm=True, **extra):
    torch.manual_seed(0)
    model = Spidory(replace(CFG, between_layers=2, between_mode="per_gap", between_merge=True,
                            boundary_norm=norm, **extra))
    wake_experts(model)
    return model


def test_merge_needs_a_between_block():
    with pytest.raises(ValueError):
        Spidory(replace(CFG, between_merge=True))


def test_merge_starts_as_plain_sum_of_state_and_input():
    model = merge_model()
    s, x = torch.randn(2, 5, CFG.d_model), torch.randn(2, 5, CFG.d_model)
    for gap in range(CFG.max_runs - 1):
        merged = model.merge_state[gap](s) + model.merge_input[gap](x)
        assert torch.allclose(merged, s + x, atol=1e-5)


def test_centre_does_not_add_the_input_again_in_merge_mode():
    model = merge_model().eval()
    s = model._unit(torch.randn(2, CFG.seq_len, CFG.d_model))
    with torch.no_grad():
        a = model._run_once(s, torch.zeros_like(s), None)
        b = model._run_once(s, torch.randn_like(s), None)
    assert torch.allclose(a, b)


@pytest.mark.parametrize("norm", [True, False])
def test_merge_learns_and_input_gets_gradient_through_it(norm):
    model = merge_model(norm).train()
    batch = tokens(4)
    model(batch[:, :-1], ks=3)["logits"].sum().backward()
    assert model.merge_state[1].weight.grad.abs().sum() > 0  # gap 2, before the last run
    assert model.merge_input[1].weight.grad.abs().sum() > 0
    assert model.merge_state[0].weight.grad is None  # gap 1 had no gradients (TBPTL)
    assert model.input_block[0].qkv.weight.grad.abs().sum() > 0


def test_no_boundary_norm_removes_the_loop_normalization():
    model = merge_model(norm=False).eval()
    calls = []
    original = model._unit
    model._unit = lambda h: (calls.append(1), original(h))[1]
    s = original(torch.randn(1, CFG.seq_len, CFG.d_model))
    with torch.no_grad():
        model._run_once(s, s, None)
        model._between(s, 1, s)
    assert calls == []
    merged = model.merge_state[0](s) + model.merge_input[0](s)
    assert not torch.allclose(merged.norm(dim=-1), torch.ones(1, CFG.seq_len), atol=1e-2)


@pytest.mark.parametrize("norm", [True, False])
def test_every_path_works_in_both_versions(norm):
    model = merge_model(norm).eval()
    batch = tokens(6)
    ks = torch.tensor([1, 2, 3, 4, 2, 3])
    with torch.no_grad():
        mixed = model(batch, ks=ks)["logits"]
        per_token = model(batch, ks=torch.full(batch.shape, 3))["logits"]
        per_sequence = model(batch, ks=3)["logits"]
    for i in range(6):
        assert torch.allclose(mixed[i], model.infer(batch[i:i + 1], k=int(ks[i]))[0][0], atol=1e-4)
    assert torch.allclose(per_token, per_sequence, atol=1e-4)

    model.train()
    model(batch, ks=cycle_ks(6, CFG.seq_len, 4, "cpu"))["logits"].sum().backward()
    assert all(layer.weight.grad is not None for layer in model.merge_input)

    joint = merge_model(norm, run_mode="joint").train()
    loss, _ = joint_step(joint, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert any(layer.weight.grad is not None for layer in joint.merge_input)
