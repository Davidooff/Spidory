from dataclasses import replace

import pytest
import torch

from spidory import rounds
from spidory.config import ModelConfig, RoundConfig, TrainConfig
from spidory.joint import joint_step
from spidory.model import Spidory, run_penalty
from spidory.train import load, make_optimizer, save, train_steps

from test_spidory import CFG, TCFG, corpus, tokens, wake_experts  # noqa: F401


def model_for(mode, neuron="net"):
    torch.manual_seed(0)
    model = Spidory(replace(CFG, run_mode="joint", neuron=neuron, neuron_train=mode)).train()
    wake_experts(model)
    return model


@pytest.mark.parametrize("mode", ["sampled", "average", "mix"])
def test_every_variant_trains_the_neuron_and_the_model(mode):
    model = model_for(mode)
    batch = tokens(16)
    loss, parts = joint_step(model, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert model.run_head[-1].weight.grad.abs().sum() > 0
    assert model.core[0].qkv.weight.grad.abs().sum() > 0
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert 1.0 <= parts["mean_runs"] <= CFG.max_runs
    assert ("neuron" in parts) == (mode != "mix")


def test_linear_and_network_neurons_are_separate():
    linear, net = model_for("sampled", "linear"), model_for("sampled", "net")
    batch = tokens(8)
    loss, _ = joint_step(linear, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert linear.run_neuron[-1].weight.grad.abs().sum() > 0
    assert linear.run_head[-1].weight.grad is None
    assert len([m for m in linear.run_neuron if isinstance(m, torch.nn.Linear)]) == 1
    loss, _ = joint_step(net, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert net.run_neuron[-1].weight.grad is None


def test_average_mode_trains_every_run_count():
    model = model_for("average")
    seen = []
    original = model._run_once
    model._run_once = lambda *a: (seen.append(torch.is_grad_enabled()), original(*a))[1]
    batch = tokens(4)
    joint_step(model, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    assert seen.count(True) == CFG.max_runs  # one gradient run for each count


def test_penalty_pushes_average_both_ways():
    gates = torch.tensor([[1.0, 0.0]] * 4)
    targets = torch.tensor([2.5, 2.5])
    for k in (0, 3):
        probs = torch.zeros(4, 4)
        probs[:, k] = 1
        assert run_penalty(probs, gates, targets) == pytest.approx(2.25)
    probs = torch.full((4, 4), 0.25, requires_grad=True)
    raised = torch.tensor([3.5, 2.5])
    run_penalty(probs, gates, raised).backward()
    assert probs.grad[0, 3] < probs.grad[0, 0]  # a higher target favours more runs


def test_neuron_objective_favours_the_run_count_with_lower_loss():
    from spidory.joint import neuron_objective
    logits = torch.zeros(3, 4, requires_grad=True)
    losses = torch.tensor([[2.0, 1.5, 1.6, 1.7], [1.0, 1.0, 1.0, 0.8], [3.0, 3.0, 3.0, 3.0]])
    optimizer = torch.optim.SGD([logits], lr=5.0)
    for _ in range(50):
        loss = neuron_objective(torch.softmax(logits, -1), losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    probs = torch.softmax(logits, -1)
    assert probs[0].argmax() == 1 and probs[1].argmax() == 3
    assert torch.allclose(probs[2], torch.full((4,), 0.25))  # no preference when losses tie


def test_groups_follow_activation_alone():
    skewed = {"a": torch.tensor([.6, .2, .1, .1]), "b": torch.tensor([.1, .2, .3, .4])}
    assert rounds.lift_group(skewed, "a", 1.5) == [0]
    even = {n: torch.full((4,), .25) for n in "ab"}
    assert rounds.lift_group(even, "a", 1.5) == []


def test_joint_rounds_raise_targets_by_activation(corpus, monkeypatch):  # noqa: F811
    model = model_for("sampled")
    monkeypatch.setattr(rounds, "lift_group", lambda *a: [1])
    monkeypatch.setattr(rounds, "keep_change", lambda *a: True)
    cfg = replace(TCFG, eval_windows=4, batch_size=2)
    rcfg = replace(RoundConfig(), passes=1, steps=1)
    rounds.run_rounds(model, make_optimizer(model, cfg), corpus, cfg, rcfg, 8, 0, False, lambda _: None)
    assert model.run_targets[1] > CFG.target_mean


def test_joint_training_runs_in_stage_three(corpus):  # noqa: F811
    model = model_for("mix")
    optimizer = make_optimizer(model, TCFG)
    before = model.run_head[-1].weight.clone()
    train_steps(model, optimizer, corpus, TCFG, 8, 11)
    assert not torch.equal(before, model.run_head[-1].weight)


def test_checkpoint_without_single_neuron_still_loads(corpus, tmp_path):  # noqa: F811
    model = Spidory(CFG)
    path = tmp_path / "old.pt"
    save(path, model, make_optimizer(model, TCFG), 1, TCFG, corpus)
    state = torch.load(path, weights_only=True)
    for key in [k for k in state["model"] if k.startswith("run_neuron.")]:
        del state["model"][key]
    torch.save(state, path)
    loaded = load(path, torch.device("cpu"))[0]
    assert loaded.run_neuron[-1].weight.shape[0] == CFG.n_experts * CFG.max_runs


def test_learned_inference_uses_the_chosen_neuron():
    model = model_for("sampled", "linear").eval()
    with torch.no_grad():
        model.run_neuron[-1].weight.zero_()
        model.run_neuron[-1].bias.zero_()
        model.run_neuron[-1].bias[2::CFG.max_runs] = 5.0  # three runs for every expert
    assert model.decide(tokens(6), learned=True)[1].eq(3).all()


def test_two_active_experts_share_each_sequence():
    model = Spidory(replace(CFG, active_experts=2)).eval()
    wake_experts(model)
    gates = model.route(model.router(model._encode(tokens(8))[1]), 1.0, False)
    assert torch.allclose(gates.sum(-1), torch.ones(8), atol=1e-5)
    assert (gates > 0).sum(-1).eq(2).all()


def test_two_active_experts_train_and_group_correctly():
    model = Spidory(replace(CFG, active_experts=2, run_mode="joint")).train()
    wake_experts(model)
    batch = tokens(16)
    loss, _ = joint_step(model, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert model.router[-1].weight.grad.abs().sum() > 0
    model.eval()
    grouped, experts, ks = model.infer(batch, k=2)
    for i in range(4):
        assert torch.allclose(grouped[i], model.infer(batch[i:i + 1], k=2)[0][0], atol=1e-4)


def test_one_active_expert_is_unchanged():
    torch.manual_seed(0)
    model = Spidory(CFG).eval()
    wake_experts(model)
    gates = model.route(model.router(model._encode(tokens(6))[1]), 1.0, False)
    assert gates.eq(0).logical_or(gates.eq(1)).all() and gates.sum(-1).eq(1).all()
