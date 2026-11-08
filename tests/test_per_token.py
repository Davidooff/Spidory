"""Per-token routing: each token picks its own expert and run count from its own state."""
from dataclasses import replace

import pytest
import torch

from spidory import rounds
from spidory.evaluate import decisions, sequence_losses
from spidory.flops import tflops_by_k
from spidory.joint import joint_step
from spidory.model import Spidory
from spidory.train import make_optimizer, sample_ks, train_steps
from spidory.ui import server

from test_spidory import CFG, TCFG, corpus, tokens, wake_experts  # noqa: F401

TOKEN = replace(CFG, route_per_token=True, run_mode="joint")


def woken(cfg=TOKEN):
    model = Spidory(cfg).eval()
    wake_experts(model)
    model.run_head_ready.fill_(True)
    return model


def test_routing_is_per_token_and_sees_past_the_prefix():
    """The old router read only the first `prefix` tokens. Now a later token's choice depends on it."""
    model = woken()
    batch = tokens(2)
    changed = batch.clone()
    changed[:, -10:] = (changed[:, -10:] + 7) % CFG.vocab_size
    a, b = model(batch, ks=2), model(changed, ks=2)
    assert a["router_logits"].shape == (2, CFG.seq_len, CFG.n_experts)
    assert torch.allclose(a["router_logits"][:, :-10], b["router_logits"][:, :-10], atol=1e-5)  # causal
    assert not torch.allclose(a["router_logits"][:, -10:], b["router_logits"][:, -10:])


def test_tokens_can_use_different_experts():
    model = woken()
    experts, ks = model.decide(tokens(4), learned=True)
    assert experts.shape == ks.shape == (4, CFG.seq_len)
    assert len(experts.unique()) > 1
    assert ks.min() >= 1 and ks.max() <= CFG.max_runs


def test_inference_uses_each_tokens_own_choices():
    model = woken()
    batch = tokens(3)
    logits, experts, ks = model.infer(batch, learned=True)
    gates = torch.nn.functional.one_hot(experts, CFG.n_experts).float()
    expected = model(batch, ks=ks, gate_override=gates)["logits"]
    assert torch.allclose(logits, expected, atol=1e-5)
    alone = torch.cat([model.infer(batch[i:i + 1], learned=True)[0] for i in range(3)])
    assert torch.allclose(logits, alone, atol=1e-5)


def test_fixed_runs_average_and_expert_lock():
    model = woken()
    batch = tokens(2)
    _, _, ks = model.infer(batch, k=3)
    assert (ks == 3).all()
    _, _, ks = model.infer(batch, mean_runs=2.5)
    assert set(ks.unique().tolist()) <= {2, 3} and abs(ks.float().mean().item() - 2.5) < 0.25
    _, experts, _ = model.infer(batch, k=2, expert=1)
    assert (experts == 1).all()


@pytest.mark.parametrize("mode", ["sampled", "average"])
def test_joint_step_per_token_trains_router_neuron_and_experts(mode):
    model = Spidory(replace(TOKEN, neuron_train=mode))
    wake_experts(model)
    batch = tokens(4)
    loss, parts = joint_step(model, batch[:, :-1], batch[:, 1:], TCFG, tau=1.0)
    loss.backward()
    assert torch.isfinite(loss) and 1 <= parts["mean_runs"] <= CFG.max_runs
    assert model.run_head[-1].weight.grad.abs().sum() > 0
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert any(e.up.weight.grad is not None and e.up.weight.grad.abs().sum() > 0 for e in model.experts)


def test_mix_is_refused_per_token():
    with pytest.raises(ValueError):
        Spidory(replace(TOKEN, neuron_train="mix"))
    with pytest.raises(ValueError):
        Spidory(replace(TOKEN, use_experts=False))


def test_neuron_moves_toward_lower_loss_runs_per_token():
    """When one run count is cheaper in loss for every token, the neuron raises its probability."""
    model = Spidory(TOKEN)
    wake_experts(model)
    from spidory.joint import neuron_objective
    probs = torch.full((10, CFG.max_runs), 0.25, requires_grad=True)
    losses = torch.tensor([[2.0, 1.0, 2.0, 2.0]]).expand(10, -1)
    neuron_objective(probs, losses).backward()
    assert probs.grad[:, 1].mean() < probs.grad[:, 0].mean()


def test_table_mode_stage_three_gives_per_token_runs():
    model = Spidory(replace(TOKEN, run_mode="table"))
    experts = torch.zeros(4, CFG.seq_len, dtype=torch.long)
    ks = sample_ks(TCFG.stage1_steps + TCFG.stage2_steps + 1, TCFG, model, experts, CFG.seq_len)
    assert ks.shape == (4, CFG.seq_len)
    early = sample_ks(TCFG.stage1_steps + 1, TCFG, model, experts, CFG.seq_len)
    assert early.shape == (4,)  # stage 2 keeps one count per sequence


def test_trains_every_stage_and_is_evaluated(corpus):  # noqa: F811
    cfg = replace(TCFG, stage1_steps=1, stage2_steps=1, cycle_steps=2, steps=8, eval_windows=4)
    for run_mode in ("table", "joint"):
        model = Spidory(replace(TOKEN, run_mode=run_mode, run_signal=True))
        train_steps(model, make_optimizer(model, cfg), corpus, cfg, 0, 8)
        model.run_head_ready.fill_(True)
        found = decisions(model, corpus, 4, learned=run_mode == "joint")
        assert sum(sum(v) for v in found["activation"].values()) == 3 * 4 * CFG.seq_len  # every token of every window
        assert set(sequence_losses(model, corpus, "test", 4, mean_runs=2.5)) == {"story", "maths", "qa"}
        shares = rounds.measure_activation(model, corpus, "val", 4)
        assert all(abs(float(s.sum()) - 1.0) < 1e-5 for s in shares.values())


def test_compute_counts_every_expert_per_token():
    single, per_token = tflops_by_k(CFG), tflops_by_k(TOKEN)
    assert all(per_token[k] > single[k] for k in single)


def test_playground_shows_the_last_tokens_choice(tmp_path, corpus):  # noqa: F811
    from spidory.train import save
    model = woken()
    save(tmp_path / "m" / "last.pt", model, make_optimizer(model, TCFG), 1, TCFG, corpus)
    registry = server.Registry(tmp_path, [{"id": "m", "name": "m", "note": "", "path": "m/last.pt"}])
    from test_ui import FakeTok
    events = []
    request = {"prompt": "hello", "max_tokens": 3,
               "columns": [{"model": "m", "setting": {"kind": s}} for s in ("neuron", "frozen", "cycle")]}
    request["columns"][1]["setting"]["value"] = 2
    request["columns"][2]["setting"]["value"] = 4
    server.run_columns(registry, FakeTok(), request, events.append)
    assert not [e for e in events if "error" in e]
    made = [e for e in events if "text" in e]
    assert made and all(0 <= e["expert"] < CFG.n_experts and len(e["path"]) == e["runs"] for e in made)
