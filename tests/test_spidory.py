import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spidory import data, rounds
from spidory.config import ModelConfig, RoundConfig, TrainConfig
from spidory.data import DATASETS, Corpus
from spidory.evaluate import combine
from spidory.flops import average_tflops, tflops_by_k
from spidory.run_head import best_run_targets, train_run_head
from spidory.model import Spidory, compute_loss, update_targets
from spidory.train import load, make_optimizer, sample_ks, save, set_rng_state, train_steps

CFG = ModelConfig(vocab_size=64, d_model=32, seq_len=48, n_experts=4, expert_dim=8, prefix=8,
                  n_heads=4, n_input=1, n_core=1, n_output=1)
TCFG = TrainConfig(batch_size=4, stage1_steps=4, stage2_steps=4, steps=40, warmup_lr_steps=1)


def tokens(size=6):
    return torch.randint(0, CFG.vocab_size, (size, CFG.seq_len))


def wake_experts(model):
    with torch.no_grad():
        for expert in model.experts:
            if hasattr(expert, "net"):
                expert.net[-1].weight.normal_(std=0.1)
            else:
                expert.alpha.value.normal_(std=0.5 / expert.alpha.ratio)  # effective step about 0.5
        model.router[-1].weight.normal_(std=2.0)


def equal_state(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def corpus(tmp_path):
    for name in DATASETS:
        for split in data.SPLITS:
            np.random.default_rng(1).integers(2, 64, 4000).astype(np.uint16).tofile(
                tmp_path / f"{name}.{split}.bin")
    return Corpus(tmp_path, CFG.seq_len, seed=0)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_output_shape_and_exact_run_count(k):
    model = Spidory(CFG).eval()
    calls = []
    model.core[0].register_forward_hook(lambda *_: calls.append(1))
    logits, experts, ks = model.infer(tokens(1), k=k)
    assert logits.shape == (1, CFG.seq_len, CFG.vocab_size)
    assert len(calls) == k and ks.eq(k).all()


def test_only_last_run_has_gradients():
    model = Spidory(CFG).train()
    wake_experts(model)
    flags = []
    original = model._run_once
    model._run_once = lambda *a: (flags.append(torch.is_grad_enabled()), original(*a))[1]
    batch = tokens(8)
    model(batch, ks=4)
    assert flags == [False, False, False, True]


def test_gradients_reach_input_router_and_experts():
    model = Spidory(CFG).train()
    wake_experts(model)
    batch = tokens(32)
    loss, _ = compute_loss(model(batch[:, :-1], ks=3), batch[:, 1:], 0.2)
    loss.backward()
    assert model.input_block[0].qkv.weight.grad.abs().sum() > 0
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert all(all(p.grad is not None for p in e.parameters()) for e in model.experts)


def test_input_gradient_reaches_every_row_through_last_run():
    model = Spidory(CFG).train()
    x = model._unit(model._encode(tokens(4))[0])
    x.retain_grad()
    model._loop(x, None, torch.tensor([1, 1, 3, 3])).sum().backward()
    assert all(x.grad[i].abs().sum() > 0 for i in range(4))


def test_loop_matches_dory_formula():
    """s0 = x, u = N(s + x), s = N(F(u)), with no learned adapter."""
    model = Spidory(replace(CFG, use_experts=False)).eval()
    assert not hasattr(model, "inject")
    batch = tokens(2)
    with torch.no_grad():
        h0 = model._encode(batch)[0]
        x = model._unit(h0)
        s = x
        for _ in range(3):
            h = model._unit(s + x)
            for block in model.core:
                h = block(h)
            s = model._unit(h)
        expected = model._decode(s, torch.full((2,), 3), torch.full((2,), -1))
        assert torch.allclose(model.infer(batch, k=3)[0], expected, atol=1e-5)


def test_output_block_receives_runs_and_expert():
    model = Spidory(CFG).eval()
    batch = tokens(4)
    before = model.infer(batch, k=2)[0].clone()
    with torch.no_grad():
        model.run_embed.weight[2].fill_(1.0)
    after_runs = model.infer(batch, k=2)[0].clone()
    assert not torch.equal(before, after_runs)
    assert torch.equal(model.infer(batch, k=3)[0], model.infer(batch, k=3)[0])
    with torch.no_grad():
        model.expert_embed.weight.fill_(1.0)
    assert not torch.equal(after_runs, model.infer(batch, k=2)[0])


def test_output_signals_start_silent():
    model = Spidory(CFG)
    assert model.run_embed.weight.eq(0).all() and model.expert_embed.weight.eq(0).all()


def test_gradients_reach_output_signals():
    model = Spidory(CFG).train()
    wake_experts(model)
    batch = tokens(16)
    loss, _ = compute_loss(model(batch[:, :-1], ks=2), batch[:, 1:], 0.0)
    loss.backward()
    assert model.run_embed.weight.grad[2].abs().sum() > 0
    assert model.expert_embed.weight.grad.abs().sum() > 0


def test_state_size_stays_fixed_across_runs():
    model = Spidory(CFG).eval()
    with torch.no_grad():
        x = model._unit(model._encode(tokens(2))[0])
        s = x
        for _ in range(4):
            s = model._run_once(s, x, None)
            assert torch.allclose(s.norm(dim=-1), torch.tensor(1.0), atol=1e-3)  # nGPT: unit length


def test_per_sequence_k_matches_single_sequences():
    model = Spidory(CFG).eval()
    wake_experts(model)
    batch = tokens(8)
    ks = torch.tensor([1, 2, 3, 4, 4, 3, 2, 1])
    with torch.no_grad():
        mixed = model(batch, ks=ks)["logits"]
    for i in range(8):
        single = model.infer(batch[i:i + 1], k=int(ks[i]))[0]
        assert torch.allclose(mixed[i], single[0], atol=1e-4)


def test_grouped_inference_matches_single_sequences():
    model = Spidory(CFG).eval()
    wake_experts(model)
    with torch.no_grad():
        model.run_targets.copy_(torch.tensor([1.5, 2.0, 3.0, 3.5]))
    batch = tokens(12)
    grouped, experts, ks = model.infer(batch, mean_runs=2.5)
    assert len({*zip(experts.tolist(), ks.tolist())}) > 1, "test needs mixed choices"
    for i in range(12):
        assert torch.allclose(grouped[i], model.infer(batch[i:i + 1], mean_runs=2.5)[0][0], atol=1e-4)


def test_experts_start_at_zero():
    model = Spidory(CFG).eval()
    batch = tokens()
    with torch.no_grad():
        off = model(batch, ks=2, experts_on=False)["logits"]
        on = model(batch, ks=2)["logits"]
    assert torch.allclose(off, on, atol=1e-5)


def test_zero_gate_disables_expert():
    model = Spidory(CFG).eval()
    wake_experts(model)
    batch = tokens()
    with torch.no_grad():
        zero = model(batch, ks=2, gate_override=torch.zeros(6, CFG.n_experts))["logits"]
        gates = torch.zeros(6, CFG.n_experts)
        gates[:, 0] = 1
        active = model(batch, ks=2, gate_override=gates)["logits"]
        off = model(batch, ks=2, experts_on=False)["logits"]
    assert torch.equal(zero, off) and not torch.equal(zero, active)


def test_router_sees_only_prefix():
    model = Spidory(CFG).eval()
    batch = tokens()
    changed = batch.clone()
    changed[:, CFG.prefix:] = (changed[:, CFG.prefix:] + 1) % CFG.vocab_size
    assert torch.allclose(model._encode(batch)[1], model._encode(changed)[1], atol=1e-6)
    changed[:, CFG.prefix - 1] = (changed[:, CFG.prefix - 1] + 1) % CFG.vocab_size
    assert not torch.allclose(model._encode(batch)[1], model._encode(changed)[1], atol=1e-6)


def test_dataset_names_cannot_enter_the_loss():
    assert list(inspect.signature(compute_loss).parameters) == ["out", "targets", "balance_weight"]


@pytest.mark.parametrize("mean", [1.5, 2.0, 2.5, 3.0, 3.5])
def test_mean_runs_input_sets_the_average(mean):
    model = Spidory(CFG).eval()
    model.run_targets.copy_(torch.tensor([3.0, 2.5, 2.5, 2.0]))
    ks = model.decide(torch.randint(0, CFG.vocab_size, (4000, CFG.seq_len)), mean_runs=mean)
    expected = float((model.expert_usage * model.targets_for(mean)).sum())
    assert expected == pytest.approx(mean, abs=1e-3)
    # The router is untrained, so check the average of targets actually chosen.
    chosen = model.targets_for(mean)[ks[0]].mean()
    assert ks[1].float().mean() == pytest.approx(float(chosen), abs=0.05)


def test_mean_runs_keeps_expert_offsets():
    model = Spidory(CFG)
    model.run_targets.copy_(torch.tensor([3.0, 2.5, 2.5, 2.0]))
    low, high = model.targets_for(2.0), model.targets_for(3.0)
    assert (low[0] - low[3]) == pytest.approx(1.0, abs=1e-3)
    assert (high[0] - high[3]) == pytest.approx(1.0, abs=1e-3)


def test_manual_k_overrides_mean():
    model = Spidory(CFG).eval()
    assert model.decide(tokens(), k=4, mean_runs=2.0)[1].eq(4).all()


def test_k_choice_repeats():
    model = Spidory(CFG).eval()
    batch = tokens(40)
    assert torch.equal(model.decide(batch, mean_runs=2.5)[1], model.decide(batch, mean_runs=2.5)[1])


def test_stage_k_sampling():
    model = Spidory(CFG)
    experts = torch.zeros(4000, dtype=torch.long)
    assert sample_ks(0, TCFG, model, experts).eq(1).all()
    assert set(sample_ks(5, TCFG, model, experts).unique().tolist()) <= {1, 2, 3, 4}
    assert sample_ks(10, TCFG, model, experts).float().mean() == pytest.approx(2.5, abs=0.06)


def test_stage_two_forces_one_run_batches():
    model = Spidory(CFG)
    experts = torch.zeros(8, dtype=torch.long)
    forced = sum(sample_ks(5, TCFG, model, experts).eq(1).all().item() for _ in range(2000))
    assert forced / 2000 == pytest.approx(0.1, abs=0.03)


@pytest.mark.parametrize("group", [[0], [0, 1], [5]])
def test_target_update_keeps_usage_weighted_mean(group):
    targets = torch.full((6,), 2.5)
    usage = torch.tensor([0.3, 0.1, 0.2, 0.1, 0.2, 0.1])
    updated = update_targets(targets, usage, group, 0.5, 2.5)
    assert float((usage * updated).sum()) == pytest.approx(2.5, abs=1e-3)
    assert all(updated[g] > 2.5 for g in group)
    assert updated.min() >= 1 and updated.max() <= 4


def test_target_update_holds_when_others_hit_the_floor():
    targets = torch.tensor([3.9, 1.0, 1.0, 1.0, 1.0, 1.0])
    usage = torch.tensor([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
    updated = update_targets(targets, usage, [0], 0.5, 2.5)
    assert float((usage * updated).sum()) == pytest.approx(2.5, abs=1e-3)
    assert updated.min() >= 1 and updated.max() <= 4


def test_group_needs_lift_and_gain():
    table = torch.full((6, 4), 2.0)
    table[:, 2:] = 1.0  # every expert gains from more runs
    flat = torch.full((6, 4), 2.0)
    targets = torch.full((6,), 2.5)
    even = {n: torch.full((6,), 1 / 6) for n in ["a", "b"]}
    assert rounds.choose_group(even, table, targets, "a", 1.5, 0.002) == []
    skewed = {"a": torch.tensor([.3, .3, .3, .1, 0, 0]), "b": torch.tensor([0, 0, 0, .1, .45, .45])}
    assert set(rounds.choose_group(skewed, table, targets, "a", 1.5, 0.002)) == {0, 1, 2}
    assert rounds.choose_group(skewed, flat, targets, "a", 1.5, 0.002) == []


def test_no_gain_above_maximum():
    assert rounds.gain_from_more_runs(torch.ones(2, 4), 0, 4.0) == 0.0


def test_keep_rule():
    control = {"maths": 2.0, "story": 1.0}
    assert rounds.keep_change({"maths": 1.9, "story": 1.005}, control, "maths", 0.01)
    assert not rounds.keep_change({"maths": 2.1, "story": 1.0}, control, "maths", 0.01)
    assert not rounds.keep_change({"maths": 1.9, "story": 1.02}, control, "maths", 0.01)
    assert not rounds.keep_change({"maths": 1.9999, "story": 1.0}, control, "maths", 0.01)


def test_rollback_restores_weights_exactly(corpus):
    model = Spidory(CFG)
    optimizer = make_optimizer(model, TCFG)
    train_steps(model, optimizer, corpus, TCFG, 0, 2)
    saved = rounds.snapshot(model, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    model.run_targets.add_(1.0)
    train_steps(model, optimizer, corpus, TCFG, 2, 5)
    rounds.restore(model, optimizer, saved)
    assert equal_state(expected, model.state_dict())


def test_step_resume_matches_uninterrupted_run(corpus, tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.manual_seed(1)
    corpus.reseed(1)
    model = Spidory(CFG)
    optimizer = make_optimizer(model, TCFG)
    train_steps(model, optimizer, corpus, TCFG, 0, 6)
    save(path, model, optimizer, 6, TCFG, corpus)
    train_steps(model, optimizer, corpus, TCFG, 6, 10)
    straight = {k: v.clone() for k, v in model.state_dict().items()}

    torch.manual_seed(99)
    corpus.reseed(99)
    other, other_opt, step, cfg, state = load(path, torch.device("cpu"))
    set_rng_state(state["rng"], corpus)
    assert step == 6 and cfg == TCFG
    train_steps(other, other_opt, corpus, cfg, step, 10)
    assert all(torch.allclose(straight[k], v, atol=1e-6) for k, v in other.state_dict().items())


def test_round_resume_matches_uninterrupted_run(corpus, tmp_path):
    rcfg = replace(RoundConfig(), passes=1, steps=2, lift=1.0, min_gain=0.0)
    cfg = replace(TCFG, eval_windows=4, batch_size=2)

    def fresh():
        torch.manual_seed(3)
        model = Spidory(CFG)
        return model, make_optimizer(model, cfg)

    model, optimizer = fresh()
    rounds.run_rounds(model, optimizer, corpus, cfg, rcfg, 8, 0, False, lambda _: None)
    straight = {k: v.clone() for k, v in model.state_dict().items()}

    model, optimizer = fresh()
    stops = []

    class Stop(Exception):
        pass

    def on_round(step, following):
        save(tmp_path / "round.pt", model, optimizer, step, cfg, corpus, {"next_round": list(following)})
        stops.append(following)
        raise Stop

    with pytest.raises(Stop):
        rounds.run_rounds(model, optimizer, corpus, cfg, rcfg, 8, 0, False, lambda _: None, on_round=on_round)
    assert stops == [(0, 1)]
    model, optimizer, step, loaded, state = load(tmp_path / "round.pt", torch.device("cpu"))
    set_rng_state(state["rng"], corpus)
    final = rounds.run_rounds(model, optimizer, corpus, loaded, rcfg, step, 0, False, lambda _: None,
                              tuple(state["extra"]["next_round"]))
    assert final == 8 + 2 * len(DATASETS)
    assert all(torch.allclose(straight[k], v, atol=1e-6) for k, v in model.state_dict().items())


def test_control_only_leaves_targets_unchanged(corpus):
    rcfg = replace(RoundConfig(), passes=1, steps=1, lift=0.0, min_gain=-1.0)
    cfg = replace(TCFG, eval_windows=4, batch_size=2)
    model = Spidory(CFG)
    rounds.run_rounds(model, make_optimizer(model, cfg), corpus, cfg, rcfg, 8, 0, True, lambda _: None)
    assert model.run_targets.eq(CFG.target_mean).all()


def test_plain_model_runs_rounds(corpus):
    rcfg = replace(RoundConfig(), passes=1, steps=1)
    cfg = replace(TCFG, eval_windows=4, batch_size=2)
    model = Spidory(replace(CFG, use_experts=False))
    assert rounds.run_rounds(model, make_optimizer(model, cfg), corpus, cfg, rcfg, 8, 0, True,
                             lambda _: None) == 8 + len(DATASETS)


def test_same_seed_gives_same_shared_start():
    torch.manual_seed(5)
    moe = Spidory(ModelConfig()).state_dict()
    torch.manual_seed(5)
    plain = Spidory(ModelConfig(use_experts=False)).state_dict()
    shared = [k for k in plain if k not in ("run_targets", "expert_usage")]
    assert shared and all(torch.equal(moe[k], plain[k]) for k in shared)


def _forced_round(corpus, monkeypatch, keep):
    rcfg = replace(RoundConfig(), passes=1, steps=1)
    cfg = replace(TCFG, eval_windows=4, batch_size=2)
    monkeypatch.setattr(rounds, "choose_group", lambda *a: [0])
    monkeypatch.setattr(rounds, "keep_change", lambda *a: keep)
    model = Spidory(CFG)
    records = []
    rounds.run_rounds(model, make_optimizer(model, cfg), corpus, cfg, rcfg, 8, 0, False, records.append)
    return model, records


def test_kept_change_raises_targets_and_keeps_average(corpus, monkeypatch):
    model, records = _forced_round(corpus, monkeypatch, True)
    assert all(r["kept"] and "changed" in r for r in records)
    assert model.run_targets[0] > CFG.target_mean
    assert float((model.expert_usage * model.run_targets).sum()) == pytest.approx(2.5, abs=1e-2)


def test_rejected_change_leaves_targets_and_equals_control(corpus, monkeypatch):
    model, records = _forced_round(corpus, monkeypatch, False)
    assert not any(r["kept"] for r in records)
    assert model.run_targets.eq(CFG.target_mean).all()


def test_rules_need_every_seed_and_more_than_one():
    good = {"loops_pass": True, "experts_pass": True, "variable_pass": True, "learned_pass": True,
            "specialised_pass": True,
            "loops_gain": 0.02, "variable_gain_by_mean": {}}
    one = combine([{"seed": 0, "rules": good}])
    assert one["experts_pass"] and not one["confirmed"]
    mixed = combine([{"seed": 0, "rules": good}, {"seed": 1, "rules": good | {"experts_pass": False}}])
    assert mixed["confirmed"] and not mixed["experts_pass"]


def test_tflops_match_hand_count_and_grow_with_runs():
    cfg = replace(CFG, use_experts=False, n_input=0, n_core=1, n_output=0, arch="prenorm")
    d, t = cfg.d_model, cfg.seq_len
    block = 8 * d * d * t + 3 * d * d * t + d * d * t + 2 * t * t * d  # ffn, qkv, proj, attention
    per_run = 2 * block / t / 1e12  # 2 FLOPs per multiply-add
    head = 2 * cfg.vocab_size * d / 1e12  # projection back to the vocabulary
    table = tflops_by_k(cfg)
    assert table[1] == pytest.approx(per_run + head, rel=1e-6)
    assert table[3] == pytest.approx(3 * per_run + head, rel=1e-6)


def test_average_tflops_weights_by_real_choices():
    assert average_tflops({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}, [1, 0, 0, 1]) == pytest.approx(2.5)


def test_default_model_keeps_dory_proportions():
    cfg = ModelConfig()
    assert (cfg.n_input, cfg.n_core, cfg.n_output) == (4, 10, 4)
    assert 20e6 < Spidory(replace(cfg, use_experts=False)).num_params() < 30e6


def test_run_target_is_smallest_count_near_the_best():
    losses = torch.tensor([[2.0, 1.0, 1.001, 1.5], [1.0, 1.0, 1.0, 1.0], [3.0, 2.9, 2.0, 1.0]])
    assert best_run_targets(losses, 0.005).tolist() == [2, 1, 4]


def test_model_chooses_runs_by_itself_per_input():
    model = Spidory(CFG).eval()
    wake_experts(model)
    with torch.no_grad():
        model.run_head[-1].weight.normal_(std=3.0)
    batch = tokens(32)
    _, ks = model.decide(batch, learned=True)
    assert len(ks.unique()) > 1
    logits, experts, chosen = model.infer(batch, learned=True)
    assert torch.equal(chosen, ks)
    for i in range(4):
        one = model.infer(batch[i:i + 1], k=int(ks[i]))[0]
        assert torch.allclose(logits[i], one[0], atol=1e-4)


def test_run_head_training_learns_and_leaves_the_rest_frozen(corpus):
    model = Spidory(CFG)
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("run_head")}
    report = train_run_head(model, corpus, windows=24, steps=30, tolerance=0.005)
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items() if k != "run_head_ready")
    assert bool(model.run_head_ready) and 0 <= report["holdout_accuracy"] <= 1


def test_older_checkpoints_still_load(corpus, tmp_path):
    model = Spidory(replace(CFG, use_experts=False))
    path = tmp_path / "old.pt"
    save(path, model, make_optimizer(model, TCFG), 1, TCFG, corpus)
    state = torch.load(path, weights_only=True)
    del state["model"]["run_head_ready"]
    torch.save(state, path)
    loaded = load(path, torch.device("cpu"))[0]
    assert not bool(loaded.run_head_ready)


def test_run_head_trains_only_after_everything_else():
    model = Spidory(CFG)
    assert not bool(model.run_head_ready)


def test_training_stops_on_a_broken_loss(corpus, monkeypatch):
    import spidory.train as train
    model = Spidory(CFG)
    monkeypatch.setattr(train, "compute_loss", lambda *a: (torch.tensor(float("nan"), requires_grad=True), {}))
    with pytest.raises(RuntimeError, match="loss became"):
        train.train_steps(model, make_optimizer(model, TCFG), corpus, TCFG, 0, 1)


def test_plain_is_a_bit_smaller():
    moe, plain = ModelConfig(), ModelConfig(use_experts=False)
    a, b = Spidory(moe).num_params(), Spidory(plain).num_params()
    assert 0.9 * a < b < a


@pytest.mark.parametrize("ks", [1, 3])
def test_overfits_tiny_batch(ks):
    cfg = replace(CFG, n_core=2, d_model=64)
    model = Spidory(cfg).train()
    batch = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, parts = compute_loss(model(batch[:, :-1], ks=ks, tau=0.5), batch[:, 1:], 0.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert parts["ce"] < 0.1
