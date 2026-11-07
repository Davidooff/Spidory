from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from spidory.config import ModelConfig
from spidory.model import Spidory
from spidory.ngpt import NBlock, Scale, rope, swiglu_width
from spidory.train import load, make_optimizer, save, train_steps

from test_spidory import CFG, TCFG, corpus, tokens  # noqa: F401


def test_default_is_ngpt_without_position_table():
    model = Spidory(CFG)
    assert model.ngpt and not hasattr(model, "pos") and not hasattr(model, "norm")


def test_states_have_unit_length_after_every_layer():
    model = Spidory(CFG).eval()
    h = model.tok(tokens(3))
    assert torch.allclose(h.norm(dim=-1), torch.ones(3, CFG.seq_len), atol=1e-4)
    for block in [*model.input_block, *model.core, *model.output_block]:
        h = block(h)
        assert torch.allclose(h.norm(dim=-1), torch.ones(3, CFG.seq_len), atol=1e-4)


def test_weights_return_to_unit_rows_after_each_step(corpus):  # noqa: F811
    model = Spidory(replace(CFG, use_experts=False))
    train_steps(model, make_optimizer(model, TCFG), corpus, TCFG, 0, 2)
    block = model.core[0]
    assert torch.allclose(block.qkv.weight.norm(dim=1), torch.ones(3 * CFG.d_model), atol=1e-4)
    assert torch.allclose(block.proj.weight.norm(dim=0), torch.ones(CFG.d_model), atol=1e-4)
    assert torch.allclose(block.down.weight.norm(dim=0), torch.ones(swiglu_width(CFG.d_model)), atol=1e-4)
    assert torch.allclose(model.tok.weight.norm(dim=1), torch.ones(CFG.vocab_size), atol=1e-4)


def test_no_weight_decay_for_ngpt():
    assert make_optimizer(Spidory(CFG), TCFG).param_groups[0]["weight_decay"] == 0.0
    old = Spidory(replace(CFG, arch="prenorm"))
    assert make_optimizer(old, TCFG).param_groups[0]["weight_decay"] == TCFG.weight_decay


def test_rope_keeps_length_and_depends_only_on_distance():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    seq = lambda v: v.expand(1, 1, 12, 16).clone()
    rq, rk = rope(seq(q)), rope(seq(k))
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1).expand(1, 1, 12), atol=1e-5)
    score = lambda i, j: float((rq[0, 0, i] * rk[0, 0, j]).sum())
    assert score(5, 2) == pytest.approx(score(9, 6), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)


def test_queries_and_keys_are_cosines():
    block = NBlock(32, 4).eval()
    captured = {}
    original = F.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        captured.update(q=q, k=k, scale=kw.get("scale"))
        return original(q, k, v, **kw)

    F.scaled_dot_product_attention, saved = spy, F.scaled_dot_product_attention
    try:
        block(F.normalize(torch.randn(2, 6, 32), dim=-1))
    finally:
        F.scaled_dot_product_attention = saved
    s_qk = block.s_qk().view(4, 8)
    unit_q = captured["q"] / s_qk[None, :, None, :]
    assert torch.allclose(unit_q.norm(dim=-1), torch.ones(2, 4, 6), atol=1e-4)
    assert captured["scale"] == pytest.approx(8 ** 0.5)


def test_scale_parametrization():
    s = Scale(4, init=0.05, scale=0.1)
    assert torch.allclose(s(), torch.full((4,), 0.05)) and torch.allclose(s.value, torch.full((4,), 0.1))


def test_old_checkpoints_load_as_the_old_architecture(corpus, tmp_path):  # noqa: F811
    old = Spidory(replace(CFG, arch="prenorm"))
    path = tmp_path / "old.pt"
    save(path, old, make_optimizer(old, TCFG), 1, TCFG, corpus)
    state = torch.load(path, weights_only=True)
    del state["model_cfg"]["arch"]
    torch.save(state, path)
    loaded = load(path, torch.device("cpu"))[0]
    assert not loaded.ngpt and hasattr(loaded, "pos")
    batch = tokens(2)
    assert torch.allclose(loaded.infer(batch, k=2)[0], old.eval().infer(batch, k=2)[0], atol=1e-5)


def test_ngpt_learns_a_tiny_batch():
    torch.manual_seed(0)
    cfg = replace(CFG, use_experts=False, n_core=2, d_model=64)
    model = Spidory(cfg).train()
    batch = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(300):
        logits = model(batch[:, :-1], ks=2)["logits"]
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_weights()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.2 * first
