"""
Smoke test: verify QuadTreeTensor, QuadTreeLinear, QTBlock, PassDelta,
and RecurrentGPT can forward/backward without errors, and that loss
decreases over a few gradient steps.

Run with:  python smoke_test.py
No data files needed — uses random token sequences.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from train_gpt import (
    QuadTreeTensor,
    QuadTreeLinear,
    QTBlock,
    PassDelta,
    RecurrentGPT,
    restore_low_dim_params_to_fp32,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"device={DEVICE}  dtype={DTYPE}")

# ── 1. QuadTreeTensor unit test ───────────────────────────────────────────────
def test_quadtree_tensor():
    print("\n[1] QuadTreeTensor")
    for (H, W, L) in [(4, 4, 2), (8, 16, 3), (64, 32, 4)]:
        qt = QuadTreeTensor(H, W, num_levels=L, init_std=0.02).to(DEVICE)
        w = qt.materialize()
        assert w.shape == (H, W), f"bad shape {w.shape}"
        # gradient flows back to vals
        loss = w.sum()
        loss.backward()
        assert qt.vals.grad is not None, "no gradient on vals"
        assert qt.vals.grad.abs().sum() > 0, "zero gradient on vals"
        qt.zero_grad()
    print("  PASS")

# ── 2. QuadTreeLinear unit test ───────────────────────────────────────────────
def test_quadtree_linear():
    print("\n[2] QuadTreeLinear")
    layer = QuadTreeLinear(64, 32, num_levels=3).to(DEVICE)
    x = torch.randn(2, 10, 64, device=DEVICE, dtype=DTYPE)
    y = layer(x)
    assert y.shape == (2, 10, 32), f"bad shape {y.shape}"
    y.sum().backward()
    assert layer.qt.vals.grad is not None
    print("  PASS")

# ── 3. QTBlock unit test ──────────────────────────────────────────────────────
def test_qt_block():
    print("\n[3] QTBlock")
    dim, heads, kv_heads = 64, 4, 2
    block = QTBlock(dim, heads, kv_heads, mlp_mult=2,
                    rope_base=10000.0, qk_gain_init=1.5, qt_levels=3).to(DEVICE)
    # Cast like the training script does
    block.bfloat16()
    restore_low_dim_params_to_fp32(block)
    x  = torch.randn(2, 16, dim, device=DEVICE, dtype=DTYPE)
    x0 = torch.randn(2, 16, dim, device=DEVICE, dtype=DTYPE)
    out = block(x, x0)
    assert out.shape == (2, 16, dim)
    out.sum().backward()
    print("  PASS")

# ── 4. PassDelta unit test ────────────────────────────────────────────────────
def test_pass_delta():
    print("\n[4] PassDelta")
    delta = PassDelta(dim=64, rank=4, qt_levels=2).to(DEVICE)
    delta.bfloat16()
    restore_low_dim_params_to_fp32(delta)
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE)
    out = delta(x)
    assert out.shape == (2, 16, 64)
    out.sum().backward()
    assert delta.down.qt.vals.grad is not None
    print("  PASS")

# ── 5. RecurrentGPT forward pass ──────────────────────────────────────────────
def test_recurrent_gpt_forward():
    print("\n[5] RecurrentGPT — forward")
    model = RecurrentGPT(
        vocab_size=256,
        num_passes=4,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        qt_levels=3,
        delta_rank=4,
        delta_qt_levels=2,
    ).to(DEVICE).bfloat16()
    restore_low_dim_params_to_fp32(model)

    bs, seq = 2, 32
    x = torch.randint(0, 256, (bs, seq), device=DEVICE)
    y = torch.randint(0, 256, (bs, seq), device=DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        loss = model(x, y)
    assert loss.isfinite(), f"non-finite loss: {loss}"
    print(f"  initial loss={loss.item():.4f}  PASS")
    return model

# ── 6. RecurrentGPT loss decreases ───────────────────────────────────────────
def test_loss_decreases():
    print("\n[6] RecurrentGPT — loss decreases over 30 steps")
    torch.manual_seed(42)
    model = RecurrentGPT(
        vocab_size=256,
        num_passes=4,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        qt_levels=3,
        delta_rank=4,
        delta_qt_levels=2,
    ).to(DEVICE).bfloat16()
    restore_low_dim_params_to_fp32(model)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs, seq = 4, 64
    # Fixed batch so we can overfit
    x_fixed = torch.randint(0, 256, (bs, seq), device=DEVICE)
    y_fixed = torch.randint(0, 256, (bs, seq), device=DEVICE)

    losses = []
    for step in range(30):
        opt.zero_grad()
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            loss = model(x_fixed, y_fixed)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == 29:
            print(f"  step {step:2d}  loss={loss.item():.4f}")

    assert losses[-1] < losses[0], (
        f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
    print(f"  loss decreased {losses[0]:.4f} → {losses[-1]:.4f}  PASS")

# ── 7. Parameter count / byte-budget estimate ─────────────────────────────────
def test_param_count():
    print("\n[7] Parameter count + byte-budget estimate")
    # Realistic config: dim=512 like the baseline
    model = RecurrentGPT(
        vocab_size=1024,
        num_passes=8,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        qt_levels=5,
        delta_rank=4,
        delta_qt_levels=3,
    )
    n = sum(p.numel() for p in model.parameters())
    # fp32 bytes (as stored in state_dict)
    fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
    # QuadTree vals are < 65536 → kept as fp16 in int8 pipeline
    fp16_bytes = sum(
        p.numel() * 2
        for name, p in model.named_parameters()
        if "qt.vals" in name or p.ndim < 2
    )
    emb_bytes = model.tok_emb.weight.numel() * 2  # fp16 passthrough
    print(f"  total params:    {n:,}")
    print(f"  fp32 footprint:  {fp32_bytes/1024:.1f} KB  ({fp32_bytes/1024/1024:.2f} MB)")
    print(f"  est. compressed: ~{(fp16_bytes + emb_bytes)/1024:.1f} KB  "
          f"({(fp16_bytes + emb_bytes)/1024/1024:.2f} MB)")
    print(f"  budget headroom: {16 - (fp16_bytes+emb_bytes)/1024/1024:.1f} MB "
          f"(to spend on wider base block)")
    print("  PASS")

# ── 8. torch.compile smoke ────────────────────────────────────────────────────
def test_compile():
    if DEVICE != "cuda":
        pytest.skip("no CUDA")
    print("\n[8] torch.compile")
    model = RecurrentGPT(
        vocab_size=256, num_passes=2, model_dim=64, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        qt_levels=2, delta_rank=4, delta_qt_levels=1,
    ).to(DEVICE).bfloat16()
    restore_low_dim_params_to_fp32(model)
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    x = torch.randint(0, 256, (2, 16), device=DEVICE)
    y = torch.randint(0, 256, (2, 16), device=DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = compiled(x, y)
    assert loss.isfinite()
    loss.backward()
    print(f"  compiled loss={loss.item():.4f}  PASS")


if __name__ == "__main__":
    test_quadtree_tensor()
    test_quadtree_linear()
    test_qt_block()
    test_pass_delta()
    test_recurrent_gpt_forward()
    test_loss_decreases()
    test_param_count()
    test_compile()
    print("\n=== ALL TESTS PASSED ===")
