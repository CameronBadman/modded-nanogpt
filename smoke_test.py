"""
Smoke test for the core recurrent trainer path.

Run with: python smoke_test.py
No data files needed; uses random token sequences.
"""

import torch

from train_gpt import PassDelta, RecurrentGPT, restore_low_dim_params_to_fp32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"device={DEVICE}  dtype={DTYPE}")


def test_pass_delta():
    print("\n[1] PassDelta")
    delta = PassDelta(dim=64, rank=4).to(DEVICE)
    delta.bfloat16()
    restore_low_dim_params_to_fp32(delta)
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE)
    out = delta(x)
    assert out.shape == (2, 16, 64)
    out.sum().backward()
    assert delta.down.weight.grad is not None
    assert delta.up.weight.grad is not None
    print("  PASS")


def test_recurrent_gpt_forward():
    print("\n[2] RecurrentGPT — forward")
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
        delta_rank=4,
    ).to(DEVICE).bfloat16()
    restore_low_dim_params_to_fp32(model)

    x = torch.randint(0, 256, (2, 32), device=DEVICE)
    y = torch.randint(0, 256, (2, 32), device=DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        loss = model(x, y)
    assert loss.isfinite(), f"non-finite loss: {loss}"
    print(f"  initial loss={loss.item():.4f}  PASS")


def test_recurrent_gpt_delta_modes():
    print("\n[3] RecurrentGPT — delta modes")
    x = torch.randint(0, 256, (2, 16), device=DEVICE)
    y = torch.randint(0, 256, (2, 16), device=DEVICE)
    for delta_mode in ("per_pass", "shared", "scalar", "none"):
        model = RecurrentGPT(
            vocab_size=256,
            num_passes=3,
            model_dim=64,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            delta_rank=4,
            delta_mode=delta_mode,
        ).to(DEVICE).bfloat16()
        restore_low_dim_params_to_fp32(model)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            loss = model(x, y)
        assert loss.isfinite(), f"{delta_mode=} produced non-finite loss"
    print("  PASS")


def test_loss_decreases():
    print("\n[4] RecurrentGPT — loss decreases")
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
        delta_rank=4,
    ).to(DEVICE).bfloat16()
    restore_low_dim_params_to_fp32(model)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_fixed = torch.randint(0, 256, (4, 64), device=DEVICE)
    y_fixed = torch.randint(0, 256, (4, 64), device=DEVICE)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            loss = model(x_fixed, y_fixed)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  loss decreased {losses[0]:.4f} -> {losses[-1]:.4f}  PASS")


def test_param_count():
    print("\n[5] Parameter count + byte-budget estimate")
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
        delta_rank=32,
    )
    n = sum(p.numel() for p in model.parameters())
    fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
    print(f"  total params:   {n:,}")
    print(f"  fp32 footprint: {fp32_bytes / 1024 / 1024:.2f} MB")
    print("  PASS")


def test_compile():
    if DEVICE != "cuda":
        print("\n[6] torch.compile  SKIP (no CUDA)")
        return
    print("\n[6] torch.compile")
    model = RecurrentGPT(
        vocab_size=256,
        num_passes=2,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        delta_rank=4,
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
    test_pass_delta()
    test_recurrent_gpt_forward()
    test_recurrent_gpt_delta_modes()
    test_loss_decreases()
    test_param_count()
    test_compile()
    print("\n=== ALL TESTS PASSED ===")
