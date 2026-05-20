"""Numerical equivalence check: gelt.blocks_optimized vs gelt.blocks.

Both models must produce bit-comparable outputs (up to floating-point
reassociation tolerance) for the same parameters and inputs. We copy the
parameters from the original into the optimized model, then compare:

  1. Forward output of one GEMHSA layer.
  2. Forward output of the full GELT stack.
  3. Backward: total grad-norm and per-parameter max abs diff.

Runs on CPU so you can sanity-check before scheduling V100 time.
"""

import torch

from gelt import SU
from gelt.blocks import GEMHSA as OrigGEMHSA, GELT as OrigGELT
from gelt.blocks_optimized import GEMHSA as OptGEMHSA, GELT as OptGELT


def copy_params(src: torch.nn.Module, dst: torch.nn.Module):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    assert set(src_sd.keys()) == set(dst_sd.keys()), (
        f"state_dict key mismatch:\n"
        f"  src - dst: {set(src_sd.keys()) - set(dst_sd.keys())}\n"
        f"  dst - src: {set(dst_sd.keys()) - set(src_sd.keys())}"
    )
    dst.load_state_dict(src_sd, strict=True)


def make_inputs(B, D, L, R, nc, n_offsets, dtype, gen):
    spatial = (L,) * D
    W = (torch.randn(B, D * (D - 1) // 2, *spatial, nc, nc, generator=gen, dtype=torch.float32)
         + 1j * torch.randn(B, D * (D - 1) // 2, *spatial, nc, nc, generator=gen, dtype=torch.float32))
    T = (torch.randn(B, n_offsets, *spatial, nc, nc, generator=gen, dtype=torch.float32)
         + 1j * torch.randn(B, n_offsets, *spatial, nc, nc, generator=gen, dtype=torch.float32))
    return W.to(dtype), T.to(dtype)


def check_layer(D, L, R, nc, nhead, d_qkv, dtype):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    group = SU(nc)
    d_input = D * (D - 1) // 2

    orig = OrigGEMHSA(group, L, D, R, d_input, nhead, d_qkv=d_qkv, gate="softplus", dtype=dtype)
    opt = OptGEMHSA(group, L, D, R, d_input, nhead, d_qkv=d_qkv, gate="softplus", dtype=dtype)
    copy_params(orig, opt)

    B = 3
    n_offsets = orig.n_offsets
    W, T = make_inputs(B, D, L, R, nc, n_offsets, dtype, gen)

    with torch.no_grad():
        y_orig = orig(W, T)
        y_opt = opt(W, T)

    max_diff = (y_orig - y_opt).abs().max().item()
    rel = max_diff / (y_orig.abs().max().item() + 1e-12)
    print(f"[GEMHSA D={D} L={L} R={R} nc={nc} nhead={nhead} d_qkv={d_qkv} dtype={dtype}] "
          f"max |Δ| = {max_diff:.3e}  rel = {rel:.3e}")
    return max_diff, rel


def check_full_model(D, L, R, nc, nhead, d_qkv, n_layers, dtype):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    group = SU(nc)

    orig = OrigGELT(group, L, D, R, nhead, gemhsa_layers=n_layers, d_qkv=d_qkv,
                    gate="softplus", dtype=dtype, mlp_hidden=16, mlp_out=1)
    opt = OptGELT(group, L, D, R, nhead, gemhsa_layers=n_layers, d_qkv=d_qkv,
                  gate="softplus", dtype=dtype, mlp_hidden=16, mlp_out=1)
    copy_params(orig, opt)

    B = 3
    n_offsets = orig.gemhsa_models[0].n_offsets
    W, T = make_inputs(B, D, L, R, nc, n_offsets, dtype, gen)

    # Forward.
    y_orig = orig(W, T)
    y_opt = opt(W, T)
    max_diff = (y_orig - y_opt).abs().max().item()
    rel = max_diff / (y_orig.abs().max().item() + 1e-12)
    print(f"[GELT  D={D} L={L} R={R} nc={nc} layers={n_layers} d_qkv={d_qkv} dtype={dtype}] "
          f"forward max |Δ| = {max_diff:.3e}  rel = {rel:.3e}")

    # Backward — compare gradient max-abs over all parameters.
    target = torch.randn_like(y_orig)
    loss_o = ((y_orig - target) ** 2).mean()
    loss_p = ((y_opt - target) ** 2).mean()
    loss_o.backward()
    loss_p.backward()

    worst_grad_diff = 0.0
    worst_name = ""
    for (no, po), (np_, pp) in zip(orig.named_parameters(), opt.named_parameters()):
        assert no == np_, f"param name mismatch: {no} vs {np_}"
        if po.grad is None and pp.grad is None:
            continue
        diff = (po.grad - pp.grad).abs().max().item()
        if diff > worst_grad_diff:
            worst_grad_diff = diff
            worst_name = no
    print(f"  worst grad |Δ| over all params: {worst_grad_diff:.3e}  ({worst_name})")
    return max_diff, rel, worst_grad_diff


if __name__ == "__main__":
    # Small problem in complex128 — tight tolerance proves the math, not luck.
    print("=== complex128, small ===")
    check_layer(D=2, L=4, R=1, nc=2, nhead=2, d_qkv=4, dtype=torch.complex128)
    check_layer(D=3, L=4, R=2, nc=2, nhead=2, d_qkv=8, dtype=torch.complex128)
    check_full_model(D=3, L=4, R=2, nc=2, nhead=2, d_qkv=8, n_layers=2, dtype=torch.complex128)

    # SU(3) sanity check.
    print("\n=== complex128, SU(3) ===")
    check_layer(D=2, L=4, R=1, nc=3, nhead=2, d_qkv=4, dtype=torch.complex128)

    # complex64 at the benchmark shape (memory-light: B=3 instead of 128).
    print("\n=== complex64, benchmark-like shape (B=3) ===")
    check_layer(D=3, L=8, R=2, nc=2, nhead=2, d_qkv=8, dtype=torch.complex64)
    check_full_model(D=3, L=8, R=2, nc=2, nhead=2, d_qkv=8, n_layers=2, dtype=torch.complex64)
