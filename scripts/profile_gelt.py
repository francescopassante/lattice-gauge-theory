"""Profile GELT on V100 to find the real hot path before optimizing.

Outputs:
  1. torch.profiler top-ops table (sorted by self_cuda_time_total) — tells you
     which low-level kernels (matmul, gather, reduce, ...) dominate.
  2. Chrome trace at trace.json — open in chrome://tracing or
     https://ui.perfetto.dev/ for a per-kernel timeline.
  3. Manual section timings (CUDA-event based) for the four phases inside one
     GEMHSA block:
       (a) QKV projection (augment + 3 matmuls)
       (b) neighbour gather K_nb / V_nb
       (c) adjoint transport T @ X @ T†  (twice: K and V)
       (d) score (Frobenius product + softmax) + value path (Q† @ V_tilde)
       (e) residual + L-Act gate
     Each is timed across many repeats so noise is small.
  4. Peak memory after a forward+backward.

Run on V100:
    python scripts/profile_gelt.py
"""

import time
import torch
import torch.nn as nn

from gelt import SU, build_plaquette_datasets, haar_ensemble
from gelt.blocks import GELT, GEMHSA
import gelt.blocks as blocks_mod


# ---------- shape config — same as scripts/bench_optimized.py ----------
D, L, R = 3, 8, 2
GROUP = SU(2)
B = 128
N_REPEATS = 20
N_WARMUP = 5

MODEL_KW = dict(
    gaugegroup=GROUP, L=L, D=D, R=R,
    nhead=2, gemhsa_layers=2, d_qkv=8,
    gate="softplus", dtype=torch.complex64,
    mlp_hidden=16, mlp_out=1,
)


def build_one_batch(device):
    ds_train, _, _ = build_plaquette_datasets(
        N=200, D=D, L=L, gaugegroup=GROUP, R=R,
        splits=[0.7, 0.15, 0.15], save=False, structured=True,
        sampler=haar_ensemble, beta=1, n_therm=200, n_skip=5,
        dtype=torch.float32,
    )
    loader = torch.utils.data.DataLoader(ds_train, batch_size=B, shuffle=False)
    X, T, y = next(iter(loader))
    return X.to(device), T.to(device), y.to(device)


def cuda_timed(fn, n=N_REPEATS, warmup=N_WARMUP):
    """Return median ms over `n` runs, after `warmup` untimed runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(start, end))
    return times[n // 2], times[0], times[-1]


def section_timings(model, X, T):
    """Manual per-section CUDA timings for ONE GEMHSA block (layer 0)."""
    layer: GEMHSA = model.gemhsa_models[0]
    W = X
    nc = W.shape[-1]
    trailing = W.shape[2:]
    Bsz = W.shape[0]

    print("\n=== per-section timings (GEMHSA layer 0, fwd only) ===")

    # (a) QKV projection
    def f_qkv():
        W_aug = layer._augment(W)
        W_aug_flat = W_aug.view(Bsz, layer.C_tilde, -1)
        wQ = layer.w_Q.view(layer.H * layer.d_qkv, layer.C_tilde)
        wK = layer.w_K.view(layer.H * layer.d_qkv, layer.C_tilde)
        wV = layer.w_V.view(layer.H * layer.d_qkv, layer.C_tilde)
        Q = torch.matmul(wQ, W_aug_flat).view(Bsz, layer.H, layer.d_qkv, *trailing)
        K = torch.matmul(wK, W_aug_flat).view(Bsz, layer.H, layer.d_qkv, *trailing)
        V = torch.matmul(wV, W_aug_flat).view(Bsz, layer.H, layer.d_qkv, *trailing)
        return Q, K, V
    Q, K, V = f_qkv()
    med, lo, hi = cuda_timed(f_qkv)
    print(f"  (a) QKV projection           : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]")

    # (b) neighbour gather
    idx = tuple(layer._nbr_idx[d] for d in range(layer.D))
    nb_indexer = (slice(None),) * 3 + idx + (slice(None), slice(None))
    def f_gather():
        return K[nb_indexer], V[nb_indexer]
    K_nb, V_nb = f_gather()
    med, lo, hi = cuda_timed(f_gather)
    print(f"  (b) neighbour gather (K, V)  : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]")

    # (c) adjoint transport T @ X @ T†, for K and V
    T_b = T.unsqueeze(1).unsqueeze(1)
    T_b_dag = layer.gaugegroup.dagger(T_b)
    def f_transport():
        K_tilde = T_b @ K_nb @ T_b_dag
        V_tilde = T_b @ V_nb @ T_b_dag
        return K_tilde, V_tilde
    K_tilde, V_tilde = f_transport()
    med, lo, hi = cuda_timed(f_transport)
    print(f"  (c) transport T·X·T† (K & V) : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]  <-- usually the hot path")

    # (d) score + softmax + value path
    import math
    def f_score_value():
        Q_e = Q.unsqueeze(3)
        score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real / math.sqrt(layer.d_qkv * nc)
        bias = layer.b_h[:, layer._orbit_idx]
        if torch.is_complex(bias):
            bias = bias.real
        score = score + bias.view(1, layer.H, layer.n_offsets, *([1] * layer.D))
        alpha = torch.softmax(score, dim=2)
        Q_dag = layer.gaugegroup.dagger(Q).unsqueeze(3)
        QdagV = torch.matmul(Q_dag, V_tilde)
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        return (alpha_b * QdagV).sum(dim=3)
    out = f_score_value()
    med, lo, hi = cuda_timed(f_score_value)
    print(f"  (d) score + softmax + value  : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]")

    # (e) channel mix + residual + L-Act gate
    import torch.nn.functional as F
    def f_residual():
        W_mix = torch.einsum("iha,bha...->bi...", layer.w_mix, out)
        W_res = W + W_mix
        trace_per_chan = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
        g = F.softplus(trace_per_chan) if layer.gate == "softplus" else F.relu(trace_per_chan)
        W_act = g.unsqueeze(-1).unsqueeze(-1) * W_res
        return W + layer.alpha * (W_act - W)
    _ = f_residual()
    med, lo, hi = cuda_timed(f_residual)
    print(f"  (e) mix + residual + gate    : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]")

    # Whole layer for reference
    def f_layer():
        return layer(W, T)
    _ = f_layer()
    med, lo, hi = cuda_timed(f_layer)
    print(f"  ----")
    print(f"  total one GEMHSA layer (fwd) : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]")


def torch_profiler_run(model, X, T, y, criterion):
    """Capture a top-ops table and a Chrome trace."""
    from torch.profiler import profile, ProfilerActivity, schedule

    print("\n=== torch.profiler — top kernels by self_cuda_time_total ===")
    sched = schedule(wait=1, warmup=2, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + 2 + 3):
            out = model(X, T)
            loss = criterion(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            prof.step()

    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25
    ))
    trace_path = "trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to {trace_path}  (open in chrome://tracing or https://ui.perfetto.dev/)")


def peak_memory_report(model, X, T, y, criterion):
    torch.cuda.reset_peak_memory_stats()
    out = model(X, T)
    loss = criterion(out, y)
    loss.backward()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\n[mem] peak CUDA memory allocated this step: {peak_mb:.1f} MB")


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "This profiler is intended for the V100."
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.perf_counter()
    X, T, y = build_one_batch(device)
    print(f"[data] one batch built in {time.perf_counter() - t0:.2f}s | "
          f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype}")

    model = GELT(**MODEL_KW).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] GELT params: {n_params:,} | n_offsets per layer: "
          f"{model.gemhsa_models[0].n_offsets}")

    criterion = nn.MSELoss()

    # Warm everything up before any timing.
    for _ in range(3):
        out = model(X, T)
        loss = criterion(out, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Per-section timings (forward only on layer 0, with no grad).
    with torch.no_grad():
        section_timings(model, X, T)

    # Full-step (fwd + bwd) timing for reference.
    def f_step():
        out = model(X, T)
        loss = criterion(out, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    med, lo, hi = cuda_timed(f_step, n=10, warmup=3)
    print(f"\n[step] full fwd+bwd (B={B})    : median {med:7.2f} ms  [min {lo:7.2f}, max {hi:7.2f}]")

    peak_memory_report(model, X, T, y, criterion)
    torch_profiler_run(model, X, T, y, criterion)


if __name__ == "__main__":
    main()
