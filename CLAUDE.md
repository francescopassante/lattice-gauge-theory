# LGT — Gauge-Equivariant Neural Networks for Lattice Gauge Theory

Master's thesis codebase. Goal: build a **gauge-equivariant graph-attention
network (G-GAT)** for SU(N_c) lattice gauge theory, starting from 2D Z₂ as
a debug-friendly testbed and scaling toward U(1)/SU(2)/SU(3) and 3+1D.

The architecture follows the L-CNN framework (Favoni et al. 2012.12901) for
primitives and gauge-equivariance proofs, but replaces the L-Conv + L-Bilin
stack with an attention block whose value path is matrix-bilinear, so that
L-CNN's loop-doubling universality argument transfers directly.

## Documents

Read these in order before touching the equivariant model:

- `gauge_invariant_NN_review.md` — full literature review of L-CNN, the
  gauge-covariant ResNet (Nagai-Tomiya 2103.11965), and CASK (2501.16955).
  Sections 0 (lattice primer) and 1 (L-CNN) are the prerequisites for
  `architecture.md`.
- `architecture.md` — implementation spec for the G-GAT block: on-site
  Q/K/V projections, parallel-transported attention, gauge-invariant
  attention scores via `Re Tr[Q · K̃]`, multiplicative value path
  `α · Q · Ṽ` (this is the key departure from a vanilla transformer —
  preserves L-CNN's loop-doubling expressivity), residual + L-Act,
  `Re Tr` head. Includes a full §10 build-order checklist.
- `roadmap.md` — staged plan from sanity checks (Phase 0, 2D Z₂) through
  4D U(1), SU(2) replication of L-CNN benchmarks, SLHMC with fermions,
  SU(3) pure gauge, and Phase-6 novel directions (cross-β transfer,
  trivializing flows, topological-sector sampling, sign-problem contour
  deformation, …). Each phase has setup / tasks / pass criteria / time
  estimates / pitfalls.
- `notes.md` — short physics primer for Z₂ lattice gauge theory.

## Status

**Position on `roadmap.md`:** Phase 0 (2D Z₂ implementation validation),
post-refactor.

The codebase has been refactored from the original OO scaffolding
(`Site` / `Link` / `Plaquette` / `Lattice` classes) to **pure tensor
operations** suitable for autograd, vectorisation, and clean generalisation
to U(1)/SU(N). The CNN baseline (`LatticeCNN`) is unchanged and trains
identically; the saved L-scan numbers in `L_scan_plots.py` are still
meaningful as a baseline.

What still does **not** exist (in priority order, per
`architecture.md` §10 + `roadmap.md` Phase 0):

1. `gauge_transform(U, Ω)` and the corresponding gauge-invariance unit
   tests. This is the single hardest gate before any equivariant work.
2. Metropolis (or heat-bath) MC sampler — needed for any β-dependent
   observable. Without it, the action labels carry no physics signal.
3. The **G-GAT block** itself.
4. Gauge-implementation stress-test validation (`architecture.md` §7).

Current data are Haar-random (uniform ±1 per link), which is fine for
sanity checks and for the CNN baseline but cannot reach any physics
result. See "Why the saved L-scan losses are not signal" below.

## Layout

- **`lattice.py`** — `GaugeGroup` ABC + `Z2` implementation; pure tensor
  functions:
  - `random_links(L, D, group, dtype)` → `(D, *Λ, nc, nc)`.
  - `plaquette_tensor(U, group)` → `(D(D-1)/2, *Λ, nc, nc)`.
  - `action(U, group, beta=1.0, plaquettes=None)` → scalar Wilson action
    `β Σ_p (1 − Re Tr P / nc)`.
  - `as_ml_input(U)`, `as_ml_plaquettes(P)` — flatten the trailing color
    axes for ML input. Real groups give `(D · nc², *Λ)`; complex groups
    split real/imag, giving `(2 · D · nc², *Λ)`.
- **`data.py`** — `build_link_datasets`, `build_plaquette_datasets`. Take
  `seed`, `dtype`, `group`, `beta`. 
  `save=True` writes
  to `datasets/`.
- **`model.py`** — `LatticeCNN(L, D, in_channels, hidden_channels)`.
  CNN baseline only; circular-padded `Conv2d`. Not gauge-equivariant —
  this is the reference against which the G-GAT will be compared.
- **`train.py`** — `train_model` (early stopping, configurable
  `checkpoint_path`); `full_pipeline` returns a `TrainResult` namedtuple
  with `test_loss`, `test_label_var`, `test_r2`, `epochs`, and the
  loss curves. Device order: cuda → mps → cpu.
- **`main.py`** — L-scan driver. Seeded; writes per-L checkpoint files
  (`best_model_L{L}.pth`).
- **`L_scan_plots.py`** — replays the saved pre-refactor L-scan numbers
  and produces an absolute-MSE panel and an R² panel (the meaningful
  one). Includes the analytic Haar-random label variance
  `Var(action) = L²` for D=2.
- **`visualize.py`** — 2D lattice visualisation; takes a link tensor
  directly (not a `Lattice` object — the wrapper class no longer exists).

## Conventions

- **Tensor layouts** (the only spec, no OO wrappers):
  - Links: `(D, L, ..., L, nc, nc)`. Direction first, spatial axes,
    color axes last.
  - Plaquettes: `(n_pairs, L, ..., L, nc, nc)` with
    `n_pairs = D(D-1)/2`, ordered by `(μ, ν)` with `μ < ν` lexicographically.
- **Color axes are always present**, even for Z₂ where `nc = 1`. Every
  product is written as `A @ B` and every inverse as `group.dagger(A)`,
  so the code ports verbatim to U(1)/SU(N).
- **Plaquette convention:**
  `P_{μν}(x) = U_μ(x) · U_ν(x + μ̂) · U_μ†(x + ν̂) · U_ν†(x)`.
- **Periodic BCs:** `torch.roll` for shifts. Never manual modulo arithmetic
  on indices (it's harder to vectorise and harder to read).
- **Wilson action:** `S = β Σ_p (1 − Re Tr P / nc)`. β defaults to 1.0,
  reproducing the legacy unnormalised form `n_plaq − Σ P` for Z₂.
- **Float32** for training; pass `dtype=torch.float64` through the
  dataset builders for high-precision gauge-invariance unit tests once
  `gauge_transform` exists. Worst-case-Ω stress tests (`architecture.md`
  §7.2) should report drift in double precision.

## Running

```bash
python main.py            # L-scan driver
python model.py           # torchsummary for a 5×5 model
python visualize.py       # plots a seeded random 5×5 lattice
python L_scan_plots.py    # regenerate R² plots from saved L-scan numbers
```

`.venv/` is local (uv-style, not gitignored). `datasets/`, `*.pth`,
`*.png` are gitignored.

## Why the saved L-scan losses are not signal

For Haar-random Z₂ links in 2D, every plaquette is ±1 with mean 0, and
plaquette pairs share either 0 or 2 links — both cases give zero
covariance, so plaquettes are independent under random ±1 links. With
`n_plaq = L²` independent ±1 contributions:

```
Var(action) = Var(n_plaq − Σ_p P_p) = n_plaq = L².
```

So absolute MSE that grows like L² is just the label scale growing — it
carries no information about generalisation. `R² = 1 − MSE / Var(y)`
(in `L_scan_plots.py`) puts every L on the same scale and reveals two
distinct regimes:

| Input | Result | Interpretation |
|---|---|---|
| plaquettes | R² ≈ 0.99 across all L | trivial: the action is a linear sum of inputs |
| links | R² ≈ 0 across all L | the CNN cannot reconstruct plaquettes from links — no inductive bias for "multiply four specific link values" |

This is the inductive-bias gap that motivates the G-GAT. Even a perfect
action regressor on Haar-random data is only memorising the action
*function* — nothing physical (β-dependence, phase transitions) is
reachable until Metropolis MC is in place.

## Things to keep in mind

- **Do not silently broadcast across color axes.** Every matmul should
  be explicit (`A @ B`) and every dagger explicit (`group.dagger(A)`);
  for Z₂ both are no-ops, but for U(1)/SU(N) any laxity is a bug that
  Z₂ cannot catch.
- **`LatticeCNN` is 2D-only** (`Conv2d` with circular padding); raises
  `NotImplementedError` for `D ≠ 2`. Generalising means switching to
  `Conv3d`/`ConvNd` or factoring the convolution layer.
- **Datasets do not store β.** Once Metropolis is in, β should become
  part of the dataset so the model can be conditioned on it (Phase 1+
  of the roadmap requires this).
- **Saved `best_model.pth` from before the refactor is from a Haar-random
  run with no β** — not useful as a checkpoint for any equivariant
  experiment.
- **Do not remove comments unless asked to**

## Suggested next steps

In strict order, per `architecture.md` §10 / `roadmap.md` Phase 0:

1. **`gauge_transform(U, Ω)` + invariance unit tests.** Write Ω as
   `(*Λ, nc, nc)` group elements; transform links by
   `Ω_x · U_{x,μ} · Ω†_{x+μ̂}`. Unit test: `plaquette_tensor` is bit-equal
   under transformation in float64. *Do not start anything else before
   this passes.*
2. **Metropolis MC.** Single-site updates for Wilson action; track
   autocorrelation. Replace the Haar-random data path in `data.py`.
3. **G-GAT block.** Build incrementally per the §10 checklist —
   covariance unit test after each step (Q/K/V projection, parallel
   transport, score, softmax, multiplicative value, residual + L-Act).
4. **Gauge-implementation stress test** on the untrained G-GAT before
   training (`architecture.md` §7) — random Ω + worst-case-Ω search.
   Drift must stay at machine epsilon; anything larger is a bug
   (almost always a missed dagger or a non-axis-aligned transport path).
5. **Replicate Phase 3 (SU(2) Wilson loops + topological charge)** of
   `roadmap.md` once the architecture is validated on Z₂.
