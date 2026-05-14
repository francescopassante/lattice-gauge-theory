# LGT — Gauge-Equivariant Neural Networks for Lattice Gauge Theory

Master's thesis codebase. Goal: build a **gauge-equivariant graph-attention
network (G-GAT)** for SU(N_c) lattice gauge theory, starting from 2D Z₂ as
a debug-friendly testbed and scaling toward U(1)/SU(2)/SU(3) and 3+1D.

The architecture follows the L-CNN framework (Favoni et al. 2012.12901)
for primitives and gauge-equivariance proofs, with two departures:
(i) the L-Conv + L-Bilin stack is replaced by an attention block whose
**value path is matrix-bilinear** (`α · Q† · Ṽ`), so L-CNN's loop-doubling
universality argument transfers directly; (ii) parallel transport between
sites is **averaged over all shortest lattice paths** in the L1-ball of
Manhattan radius R (computed by a DP recursion, not enumeration), so each
block already reaches the full L1-ball receptive field with non-axis-aligned
loop content.

## Documents

All in `notes/`. Read in order before touching the equivariant model:

- `notes/papers_review.md` — full literature review of L-CNN, the
  gauge-covariant ResNet (Nagai-Tomiya 2103.11965), and CASK (2501.16955).
  Sections 0 (lattice primer) and 1 (L-CNN) are the prerequisites for
  `architecture.md`. Equation references in `architecture.md` point here.
- `notes/architecture.md` — implementation spec for the G-GAT block:
  on-site Q/K/V projections, **shortest-path-averaged** parallel
  transport over the L1-ball (§3.3), gauge-invariant attention scores via
  `Re Tr[Q† · K̃]` (physically a two-loop correlator function — §3.4),
  multiplicative value path `α · Q† · Ṽ` (the key departure from a
  vanilla transformer — preserves L-CNN's loop-doubling expressivity),
  residual + L-Act, `Re Tr` head. Includes a full §10 build-order
  checklist.
- `notes/roadmap.md` — staged plan from sanity checks (Phase 0, 2D Z₂)
  through 3D Z₂ critical exponents, 4D U(1), SU(2) replication of L-CNN
  benchmarks, SLHMC with fermions, SU(3) pure gauge, and Phase-6 novel
  directions (cross-β transfer, attention-as-correlation-length,
  trivializing flows, topological-sector sampling, sign-problem contour
  deformation, …). Each phase has setup / tasks / pass criteria / time
  estimates / pitfalls.
- `notes/sampling.md` — strategy notes for the MC sampler (single-site
  Metropolis for Z₂; extension plan to heat-bath + overrelaxation for
  U(1)/SU(2)/SU(3)).
- `notes/resources.md` — curated textbooks, lecture notes, and ML-for-LGT
  papers with suggested reading order.
- `notes/tunnel-visualization.md` — exploratory notes on visualising
  what the topological-charge network learns about the QCD vacuum.

## Status

**Position on `notes/roadmap.md`:** Phase 0 (2D Z₂ implementation
validation), post-refactor.

The codebase was refactored from the original OO scaffolding
(`Site` / `Link` / `Plaquette` / `Lattice` classes) to **pure tensor
operations** suitable for autograd, vectorisation, and clean generalisation
to U(1)/SU(N). It was then reorganised into a proper Python package:
`lgt/` (library), `scripts/` (entry points), `tests/` (pytest). The CNN
baseline (`LatticeCNN`) is unchanged and trains identically; the saved
L-scan numbers in `scripts/L_scan.py` are still meaningful as a baseline.

What still does **not** exist (in priority order, per
`notes/architecture.md` §10 + `notes/roadmap.md` Phase 0):

1. The **G-GAT block** itself, including `build_transport_sums(U, R)` —
   the DP routine that materialises shortest-path-averaged transports
   `T_Δx(x)` over the positive octant of the L1-ball (§3.3).
2. Gauge-implementation stress-test validation (`notes/architecture.md` §7).
3. Non-Z₂ gauge groups and their production samplers.

The default dataset path uses the Z₂ Metropolis sampler. Haar-random
data remain available via `sampler=haar_ensemble`, and the saved L-scan
numbers in `scripts/L_scan.py` are still Haar-random baseline results.

## Layout

Library lives in `lgt/`; entry-point scripts in `scripts/`; pytest in
`tests/`. The package is installed editable via `pyproject.toml`.

### `lgt/`

- **`lattice.py`** — `GaugeGroup` ABC + `Z2` implementation; pure tensor
  functions:
  - `random_links(L, D, group, dtype)` → `(D, *Λ, nc, nc)`.
  - `plaquette_tensor(U, group)` → `(D(D-1)/2, *Λ, nc, nc)`.
  - `action(U, group, beta=1.0, plaquettes=None)` → scalar Wilson action
    `β Σ_p (1 − Re Tr P / nc)`.
  - `gauge_transformation(U, omega, group)` — apply site-local Ω to
    every link; used by the gauge-invariance unit tests and (once it
    exists) by the G-GAT stress test.
  - `as_ml_input(U)`, `as_ml_plaquettes(P)` — flatten the trailing color
    axes for ML input. Real groups give `(D · nc², *Λ)`; complex groups
    split real/imag, giving `(2 · D · nc², *Λ)`.
- **`sampler.py`** — `staple_sum`, `metropolis_sweep` (checkerboard-
  vectorised single-site Metropolis); `mcmc_ensemble` (thermalise +
  decorrelate + collect, dispatches per group via `_SWEEP_FN`);
  `haar_ensemble` (Haar-uniform, ignores β — shares the sampler
  interface for sanity checks). To plug in U(1)/SU(N) later, add a
  sweep function and register it in `_SWEEP_FN`.
- **`data.py`** — `build_link_datasets`, `build_plaquette_datasets`.
  Take `dtype`, `group`, `beta`, sampler controls, and validated split
  fractions. `structured=True` keeps the full `(N, D, *Λ, nc, nc)` /
  `(N, n_pairs, *Λ, nc, nc)` matrix layout for G-GAT; `structured=False`
  flattens color axes for the CNN baseline. `save=True` writes to
  `datasets/`.
- **`cnn_baseline.py`** — `LatticeCNN(L, D, in_channels, hidden_channels)`.
  CNN baseline only; circular-padded `Conv2d`. Not gauge-equivariant —
  this is the reference against which the G-GAT will be compared.
- **`train.py`** — `train_model` (early stopping, configurable
  `checkpoint_path`); `full_pipeline` returns a dict with `test_loss`,
  `test_label_var`, `test_r2`, `epochs`, and the loss curves. Device order:
  cuda → mps → cpu.

### `scripts/`

- **`main.py`** — single-(L, β) run of the CNN baseline.
- **`L_scan.py`** — replays the saved pre-refactor L-scan numbers and
  produces an absolute-MSE panel and an R² panel (the meaningful one).
  Includes the analytic Haar-random label variance `Var(action) = L²`
  for D=2.
- **`lr_scan.py`** — learning-rate sweep for the CNN baseline at fixed L.
- **`validate_sampler.py`** — four-panel sanity check on the Z₂
  Metropolis sampler: thermalisation, 2D β-scan vs `tanh(β)`,
  3D β-scan (first-order transition), plaquette autocorrelation.
- **`visualize.py`** — 2D lattice visualisation; takes a link tensor
  directly (not a `Lattice` object — the wrapper class no longer exists).

### `tests/`

- **`test_lattice.py`** — gauge-invariance checks on `plaquette_tensor` /
  `action` under `gauge_transformation` (bit-exact in Z₂ float64).
- **`test_data_model.py`** — split-validation and CNN-baseline shape
  guards.

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
- **Parallel transport (G-GAT, not yet implemented):** sum over **all**
  shortest lattice paths in the L1-ball — never a single axis-aligned
  path. Build via DP on `|Δx|_1`, materialise the positive octant of the
  L1-ball, derive negatives via the octant trick
  `T_{-Δx}(x) = dagger(T_Δx(x - Δx))`.
  See `notes/architecture.md` §3.3 + §10 step 1.
- **Float32** for training; pass `dtype=torch.float64` through the
  dataset builders for high-precision gauge-invariance unit tests.
  Worst-case-Ω stress tests (`notes/architecture.md` §7.2) should report
  drift in double precision.

## Running

The package is installed editable; scripts are run from the repo root.

```bash
python scripts/main.py                # single-(L, β) CNN run
python scripts/L_scan.py              # replay saved L-scan, regenerate R² plots
python scripts/lr_scan.py             # CNN LR sweep
python scripts/validate_sampler.py    # Z₂ Metropolis four-panel sanity check
python scripts/visualize.py           # plot a seeded random 5×5 lattice
python -m lgt.cnn_baseline            # torchsummary for a 5×5 CNN
pytest tests                          # unit tests
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
(in `scripts/L_scan.py`) puts every L on the same scale and reveals two
distinct regimes:

| Input | Result | Interpretation |
|---|---|---|
| plaquettes | R² ≈ 0.99 across all L | trivial: the action is a linear sum of inputs |
| links | R² ≈ 0 across all L | the CNN cannot reconstruct plaquettes from links — no inductive bias for "multiply four specific link values" |

This is the inductive-bias gap that motivates the G-GAT. Even a perfect
action regressor on Haar-random data is only memorising the action
*function*; β-dependent physics requires the Metropolis data path.

## Things to keep in mind

- **Do not silently broadcast across color axes.** Every matmul should
  be explicit (`A @ B`) and every dagger explicit (`group.dagger(A)`);
  for Z₂ both are no-ops, but for U(1)/SU(N) any laxity is a bug that
  Z₂ cannot catch. The same applies to the future G-GAT transport: a
  missed dagger or a wrong shortest-path step will pass Z₂ tests and
  fail at the first non-abelian Ω.
- **`LatticeCNN` is 2D-only** (`Conv2d` with circular padding); raises
  `NotImplementedError` for `D ≠ 2`. Generalising means switching to
  `Conv3d`/`ConvNd` or factoring the convolution layer.
- **Datasets do not store β.** For multi-β training, β should become
  part of the dataset so the model can be conditioned on it (Phase 1+
  of the roadmap requires this).
- **Saved `best_model.pth` from before the refactor is from a Haar-random
  run with no β** — not useful as a checkpoint for any equivariant
  experiment.
- **Sampler dispatch is by group type.** `mcmc_ensemble` looks up
  `_SWEEP_FN[type(group)]`; adding U(1)/SU(N) is a one-line registry
  entry plus the sweep function.
- **Do not remove comments unless asked to.**

## Suggested next steps

In strict order, per `notes/architecture.md` §10 / `notes/roadmap.md`
Phase 0:

1. **Lattice utility for G-GAT:** `build_transport_sums(U, R)` — DP over
   the positive octant of the L1-ball, ordered by `|Δx|_1`. Verify
   against brute-force path enumeration for `|Δx|_1 ≤ 2`; test the
   octant trick under a random Ω. (`notes/architecture.md` §10 step 1.)
2. **G-GAT block.** Build incrementally per the §10 checklist (Plaq /
   Poly / augment / Trace → Q/K/V → transport → score → softmax →
   multiplicative value → channel mix → residual + L-Act), with a
   covariance unit test after each step.
3. **Gauge-implementation stress test** on the untrained G-GAT before
   training (`notes/architecture.md` §7) — random Ω + worst-case-Ω
   search via AdamW on `ρ^a_x`. Drift must stay at machine epsilon;
   anything larger is a bug (almost always a missed dagger or a
   non-axis-aligned transport path).
4. **Replicate Phase 3 (SU(2) Wilson loops + topological charge)** of
   `notes/roadmap.md` once the architecture is validated on Z₂. The
   matched-parameter shootout vs. L-CNN at low parameter count and the
   attention-range-vs-correlation-length plot are the thesis's central
   novel results.
