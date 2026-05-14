# Gauge-Equivariant Graph Attention Network (G-GAT) — Implementation Plan

This document specifies a gauge-equivariant transformer / graph-attention
architecture for SU(N_c) lattice gauge theory. It follows the L-CNN framework
(arXiv:2012.12901) for primitives and gauge-equivariance proofs, but with
two departures: (i) the L-Conv + L-Bilin stack is replaced by an attention
block whose **value path is matrix-bilinear** (not just scalar-weighted),
so that L-CNN's loop-doubling universality argument transfers directly;
(ii) parallel transport between sites is **averaged over all shortest
lattice paths** rather than fixed to a single axis-aligned path, so each
block already reaches the full L1-ball receptive field with non-axis-aligned
loop content.

The reader is expected to have read `gauge_invariant_NN_review.md` in this
directory (especially Sec. 0 and Sec. 1). Equation references like "Eq. 5"
below refer to that document unless stated otherwise.

---

## 1. Conventions and data structures

### 1.1 Lattice

- Hypercubic lattice Λ of shape `(N_t, N_s, N_s, ..., N_s)`, dimension `D+1`.
- Periodic boundary conditions on every axis.
- Sites indexed `x ∈ Λ`. Direction index `μ ∈ {0, 1, ..., D}`.
- `N_c` = colors (use `N_c = 2` for first experiments, matching paper 1).

### 1.2 Tensors

Store everything as **complex** tensors (use `torch.complex64` /
`torch.complex128`). If the framework's complex autograd is incomplete, split
into `(Re, Im)` and process as `2·N_c²` real channels via `torch.einsum`.

- Links `U`: shape `(B, D+1, *Λ, N_c, N_c)`, each `U[b, μ, x]` ∈ SU(N_c).
- Locally covariant field `W`: shape `(B, C, *Λ, N_c, N_c)`,
  `W[b, i, x]` ∈ ℂ^{N_c × N_c}, where `C` = number of channels.
- Inverse links: `U_{x+μ̂, -μ} = U†_{x, μ}` (computed on demand, not stored).

### 1.3 Gauge transformation laws

- Links: `U_{x,μ} → Ω_x U_{x,μ} Ω†_{x+μ̂}`.
- W-channels: `W_{x,i} → Ω_x W_{x,i} Ω†_x` (locally covariant, same site).

A function is **equivariant** if `f(T_Ω U, T_Ω W) = T_Ω f(U, W)` and
**invariant** if `f(T_Ω U, T_Ω W) = f(U, W)`. Every layer below is one or
the other; this must be preserved by construction (do not rely on training).

---

## 2. Preprocessing layers (fixed, non-trainable)

These mirror Plaq, Poly, Trace from L-CNN (Sec. 1.2 of the review).

### 2.1 Plaq

Compute every 1×1 plaquette
```
U_{x,μν} = U_{x,μ} U_{x+μ̂,ν} U†_{x+ν̂,μ} U†_{x,ν}      for μ < ν
```
Store as W-channels. Output W has `(D+1)·D/2` channels per site.

### 2.2 Poly (optional)

Straight Polyakov loop along each axis through each site (Eq. 11). Add as
`(D+1)` extra channels. Required if non-contractible loops are part of the
target.

### 2.3 Channel augmentation (call this `augment`)

Before any L-Bilin-like operation, expand a W tensor of `C` channels to
`(2C + 1)` channels by prepending the identity matrix `1` and appending the
Hermitian conjugate of every channel:
```
W → [ 1, W_1, ..., W_C, W†_1, ..., W†_C ]
```
This gives bias and orientation-reversal "for free" and is the only way for
the bilinear block to reduce to identity at initialization. **Apply this
inside the G-Attn block before forming Q, K, V.**

### 2.4 Trace (Eq. 10) — invariant readout

```
T_{x,i} = Tr[ W_{x,i} ]    (complex scalar per site, channel)
```
Use only at the end of the network. Feed `Re T` and/or `Im T` into a small
MLP head.

---

## 3. The G-Attn block (core contribution)

This replaces L-CB (combined L-Conv + L-Bilin) in L-CNN. One block maps
`W^{(in)} → W^{(out)}` with the same shape, in a way that is
gauge-equivariant and that **bakes a matrix product into the output** so
that loop area roughly doubles per block (same scaling rule as L-CB:
`n_blocks ≥ ⌈log₂ N²⌉` for an N×N target loop).

### 3.1 Hyperparameters

| Symbol | Meaning                                                  |
|--------|----------------------------------------------------------|
| `H`    | number of attention heads                                |
| `d`    | per-head channel dimension (so total channels = H · d)   |
| `R`    | attention range — Manhattan radius of the receptive field (L1-ball) |
| `offsets` | set of allowed lattice offsets `Δx ∈ Z^(D+1)` with `0 < |Δx|_1 ≤ R`, where `|Δx|_1 = Σ_μ |Δx_μ|` |

Receptive field is the full L1-ball — diagonal and multi-axis offsets are
included, with parallel transport averaged over all shortest paths
(Sec. 3.3). Offset count in D+1 = 4 (generating function `[(1+z)/(1-z)]⁴`):

| R | non-origin offsets `\|Δx\|_1 ≤ R` |
|---|---|
| 1 | 8   |
| 2 | 40  |
| 3 | 128 |
| 4 | 320 |

### 3.2 On-site channel projections (Q, K, V)

Augment `W^{(in)}` with `1` and Hermitian conjugates → `W̃` with
`C̃ = 2C + 1` channels. For each head `h ∈ {1..H}` define three independent
trainable complex weight matrices `w^Q_h, w^K_h, w^V_h ∈ ℂ^{d × C̃}` and form

```
Q_{x, h, a} = Σ_{j=1..C̃}  w^Q_{h, a, j}  W̃_{x, j}      a ∈ {1..d}
K_{x, h, a} = Σ_{j=1..C̃}  w^K_{h, a, j}  W̃_{x, j}
V_{x, h, a} = Σ_{j=1..C̃}  w^V_{h, a, j}  W̃_{x, j}
```

Each Q, K, V is a stack of `H · d` matrices in ℂ^{N_c × N_c}, locally
covariant: `Q_{x,h,a} → Ω_x Q_{x,h,a} Ω†_x` (linearity in W̃).

Implementation: a single `einsum("hajc,bjcxyz...mn->bhaxyz...mn", w, W̃)`
contraction per Q/K/V, where `c` indexes channels and `x y z...` are lattice
axes; `m n` are color indices.

### 3.3 Parallel transport: average over shortest paths

For a neighbor `y = x + Δx`, parallel transport from x to y depends on
which lattice path you choose. A single fixed axis-aligned path (L-Conv's
choice) is gauge-covariant but arbitrary; the **sum over all shortest
lattice paths** is also gauge-covariant (sum of covariants), uses no
direction preferentially, and is strictly more expressive per block. We
adopt the unweighted sum:
```
T_Δx(x)  =  Σ_{P : x → x+Δx, |P| = |Δx|_1}  U_P                 (gauge-covariant)
```
with `|Δx|_1 = Σ_μ |Δx_μ|` the Manhattan length, and `U_P` the ordered
product of links along path P. By linearity it transforms as
`T_Δx(x) → Ω_x · T_Δx(x) · Ω†_{x+Δx}`.

**Compute via DP, not enumeration.** The multinomial
`|Δx|_1! / Π_μ |Δx_μ|!` paths fold into a recursion on the first step:
```
T_Δx(x)  =  Σ_{μ : Δx_μ > 0}  U_{x, μ}  ·  T_{Δx − μ̂}(x + μ̂)
```
Build `T_Δx` over the lattice for all Δx in the positive octant of the
L1-ball, in order of increasing `|Δx|_1`. Cost per new offset: ≤ D+1
matmuls per site, irrespective of path count.

**Octant trick.** Negative-octant transports are obtained from the
positive-octant table by
```
T_{−Δx}(x)  =  dagger( T_Δx(x − Δx) )
```
Cuts compute and memory by ~2^(D+1).

Transport K and V back to x:
```
K̃_{Δx → x, h, a}  =  T_Δx(x) · K_{x + Δx, h, a} · T†_Δx(x)
Ṽ_{Δx → x, h, a}  =  T_Δx(x) · V_{x + Δx, h, a} · T†_Δx(x)
```
Both are locally covariant at x.

**Cost in 4D** (per block, complex64, N_c = 2; memory for storing all
`T_Δx` in the positive octant — both K and V branches share this table):

| Lattice  | R=2 (40 off.) | R=3 (128 off.) | R=4 (320 off.) |
|----------|---------------|----------------|----------------|
| 4⁴       | <1 MB         | <1 MB          | ~5 MB          |
| 8⁴       | ~10 MB        | ~30 MB         | ~80 MB         |
| 16⁴      | ~170 MB       | ~540 MB        | ~1.3 GB        |

Multiply by batch size. **`R = 3` is the practical sweet spot** for
lattices up through 16⁴; `R = 4` is feasible at ≤ 8⁴ but starts pushing
memory at 16⁴. Don't go past `R = 4` — the second block's loop-doubling
rule (Sec. 3.6) cheaply outgrows any single-block reach.

Implementation hint: build `T_Δx` once per batch and reuse across all
H heads and the K, V branches. This is the dominant cost in 4D and
benefits substantially from `torch.compile` and from a single fused
loop ordered by `|Δx|_1`.

### 3.4 Attention score (gauge invariant)

Per head, per neighbor:
```
s_{x → y, h}  =  (1 / √(N_c · d))  ·  Re Tr [  Σ_a  Q†_{x, h, a} · K̃_{y→x, h, a}  ]
```
Both factors transform as `Ω_x · (·) · Ω†_x` at site x; `Re Tr` of their
product is invariant by Tr cyclicity. The `Σ_a` plays the role of the
"sum over per-head feature channels" of standard attention.

**Note on the score form.** `Tr[Q† · K̃]` is the natural generalisation of
the standard attention inner product `q† k` to the matrix setting: it is the
Frobenius inner product between Q and K̃, maximised when the two are aligned
in color space. Q† is used consistently in both score and value path (Sec
3.6). If the score and output paths are decoupled later (a separate `Q'` for
the output), the same `Re Tr[Q†_a · K̃_a]` form carries over unchanged.

**Relative position bias.** Translation equivariance is automatic from
weight sharing; per-offset learned scalars are still useful:
```
s_{x → y, h}  +=  b_h(Δx)          (scalar, real, trainable)
```
Optionally tie `b_h` across the lattice point group's orbit of Δx
(90° rotations + parity) for discrete point-group equivariance at
near-zero parameter cost — see §12.

**On "identity at init" (CASK-style).** The block as a whole acts as the
identity map at init through the residual (Sec 3.8) combined with small
projection weights (Sec 5): when `w^V → 0` the value-path output of Sec
3.6 vanishes, and `W_in + 0 = W_in`. The score's value at init is
irrelevant for this — softmax of near-zero invariants is approximately
uniform, but multiplied into a near-zero value sum it still gives
near-zero. No score-reference subtraction is required, and none is
implemented.

A previous draft motivated this via a subtraction
`s − Re Tr[Σ_a W^id_x · W̃^id_y]` with `w^Q = w^K = identity`; that form
does not type-check (`w^Q ∈ ℂ^{d × C̃}` is not square), and the residual
handles the same job cleanly.

**Physical interpretation: the score is a two-loop correlator.**
Substituting the transport rule from §3.3 (a similarity transform) into
the score gives, schematically per head and per feature `a`:
```
s_{x → y, h, a}  ∝  Re Tr [  Q†_{x, h, a}  ·  T_{Δx}(x)  ·  K_{y, h, a}  ·  T_{Δx}(x)†  ]
```
where `T_{Δx}(x)` is the shortest-path-averaged Wilson line from `x` to
`y = x + Δx`. This is exactly the form of the gauge-invariant **two-loop
correlator** that appears throughout lattice physics — glueball
propagators, string-tension measurements, and Polyakov-loop correlators
are all built from objects of the type
`<Tr[O†(x) · U(x, y) · O(y) · U†(x, y)]>`. The G-Attn block is therefore
asking, at every site and for every neighbor in the L1-ball, "how
correlated is my local loop content with my neighbor's, after
parallel-transporting the neighbor into my frame?"

This interpretation tells us when attention should fire:

- **High score** when `Q_x` and the transported `K_y` are aligned in
  color space — i.e. the two loops carry correlated gauge content.
  In a confined phase, neighboring plaquettes are highly correlated, so
  short-range attention weights should be large.
- **Low score** when the two loops are uncorrelated — e.g. one near a
  topological defect, the other in a smooth bulk region. The trace
  averages near zero and the softmax weight is suppressed.
- **Approximately uniform attention** when `Q` and `K̃` are essentially
  random with respect to each other (the generic init regime). The
  block then behaves like a learned-radius averaging operator, with
  data-dependent corrections built up during training.

The score is therefore *not* an arbitrary learned similarity: it is a
constrained class function of the pair `(Q_x, K̃_y)`, identical in
structure to the correlators that physicists already compute by hand.
Multi-head attention (Sec 3.1) recovers expressivity by giving the
block access to multiple independent such correlators per pair.

**Where this inductive bias should pay off:**

- *Heterogeneous configurations* — instantons, defects, near
  deconfinement — where the relevant neighbors differ from site to
  site. Attention reweights dynamically; a static gauge-equivariant
  convolution kernel cannot.
- *Multi-β / cross-β tasks* — the physical correlation length changes
  with β, and the attention pattern can adapt the effective receptive
  field without retraining a fixed kernel.
- *CASK precedent* (2501.16955) — an analogous Frobenius-style
  attention score on smeared link matrices has already been
  demonstrated to work for SU(3) pure gauge, which establishes that
  the inductive bias is at least not broken.

**Where it may *not* pay off.** A periodic translation-invariant lattice
is structurally homogeneous: there is no `[CLS]`-style token, and every
site has the same neighborhood graph. The learned attention pattern can
in principle collapse to an almost position-independent kernel, in which
case the block degenerates to (a more expensive) gauge-equivariant
convolution. The empirical question for Phase 0 / Phase 3 of the
roadmap is whether *data-dependent* reweighting actually beats a static
kernel of the same receptive field.

### 3.5 Softmax

For each `(x, h)`, normalize over the neighborhood:
```
α_{x → y, h}  =  exp(s_{x → y, h})  /  Σ_{y' ∈ N(x)} exp(s_{x → y', h})
```
Softmax of invariant scalars stays invariant. If training is unstable,
fall back to `tanh(s)` without normalization (CASK-style); the loop-doubling
argument does not depend on the normalization.

### 3.6 Multiplicative value path (L-Bilin baked in)

This is the key departure from a vanilla transformer:

```
W^{(out)}_{x, h, a}  =  Σ_{y ∈ N(x)}  α_{x → y, h}  ·  Q†_{x, h, a}  ·  Ṽ_{y → x, h, a}
```

Both `Q†_{x,h,a}` and `Ṽ_{y→x,h,a}` are covariant matrices at site x, so
their product is covariant at x; the scalar α preserves covariance. The
matrix product `Q† · Ṽ` plays exactly the role of L-Bilin's `W · W'` — every
block doubles the maximum loop length, so `n_blocks ≥ ⌈log₂(N²)⌉` to express
an N×N loop. Q† is the reverse-orientation loop with respect to Q; it is still
a valid covariant matrix and produces a two-loop correlator with the same
loop-doubling property.

Reuse of Q† in both score and output is intentional: Q† appears as the left
factor in both `Tr[Q† · K̃]` and `Q† · Ṽ`, keeping the two paths consistent
and reducing parameter count. If decoupling is needed later, introduce a
fourth projection `Q'` for the output side.

### 3.7 Channel mixing back to C output channels

`W^{(out)}` currently has `H · d` channels. Mix back to `C_out` channels via
a trainable complex linear map (no transport, on-site only):
```
W^{(out, mixed)}_{x, i}  =  Σ_{h, a}  m_{i, h, a}  ·  W^{(out)}_{x, h, a}      m ∈ ℂ
```

### 3.8 Residual connection and L-Act

```
W^{(res)}_{x, i}  =  W^{(in)}_{x, i}  +  W^{(out, mixed)}_{x, i}                  # covariant + covariant
W^{(act)}_{x, i}  =  g_{x, i}(U, W^{(res)})  ·  W^{(res)}_{x, i}
```
Multiplication of a covariant matrix by an invariant scalar preserves
covariance, so any invariant scalar gate works. Two viable realizations:
```
g_relu(W)     =  ReLU( Re Tr [ W ] / N_c )                # L-CNN default
g_softplus(W) =  softplus( Re Tr [ W ] / N_c )            # never vanishes
```

**Caveat for SU(N), N > 1.** `Tr[W]` projects out the trace part of W in
color space. For W-channels that have evolved into approximately traceless
combinations (which can happen after a few blocks — products of plaquettes
are not traceful in general), `Re Tr[W]/N_c ≈ 0` and the ReLU gate kills
the channel. Prefer `g_softplus` once stacking depth exceeds 2–3 blocks,
or initialize with a small positive bias
`g(W) = ReLU(Re Tr[W]/N_c + b₀)`. The pseudocode in Sec. 3.9 uses ReLU for
parity with L-CNN; swap if depth or `N_c` makes channels collapse.

The residual is critical: combined with small init (Sec. 5) it makes each
block start as the identity map, which is what allows stable stacking of
many G-Attn blocks.

### 3.9 Pseudocode for one block

```
def G_Attn_block(U, W_in, weights):
    # 3.2 augment + project
    W_aug = augment(W_in)                                     # (B, 2C+1, *Λ, Nc, Nc)
    Q = einsum("haj,bjxmn->bhaxmn", weights.wQ, W_aug)         # (B, H, d, *Λ, Nc, Nc)
    K = einsum("haj,bjxmn->bhaxmn", weights.wK, W_aug)
    V = einsum("haj,bjxmn->bhaxmn", weights.wV, W_aug)

    # 3.3 build shortest-path-averaged transport sums T_Δx for all
    # Δx in the positive octant of the L1-ball of radius R, via DP
    # ordered by |Δx|_1. Negative-octant transports derive as
    # T_{-Δx}(x) = dagger(T_Δx(x - Δx)).
    T = build_transport_sums(U, R)        # dict: Δx (tuple) -> (B, *Λ, Nc, Nc)
    scores = []
    Vt = []
    for dx in offsets_in_L1_ball(D+1, R):  # all Δx with 0 < |Δx|_1 ≤ R
        T_dx    = transport_sum(T, dx)     # uses octant trick + dagger for negatives
        K_shift = roll_multi(K, shift=tuple(-d for d in dx))  # bring K_{x+Δx} to index x
        V_shift = roll_multi(V, shift=tuple(-d for d in dx))
        Kt      = T_dx @ K_shift @ dagger(T_dx)               # parallel-transported (averaged) K
        Vt_off  = T_dx @ V_shift @ dagger(T_dx)
        # 3.4 score: contract a, take Re Tr
        prod    = einsum("bhaxnm,bhaxnp->bhaxmp", conj(Q), Kt)  # Q† · K̃
        s       = real(trace(prod, axes=(-2,-1))).sum(axis_a) # (B, H, *Λ)
        s      /= sqrt(Nc * d)
        s      += weights.bias[orbit_of(dx)]  # point-group-tied relative position
        scores.append(s)
        Vt.append(Vt_off)

    # 3.5 softmax over offsets
    alpha = softmax(stack(scores, axis=offset), axis=offset)  # (B, H, *Λ, |offsets|)

    # 3.6 multiplicative value path
    W_out_hd = zeros_like(Q)
    for o, dx in enumerate(offsets_in_L1_ball(D+1, R)):
        # Q† · Ṽ is the bilinear; α is a scalar gain
        W_out_hd += alpha[..., o, None, None] * (dagger(Q) @ Vt[o])

    # 3.7 channel mix back to C_out
    W_mix = einsum("iha,bhaxmn->bixmn", weights.mix, W_out_hd)

    # 3.8 residual + L-Act
    W_res = W_in + W_mix
    gate  = relu(real(trace(W_res, axes=(-2,-1))) / Nc)        # (B, C, *Λ)
    W_act = gate[..., None, None] * W_res

    return W_act
```

---

## 4. Full network

```
U                                            # input links
│
├── Plaq, (Poly?)  ─────────────────►  W^(0)     # 2.1, 2.2
│
W^(0)
│
├── G_Attn_block × L                            # 3 — stack L blocks
│
W^(L)
│
├── Trace + (Re, Im)                            # 2.4
│
└── small MLP  (per-site)                        # invariant scalar prediction per site
                                                 # NO global average pool
```

### 4.1 Choosing L

Use `L = ⌈log₂(N²)⌉` for an `N × N` Wilson loop target. For topological
charge density (paper 1's 3+1D test): `L = 4`–`6` with `H = 2`, `d = 4`,
`R = 2`, plaquette + rectangle channels in W^(0).

### 4.2 No global average pool

Emit per-site predictions and aggregate downstream only if the target is a
single global scalar (sum over sites for total topological charge, or for
the action). Per L-CNN supp. Table XVI, GAP destabilizes training in this
regime.

### 4.3 Optional L-Exp head

If the task requires updating links (HMC, normalizing flow, gradient flow),
add an L-Exp layer after the G-Attn stack:
```
U_{x,μ} → exp( i Σ_i β_{μ,i} · [W^(L)_{x,i}]_ah ) · U_{x,μ}
```
where `[X]_ah = (X − X†)/(2i) − Tr(X − X†)/(2i N_c) · 1` (Sec. 1.2 of the
review). Re-run Plaq/Poly after each L-Exp.

---

## 5. Initialization

- All complex projection weights `w^Q, w^K, w^V`: small Gaussian, scale
  `σ ≈ 0.02 / √C̃`. Real and imaginary parts independent.
- Channel mix `m`: same.
- Position bias `b_h(μ, k)`: zero.
- L-Exp `β`: zero (only relevant if L-Exp is used).
- The combination of (small init) + (residual) means each block starts
  approximately as the identity map (Sec. 3.4), so the network as a whole
  is equivalent to passing `W^(0)` straight through to Trace at init. The
  attention pattern is approximately uniform across offsets (softmax of
  near-zero scores; position biases zero). Verify the identity-at-init
  behavior empirically before training.

---

## 6. Training

### 6.1 Loss

- Regression: MSE between prediction and target.
- For tiny labels (topological charge density), pre-multiply labels by a
  fixed constant `C_scale` (e.g. `C_scale = 100` per L-CNN); divide back at
  inference.

### 6.2 Optimizer / schedule

- Optimizer: **AdamW**, weight decay 0.
- Initial LR: `3e-3` for small Wilson loops, `1e-3` for medium, `3e-4` for
  topological charge.
- Batch size: 50.
- Epochs: 20 (small loops) / 100 (large loops); early stopping with
  patience 5–25 on validation MSE.
- Train each architecture with **≥ 5 random seeds**; report median.

### 6.3 Sanity checks during training

- Verify gauge invariance of the loss on a transformed configuration:
  apply a random `Ω_x ∈ SU(N_c)` to U and W, the loss must be identical to
  machine precision (single precision: ~1e-6; double: ~1e-14).
- Check that with all weights at init the prediction equals the
  Trace-of-W^(0) baseline (residual identity behavior).

### 6.4 Parameter counting for cross-architecture comparison

When comparing G-GAT against L-CNN or other real-valued baselines at
*matched parameter count*, count each complex weight as **two real
parameters**. The projection weights `w^Q, w^K, w^V ∈ ℂ^{d × C̃}` and the
channel-mix `m ∈ ℂ^{C_out × H × d}` are complex; failing to double their
count silently gives G-GAT 2× the capacity of the supposed match. The
position biases `b_h(μ, k) ∈ ℝ` count as one each.

---

## 7. Validation: gauge invariance tests

Required for any thesis-level claim of gauge invariance.

### 7.1 Random gauge transformation

For `α > 0`, sample `χ^a_x ~ N(0, α²)` for each color generator `a` and
each site `x`. Form
```
Ω_x = exp( i · t^a · χ^a_x )    (sum over a)
```
where `t^a` are SU(N_c) generators. Apply to `(U, W)`. Measure the relative
prediction drift `|y_pred(U, W) − y_pred(T_Ω U, T_Ω W)| / |y_pred(U, W)|`.

Target: `< 1e−5` in single precision, `< 1e−13` in double.

### 7.2 Gauge-implementation stress test

This is a unit test of the *implementation*, not an adversarial-robustness
test in the ML sense: if the architecture is exactly equivariant the loss
is analytically zero for *all* Ω, and the optimizer only probes the
floating-point ceiling. The point is to find the worst-case Ω so that
implementation bugs which only show up at a thin slice of gauge space are
not missed.

Parametrize `Ω_x = exp(i t^a ρ^a_x)` with trainable `ρ^a_x ∈ ℝ`. Optimize
ρ via AdamW (lr 1e−2, ~200 steps) to **maximize**
```
L_stress  =  | y_pred(U, W) − y_pred(T_Ω(ρ) U, T_Ω(ρ) W) |
```
A correctly-implemented G-GAT must keep `L_stress` bounded by
floating-point noise. Any larger drift is a bug — almost certainly a
missed dagger or a non-axis-aligned transport path.

---

## 8. Universality argument (for thesis)

Each G-Attn block produces, at site x, matrices of the form
```
α(invariant) · Q†_x · U_P · K^or^V_y · U†_P
```
i.e. a parallel-transported, attention-weighted bilinear. By the same
induction as L-CNN's Sec. 3.2 of the Letter:

- One block: with K=R=1 and unit kernels, `Q · Ṽ` produces all 1×k loops up
  to k = R, attention-weighted.
- Stacking L blocks: max loop length doubles each block (Eq. 15 supp.),
  reaching `2^L` after L blocks.
- L-Act adds non-linearity in the trace channel, making the post-block
  output an arbitrary non-linear function of the loops accessible at that
  depth.
- Combined with Plaq + Poly preprocessing, the network can in principle
  represent any non-linear function of any Wilson / Polyakov loop in its
  receptive field.

Quote this argument by reduction to L-CNN universality + softmax /
tanh-attention as a special case of L-Act gating.

**Per-block loop content.** Because parallel transport is averaged over
**all** shortest lattice paths in the L1-ball of radius R (Sec. 3.3), a
single G-Attn block already produces W-channels that are linear
combinations of every shortest-path loop closing within `|Δx|_1 ≤ R` of
x — including all multi-axis and "corner" loops, not just rectangular
ones. The `|Δx|_1! / Π_μ |Δx_μ|!` multinomial paths per offset all
contribute (the DP folds them transparently). This recovers, per block,
the non-axis-aligned loop content that L-CNN's axis-aligned L-Conv only
reaches by stacking, and gives a strictly richer loop *shape* set per
block than CASK's `s = 1..R` extended staples (which fix one ν direction
per staple). Loop-doubling per block still holds via the bilinear `Q · Ṽ`
product (Sec. 3.6).

---

## 9. Reference test problems

Replicate L-CNN's benchmarks first to confirm the implementation is sound:

### 9.1 Wilson loop regression (1+1D, SU(2))

- Lattice `8 × 8`, β ∈ {0.1, ..., 6.0}, 10⁴ train / 10³ val / 10³ test.
- Target: `(1/N_c) Re Tr [ U^{(m×n)} ]` for `(m,n) ∈ {(1,2), (2,2), (4,4)}`.
- Test generalization on `16²`, `32²`, `64²` *without retraining*.
- Pass criterion: MSE within 1× of L-CNN at matched parameter count
  (L-CNN: ~1e−8 to 1e−7 across these tasks). Beat baseline CNN by ≥ 4
  orders of magnitude.

### 9.2 Topological charge density (3+1D, SU(2))

- Lattice `4 × 8³`, train; test up to `8 × 16³`.
- Target: plaquette discretization (Eq. 13 of review).
- Combine with Wilson flow at inference; recover near-integer global Q_P.

### 9.3 Gauge-implementation stress test (trained model)

- Run Sec. 7.2 on the trained `W^(1×2)` model. Target: drift < 1e−5
  (single) or < 1e−13 (double).

---

## 10. Implementation checklist

Build in this order. Verify each step before moving on.

1. Lattice utilities: roll-with-PBC, dagger, and `build_transport_sums(U, R)`
   — a DP routine that materialises the shortest-path-averaged transport
   `T_Δx(x)` for every Δx in the positive octant of the L1-ball of radius
   R (Sec. 3.3), in order of increasing `|Δx|_1`. Verify against brute-force
   path enumeration for `|Δx|_1 ≤ 2` (40 offsets in 4D); test the octant
   trick `T_{-Δx}(x) = dagger(T_Δx(x - Δx))` under a random gauge transform.
2. Plaq, Poly, augment, Trace (fixed ops). Test each: apply random Ω, check
   transformation laws hold to machine precision.
3. On-site Q/K/V projection. Test: `Q(T_Ω W̃) = T_Ω Q(W̃)`.
4. Parallel transport: `Ω_x · K_y · Ω†_y → (Ω_x U_P Ω†_y) · K_y · (Ω_y U†_P Ω†_x)`
   = `Ω_x · (U_P K_y U†_P) · Ω†_x`. Test as a unit.
5. Score `Re Tr[Q · K̃]`: gauge invariance unit test.
6. Softmax: numerically stable (subtract max).
7. Multiplicative value path `α · Q · Ṽ`: covariance unit test.
8. Channel mix + residual + L-Act: covariance unit test on the full block.
9. Trace + MLP head: invariance unit test on the full network.
10. Random + worst-case gauge invariance tests (Sec. 7) on an untrained
    model — must pass before any training.
11. Train on `W^(1×2)` regression; reproduce L-CNN-level MSE.
12. Scale up.

---

## 11. PyTorch-specific notes

- Use `torch.einsum` for all index contractions; it makes the gauge-equivariance
  manifest in the code.
- For complex autograd issues, split tensors into a final dim of size 2
  (real, imag) and provide custom matmul/dagger ops; benchmarks before
  committing to this — recent PyTorch versions support complex autograd for
  most ops.
- Cache parallel-transport products `U_P(μ, k)` per batch — they are reused
  across heads and across the K and V branches.
- Use `torch.compile` on the per-block forward; the einsum-heavy kernel
  benefits substantially.
- Precision: train in float32, validate gauge invariance in float64 (cast
  the model + a single batch).

---

## 12. Variants worth trying after the baseline works

- **Decoupled output query** `Q'` (separate from score Q). +33% projection
  params, may help when the score and bilinear want different feature mixes.
- **CASK-style extended staples** as an alternative kernel: replace the
  shortest-path-averaged transport by 1×s extended staples (one ν per
  staple). Strictly less expressive per block than the L1-ball path
  average (which mixes ν directions automatically), but useful as an
  ablation since CASK is the closest published baseline.
- **Single fixed shortest path** (L-CNN-style): replace `T_Δx` by one
  chosen path per offset. Cheaper but breaks the discrete point-group
  symmetry of the transport. Run as an ablation to isolate the
  contribution of path-averaging vs the rest of the architecture.
- **Tanh score** instead of softmax: sometimes more stable on small lattices.
- **Multi-block residual stream à la pre-norm transformer**: separate the
  attention residual from the L-Act residual.
- **Edge-feature attention**: condition the per-offset bias `b_h(μ, k)` on
  invariants of the local plaquette as well as `(μ, k)`.
- **Point-group equivariance** (90° rotations + reflections of the
  hypercubic lattice). Translation equivariance is automatic from weight
  sharing; the discrete rotational/reflection group is not. Tie
  `b_h(Δx)` across the lattice point group's orbits of Δx (e.g. all 8
  axis-aligned `|Δx|_1 = 1` offsets share one weight; all 24 corner
  `|Δx|_1 = 2` offsets share one; etc.). Expected to improve MSE on
  rotation-symmetric targets (Wilson loops, topological charge) at near-
  zero parameter cost. Cheap to try after Sec. 9.1.

Defer all of these until Sec. 9.1 passes.
