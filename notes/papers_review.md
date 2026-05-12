# Gauge-invariant / Gauge-covariant Neural Networks for Lattice Gauge Theory

This document summarizes three approaches to building neural networks that
respect lattice gauge symmetry. The goal is to provide enough architectural
detail (layer-by-layer math, inputs, outputs, training scheme, results) for
another LLM to be able to reimplement the methods.

The three papers covered:

1. **L-CNN** — Favoni, Ipp, Müller, Schuh, *Preserving gauge invariance in
   neural networks* (arXiv:2112.11239, proceedings; full paper 2012.12901). A
   gauge-equivariant **convolutional** architecture.
2. **CASK** — Nagai, Ohno, Tomiya, *CASK: A Gauge Covariant Transformer for
   Lattice Gauge Theory* (arXiv:2501.16955). A gauge-covariant **transformer**
   built on top of (3).
3. **Gauge covariant neural network** — Nagai, Tomiya,
   *Gauge covariant neural network for quarks and gluons* (arXiv:2103.11965).
   A trainable **stout-smearing / residual flow** architecture. CASK is a
   transformer extension of this network.

All three are designed for SU(N_c) lattice gauge theory in D+1 dimensions.

---

## 0. Common background (lattice gauge theory primer)

You need these objects in any implementation:

- **Lattice** Λ: hypercubic, with periodic boundary conditions, dimension D+1
  (paper 1 uses D=1 (2D lattice) for tests; paper 3 uses 4D; CASK uses 4⁴ at
  β=2.7).
- **Link variables** U_{x,μ} ∈ SU(N_c), defined on each lattice edge from x to
  x+μ. Inverse link: U_{x+μ, -μ} = U†_{x,μ}.
- **Gauge transformation**: choose Ω_x ∈ SU(N_c) at each site. Links transform
  as
  ```
  U_{x,μ}  →  Ω_x  U_{x,μ}  Ω†_{x+μ}                  (covariant)
  ```
- **Locally-transforming field** W_{x,i} ∈ C^{N_c × N_c} (a "feature matrix"
  living at a single site, in some channel i):
  ```
  W_{x,i}  →  Ω_x  W_{x,i}  Ω†_x                      (covariant, same site)
  ```
- **Wilson line** along a path P from x: product of links along P. Transforms
  as Ω_x · WilsonLine · Ω†_endpoint.
- **Plaquette** (smallest closed loop, 1×1):
  ```
  U_{x,μν} = U_{x,μ} U_{x+μ,ν} U†_{x+μ+ν,μ-translated} U†_{x,ν}
  ```
  Transforms covariantly at site x: U_{x,μν} → Ω_x U_{x,μν} Ω†_x.
- **Wilson loop** (any closed path) is gauge-covariant; its **trace** is gauge
  **invariant** — this is how you produce invariant scalars from covariant
  matrices.
- **Polyakov loop**: closed loop wrapping the periodic boundary; needed if you
  want non-contractible loops as input.
- **Staple** around link U_{x,μ}: sum of "C-shaped" 3-link products such that
  staple · U_{x,μ}† = plaquette. Used in smearing.

Key invariance recipe used everywhere:
- Build *covariant* features (matrices that transform like U_{x,μ} or W_{x,i}).
- For an *invariant* output, take traces (or `Re Tr`) at the end.

---

## 1. L-CNN (Favoni, Ipp, Müller, Schuh — full paper 2012.12901)

**Idea.** Generalize CNNs so the convolution kernel uses parallel transport
via link variables. The framework is built from a small set of equivariant
"primitive" layers (Plaq, Poly, L-Conv, L-Bilin, L-Act, L-Exp, Trace) which
can be stacked freely. The intended pipeline is:

```
U  ─Plaq, Poly─►  (U, W)  ─[L-Conv | L-Bilin | L-Act | L-Exp]^L─►  (U', W')
                                                                       │
                                                                       ▼ Trace
                                                              gauge-invariant scalars
                                                                       │
                                                                       ▼
                                                                 MLP / CNN head
```

### 1.1 Data point

A data point is a tuple `(U, W)`:

- `U = {U_{x,μ}}` — gauge links.  Transform non-locally (Eq. 1):
  `U_{x,μ} → Ω_x U_{x,μ} Ω†_{x+μ}`.
- `W = {W_{x,i}}` with `W_{x,i} ∈ C^{N_c × N_c}`, channel index `i = 1..N_ch`.
  Transform locally (Eq. 4): `W_{x,i} → Ω_x W_{x,i} Ω†_x`.

A function `f` on `(U, W)` is **gauge equivariant** iff
`T_Ω f(U,W) = f(T_Ω U, T_Ω W)` and **gauge invariant** iff
`f(T_Ω U, T_Ω W) = f(U, W)`. Every layer below is one or the other.

For storage on a `N_t × N_s^D` lattice with SU(N_c) the input array has size
`N_input = 2 N_c² · N_t · N_s^D · ((D+1) + (D+1)D/2)` real numbers (links +
all positively-oriented plaquettes). Datasets grow fast — paper uses HDF5.

### 1.2 Primitive layers (full set from the paper)

#### Plaq (preprocessing)

Given links `U`, computes every 1×1 plaquette
`U_{x,μν} = U_{x,μ} U_{x+μ,ν} U†_{x+ν,μ} U†_{x,ν}` (Eq. 3) and writes them as
extra channels of W. Restrict to `μ < ν` to avoid storing both orientations
(the dagger is a free Hermitian conjugate). For SU(N_c) the resulting W has
`(D+1)D/2` channels per site.

#### Poly (preprocessing)

Computes every straight Polyakov loop at each site (Eq. 11):
```
L_{x,μ}(U) = U_{x,μ} U_{x+μ,μ} U_{x+2μ,μ} ... U_{x−μ,μ}
```
i.e. the product of all links wrapping the periodic boundary along direction
μ starting at x. Adds them to W. **Required if you want to express
non-contractible loops** — L-Conv+L-Bilin alone can never produce them from
purely contractible inputs.

#### L-Conv (Eq. 5)

```
W_{x,i}  →  Σ_{j, μ, k}  ω_{i,j,μ,k}  ·  U_{x, k·μ}  ·  W_{x+k·μ, j}  ·  U†_{x, k·μ}
```
Trainable weights `ω_{i,j,μ,k} ∈ C`. Indices: `1 ≤ i ≤ N_ch,out`,
`1 ≤ j ≤ N_ch,in`, `0 ≤ μ ≤ D`, `−K ≤ k ≤ K` (kernel half-size K).

Important caveat noted in the paper: the parallel transport between two
sites is **path-dependent**; on curved-manifold gauge networks the geodesic
is the natural choice, but here L-Conv uses **only the straight axis-aligned
path**. This is why convolutions are restricted to a single direction μ at a
time; cross-axis 2D patches would be ambiguous unless you fix a path
convention. Variants discussed: add a bias channel; use sparser dilated
convolutions.

The implementation in the paper additionally restricts to **positive shifts**
`0 ≤ k ≤ K` (negative shifts are subsumed by Hermitian-conjugate channel
augmentation). Empirically that simplifies training without losing
expressivity in practice.

#### L-Bilin (Eq. 6)

```
W_{x,i}  →  Σ_{j, k}  α_{i,j,k}  ·  W_{x,j}  ·  W'_{x,k}
```
Two locally transforming inputs `W, W'` at the same site, `α ∈ C`. Standard
augmentation: enlarge both `W` and `W'` with the unit element `1` and all
Hermitian conjugates,
```
(W_{x,1}, ..., W_{x,N_ch})  →  (1, W_{x,1}, ..., W_{x,N_ch}, W†_{x,1}, ..., W†_{x,N_ch})
```
This makes L-Bilin general enough to act as a **residual module** (set one
factor to 1) and to express loops with reversed orientation.

#### L-CB (combined L-Conv + L-Bilin) — actual implementation block

The reference PyTorch code merges L-Conv and L-Bilin into a single layer
`L-CB(K, N_in, N_out)`. Pseudocode:

```
W'_{x+k·μ, j} = U_{x, k·μ} W_{x+k·μ, j} U†_{x, k·μ}    # parallel-transport (L-Conv part, Eq. 11 supp.)
W_{x,i}       = Σ_{j, j', k}  α_{i, j, j', k}  W_{x,j}  W'_{x+k·μ, j'}     # bilinear with transported (Eq. 12 supp.)
```
Augmentation by 1 and W† is applied to both factors before the bilinear, so
L-CB also subsumes the residual / linear / bias paths. The paper uses this
combined block because *every* L-Conv layer in their networks is followed by
an L-Bilin anyway — single-block notation simplifies counting and
backprop.

A useful estimate from this: starting from 1×1 plaquettes, every L-CB
**doubles the maximum loop area**. So to express an `N × N` loop you need
```
n_layers ≥ ⌈ log₂(N²) ⌉                              (Eq. 15 supp.)
```
e.g. `n=2` for 1×2, `n=3` for 2×2, `n=5` for 4×4.

#### L-Act — gauge-equivariant activation (Eq. 7)

```
W_{x,i}  →  g_{x,i}(U, W)  ·  W_{x,i}
```
where `g_{x,i}` is **any scalar-valued, gauge-invariant function** of the
data (a number per site/channel). Multiplying a covariant matrix by an
invariant scalar preserves covariance.

Concrete equivariant ReLU realization the paper highlights:
```
g_{x,i}(U, W)  =  ReLU( Re Tr [ W_{x,i} ] )
```
i.e. take the trace (gauge invariant), apply ReLU (or any pointwise
nonlinearity), use the result as a per-site, per-channel gain. In general
`g` may depend on values at any site and on additional trainable parameters.

This is the **answer to "do we need L-Bilin if we have nonlinearities"**:
even with L-Act the network is still missing the at-site matrix product
W·W'. L-Act can only modulate magnitudes — it cannot synthesize new loops.
You need both: L-Bilin to grow loops, L-Act for non-linear functions of
loops.

#### L-Exp — equivariant link update (Eq. 8 / 9)

Modifies the **links** themselves:
```
U_{x,μ}  →  ε_{x,μ}  ·  U_{x,μ}      with    ε_{x,μ} ∈ SU(N_c) covariant
```
Concrete realization: build ε from the W variables via the matrix
exponential
```
ε_{x,μ}(W) = exp( i Σ_i β_{μ,i} · [W_{x,i}]_ah )
```
where `[X]_ah = (X − X†)/(2i) − Tr(X − X†)/(2i N_c) · 1` is the
traceless-anti-Hermitian projector and `β_{μ,i} ∈ R` are trainable. The
projection guarantees that ε remains in SU(N_c) (so unitarity and unit
determinant of the updated link are preserved).

**Why you need L-Exp.** With L-Conv, L-Bilin, L-Act, Trace alone the input
links U are never updated — the network produces only invariant readouts.
For tasks that require returning a transformed configuration (classical
field evolution, gradient/Wilson flow, normalizing flows for sampling) you
must update U, and L-Exp is the equivariant way to do it. After an L-Exp,
re-run Plaq and Poly so the W-channels are consistent with the new links.

#### Trace (Eq. 10) — the only invariant layer

```
T_{x,i}(U, W) = Tr [ W_{x,i} ]
```
Yields a complex scalar per site/channel that is gauge **invariant** (cyclic
property). After Trace, feed real and imaginary parts (or just `Re Tr`) into
a standard CNN/MLP head — at that point gauge symmetry is no longer at
risk.

### 1.3 Universality (Sec. 3.2 + Fig. 2 of Letter)

**Theorem.** Stacks of L-Conv + L-Bilin starting from `(U, plaquettes)` can
produce any **contractible** Wilson loop on the lattice as a linear
combination in W. With Polyakov-loop inputs (Poly preprocessing) the same is
true for non-contractible loops, so together they span the full Wilson loop
basis.

**Proof sketch.** Induction on loop area:
- An arbitrary contractible loop of area n tessellates into a loop of
  area n−1 plus a parallel-transported plaquette along the boundary
  (Fig. 2 of the Letter).
- L-Conv produces the transported plaquette (one transport per L-Conv
  layer, so a long transport needs several stacked L-Conv).
- L-Bilin multiplies the (n−1)-loop and the transported plaquette into the
  full n-loop.

Together with the universal-approximation theorem for deep CNNs, adding
L-Act lets the network represent arbitrary **non-linear** functions of
arbitrary Wilson loops — i.e. any gauge-invariant function on the lattice
that depends on the gauge connection.

### 1.4 Loop counting (combinatorial expressivity, Sec. VI.B supp.)

The number of distinct **closed random walks of length L** on a D-dim
lattice is `(2D)^L · P_return(L)`. Closed-form expressions (supp. Eqs. 16–19,
even L only — odd L cannot return):
```
D=1:  L! / [(L/2)!]²
D=2:  Σ_{i=0..L/2}  L! / (i!² · ((L/2 − i)!)²)        =  ( L choose L/2 )²
D=3:  Σ_{i,j} L! / (i!² j!² ((L/2 − i − j)!)²)
D=4:  Σ_{i,j,k} L! / (i!² j!² k!² ((L/2 − i − j − k)!)²)
```
Removing trivial back-and-forth "appendices" (which collapse on link
configurations, since `U U† = 1`) leaves "untraced" loops. Tracing
identifies cyclic shifts. In 2D, length 8 has 312 closed walks → 28 traced
distinct loops; in 4D, length 8 has 190,120 → 3,624 traced loops.

The paper tabulates how many of these distinct traced loops are actually
covered by each architecture in Table V (1+1D) and Table VI (3+1D). Key
takeaways:
- Coverage grows roughly *quadratically* in the number of channels per L-CB
  layer (because L-Bilin multiplies channels pairwise).
- "Small" L-CNN architectures cover several hundred distinct loops; "Large"
  cover tens of thousands. With ≥3 L-CB layers, ~5×10⁶ distinct loops are
  covered.
- Restricting to positive shifts only (k ≥ 0) excludes loops whose starting
  point lies in a corner outside the kernel reach; full coverage requires
  either both signs or one extra L-CB layer.

### 1.5 Test problems (Sec. IV / V + supp.)

#### A. Wilson-loop regression in 1+1D

- Lattice: 1+1D (D=1), SU(2). Train on 8·8, validate on 8·8, test on
  8·8, 16·16, 32·32, 64·64.
- Targets: `W^{(m×n)} = (1/N_c) Re Tr [ U^{(m×n)}_{x,01} ]` for
  `(m,n) = (1,1), (1,2), (2,2), (4,4)` — the 1×1 case is trivial because the
  label is already in the input as a plaquette.
- Coupling β ∈ {0.1, ..., 6.0} sampled uniformly with `N_β = 10` steps; 10⁴
  training, 10³ validation, 10³ test examples per lattice size.

#### B. Topological-charge density in 3+1D

- Lattice: 3+1D, SU(2). Train on 4·8³, test up to 8·16³.
- Target: plaquette discretization (Eq. 13)
  ```
  q^plaq_x = ε_{μνρσ} / (32π²) · Tr[ (U_{x,μν} − U†_{x,μν}) / (2i) ·
                                     (U_{x,ρσ} − U†_{x,ρσ}) / (2i) ]
  ```
  This is gauge invariant, real, and approximates the continuum topological
  charge density. Network also tested on **Wilson-flowed** configurations
  (without retraining) to recover near-integer global Q_P.

### 1.6 Training procedure

- Framework: **PyTorch** + PyTorch Lightning. PyTorch's incomplete complex
  support at the time forced complex N_c×N_c matrices to be split into
  real/imag and processed as `2·N_c²` real channels via batched
  `torch.einsum`.
- Loss: MSE.
- Optimizer: **AdamW**, zero weight decay.
- Learning rate: 3·10⁻³ for `W^{(1×1)}`, `W^{(1×2)}`; 1·10⁻³ for `W^{(2×2)}`,
  `W^{(4×4)}`; 3·10⁻⁴ for the topological charge `Q_P` task.
- Batch size: 50; max 20 epochs (small loops) or 100 epochs (large loops);
  early stopping with patience 5–25 based on validation loss.
- Each architecture trained with **10 random seeds** to form ensembles
  (5 for 3+1D); medians reported. Total: 2680 baseline models, ~125 L-CNN
  models.
- Label scaling: a uniform multiplicative `C > 0` is allowed on labels
  (`q̃_x^plaq = C q_x^plaq` with `C = 100` for `Q_P`, `C = 1` for Wilson
  loops). This is just a "whitening" trick — important because the
  topological-charge-density labels are otherwise tiny.
- Crucially, **L-CNNs do *not* use a global average pooling layer** at the
  end; they emit a prediction per site. Authors found training was much more
  stable that way; baselines, in contrast, *require* GAP for translation
  invariance and converge much better with it (Table XVI of supp.).

Hardware: training the largest 1+1D L-CNN ≈ 36 s/epoch and ≤ 1 GB GPU on a
Titan V; the largest 3+1D L-CNN ≈ 13 min/epoch, ≈ 8 GB, ≈ 11 h for 50
epochs.

### 1.7 Quantitative results

#### Wilson-loop MSE (1+1D, median over ensemble; lower is better)

| Loop      | Best baseline CNN (8·8) | Best L-CNN (8·8) | L-CNN at 64·64 |
|-----------|------------------------|------------------|----------------|
| W^(1×1)   | ~7·10⁻⁹ (trivial)      | 2.19·10⁻⁸        | 2.19·10⁻⁸      |
| W^(1×2)   | 2.0·10⁻³               | 2.1·10⁻⁹         | (generalizes)  |
| W^(2×2)   | 4.0·10⁻³               | 1.1·10⁻⁸         | (generalizes)  |
| W^(4×4)   | 4.2·10⁻³ (≈ collapses to mean) | 1.4·10⁻⁷ | 3.10·10⁻⁸     |

Takeaways:
- L-CNN beats baseline by **4–6 orders of magnitude** for loops larger than
  1×1.
- For 4×4 loops the baseline collapses entirely to predicting the training
  mean — it can't learn the relationship at all.
- L-CNN MSE is essentially flat across lattice sizes ⇒ near-perfect
  generalization to volumes 64× larger than training.

#### Topological charge in 3+1D (Table XVII)

| Lattice  | Variance of label | L-CNN-S MSE | L-CNN-M MSE |
|----------|------------------|-------------|-------------|
| 4·8³     | 2.91·10⁻⁷         | 3.18·10⁻⁹    | (n/a)       |
| 8·16³    | 1.87·10⁻⁷         | 3.17·10⁻⁹    | (n/a)       |

Train on 4·8³, test up to 8·16³, MSE flat to ~3·10⁻⁹ — about 2 orders of
magnitude below label variance. Combined with Wilson flow (Δτ = 0.005, gauge
cooling) the L-CNN's predictions reproduce the integer-valued global
topological charge `Q_P(τ)` of an 8·24³ flowed configuration to high
accuracy — without ever being trained on flowed data (Fig. 5 of the Letter).

#### Adversarial gauge attacks (Sec. VI.A supp.)

Test gauge-symmetry breaking with two attacks:
- **Random** Ω_x = exp(i t^a χ^a_x) with χ^a normal, amplitude α > 0.
- **Adversarial**: optimize `Ω_x = exp(i t^a ρ^a_x)` over ρ^a via AdamW to
  maximize `|y_pred − y_trans|`.

Numbers (W^(1×2) regression, 1+1D):
- Random gauge transformation: baseline CNN predictions drift by **up to
  16 %**.
- Adversarial: drift up to **79 %** — i.e. an attacker can almost completely
  invalidate baseline predictions.
- L-CNN: drift bounded by **~ 10⁻⁶** (single-precision GPU noise floor).
  In double precision, indistinguishable from machine epsilon.

This is the strongest empirical argument for using equivariant networks in
physics: even a baseline that *appears* gauge-invariant on Monte Carlo data
(because MC samples are not adversarial) is in fact severely brittle.

### 1.8 Architecture catalog (concrete sizes used)

The L-CNN architectures (Table V/VI of supp.) follow the pattern
`L-CB → L-CB → ... → Trace → Linear`. Notation `L-CB(K, N_in, N_out)`.

| Task        | "Small"           | "Medium"          | "Large"        |
|-------------|-------------------|-------------------|----------------|
| W^(1×2)     | L-CB(2,1,2) → Tr → Lin(4,1)  (35 params)   | L-CB(3,1,4) (117) | L-CB(4,1,8) (329) |
| W^(2×2)     | L-CB(2,1,2)→L-CB(2,2,2) (125)  | (1,305)          | (13,521)       |
| W^(4×4)     | 4×L-CB → Tr → Lin (465)  | (4,833)             | (39,905)        |
| W^(2×2) 3+1D| L-CB(2,6,2)→L-CB(2,2,2)→Tr→Lin(4,1) (1,801) | (8,305)  | —              |
| W^(4×4) 3+1D| 4×L-CB→Tr→Lin (2,109) | (14,377)              | —              |
| q^plaq 3+1D | L-CB(2,6,4)→Tr→Lin(8,1) (3,181) | —          | —              |

The smallest L-CNN with 35 parameters already beats the largest 100,000-param
baseline CNN by orders of magnitude on W^(1×2) — the right inductive bias
matters far more than capacity.

### 1.9 Implementation hints

- Keep complex N_c×N_c matrices; if the framework lacks complex autograd,
  split into (Re, Im) and use real `einsum` for matrix products.
- Always include the `1` channel and Hermitian conjugates before bilinear
  combination — gives bias and orientation reversal "for free".
- Build a single `L-CB(K, N_in, N_out)` block (transports along axes →
  bilinear with all local channels) rather than separate Conv/Bilin layers;
  it's what the reference code does.
- Avoid GAP at the end — emit per-site predictions; aggregate only if the
  target is global.
- Implement Plaq, Poly, Trace as fixed (non-trainable) preprocessing /
  postprocessing, and call them again after every L-Exp.
- For tasks with tiny labels (topological density), apply a uniform `C·label`
  scale during training; remove at inference.
- Public reference implementation: <https://gitlab.com/openpixi/lge-cnn>.

---

## 2. Gauge covariant neural network — "trainable stout smearing"
(Nagai & Tomiya, arXiv:2103.11965)

This is the **building block of CASK**, and is independent of L-CNN. The key
insight: **stout smearing already is a gauge-covariant ResNet-like layer**.
Make its parameters trainable, and you get a covariant NN.

### 2.1 Smearing as a residual layer

A gauge-covariant map sends links to links (same shape):
```
U^{(l+1)}_μ(n)  =  N( w1^{(l-1)} · U^{(l-1)}_μ(n)  +  w2^{(l-1)} · G^{θ̄}_{μ,n}[U^{(l-1)}] )
```
where:
- `G^{θ̄}_{μ,n}[U]` is a "filter function" that transforms exactly like a link
  at (n, μ). I.e. it is a sum/product of gauge-covariant building blocks
  (staples, plaquettes, Polyakov loops, etc.).
- `w1, w2` ∈ R are trainable scalars (real).
- `N(·)` is a local activation: e.g. SU(N_c) projection (APE/HYP) or
  the identity (stout). For stout: w1 = w2 = 1, N = identity, and G is built
  from the matrix exponential of a traceless-anti-Hermitian closed-loop sum.

This is a **rank-2 ResNet**: the additive structure is exactly residual, with
the rank-2 tensor (matrix) replacing the usual scalar feature.

### 2.2 Stout-type covariant layer (concrete formulas)

Update rule used in 2103.11965:
```
U^{(l+1)}_μ(n)  =  exp( i · Q^{(l)}_μ(n) ) · U^{(l)}_μ(n)
```
with
```
Q_μ(n) = (i/2) · (Ω†_μ(n) − Ω_μ(n))  −  (i / (2 N_c)) · Tr[ Ω†_μ(n) − Ω_μ(n) ] · 1
```
i.e. Q ∈ su(N_c) (traceless-anti-Hermitian); Ω_μ(n) ∈ SU(N_c) is a sum of
*untraced* closed loops surrounding the link U_μ(n).

A useful trainable parametrization (paper 3, Eq. C1 / Eq. 33):
```
Ω^{(l)}_μ(n) = ρ^{(l)}_plaq · O^plaq_μ(n)
             + ρ^{(l)}_rect · O^rect_μ(n)
             + ρ^{(l)}_poly · O^poly_μ(n)
             + ...
```
where `O^X_μ(n)` are untraced loop operators of type X (plaquette,
rectangle, Polyakov, ...) attached to the link (n,μ), and ρ ∈ R are the
trainable weights. Use one ρ per loop type per layer; optionally tie spatial
weights together for rotational symmetry.

Layers are stacked: U → U^{(1)} → U^{(2)} → ... → U^{(L)} = U^{eff}.

### 2.3 Why it is covariant

Each Ω_μ(n) transforms as Ω_x · Ω_μ(n) · Ω†_{x+μ} (since it is a sum of
closed loops touching the link), the exponential exp(iQ) transforms as
Ω_x · exp(iQ) · Ω†_x at the link's start, and U^{(l+1)}_μ(n) inherits the
correct link-transformation rule by construction.

### 2.4 Gauge-invariant loss

Use any gauge-invariant scalar built from U^{eff}, e.g. a lattice action:
```
L({U^eff}) = Σ_i  | S({U_i}) − S^eff({U^eff_i}) |²    (Eq. 18)
```
or an HMC acceptance-related quantity. In SLHMC (see below) one trains
against the difference between true action and effective action.

### 2.5 Backprop: rank-2 / matrix-valued delta rule

The novel contribution: derivatives are taken with respect to *complex
matrix* variables. Define:
- Matrix derivative: `[∂f/∂A]^i_j ≡ ∂f / ∂A^j_i`  (rank-2).
- Rank-4 tensor for `∂M/∂A`.
- "Star product" `[A ⋆ T]^i_j ≡ Σ_{kl} A^l_k T^k_j ^i_l` linking rank-2 and
  rank-4 tensors.
- A Wirtinger-style treatment for complex matrices, with α ∈ {I, II} indexing
  whether the derivative is w.r.t. A or A†.

Recursion (paper 3, Eqs. 27–29, E5):
```
δ^{(L)}_μ(n) = Σ_α  ( ∂S_θ / ∂U^{(L)α}_μ(n) ) ⋆ ( ∂N[z^{(L)}_μ(n)]_α / ∂z^{(L)}_μ(n) )

δ^{(l)}_{μ,α}(n) = Σ_β  w1^{(l)} δ^{(l+1)}_{μ,β}(n) ⋆ ( ∂N / ∂z^{(l)} )
                 + w2^{(l)} Σ_{μ',m,β} δ^{(l+1)}_{μ',β}(m) ⋆
                                   ( ∂G_{μ',m}(U^{(l)})_β / ∂U^{(l)γ}_μ(n) ) ⋆
                                   ( ∂N[z^{(l)}_μ(n)]_γ / ∂z^{(l)}_μ(n) )
```
The HMC force is exactly `δ^{(0)}_μ(n)`, so backprop and the "smeared force"
formula of Morningstar-Peardon are the *same* construction. This is the
paper's main theoretical observation.

### 2.6 SLHMC demonstration

- 2-color QCD (SU(2)), 4D lattice 4³×4, β=2.7, staggered fermions, m_l = 0.3.
- Effective action `S_θ = S_g[U] + S_f[ϕ, U^{NN}_θ[U]; m_h]` with m_h ∈
  {0.4, 0.5, 0.75, 1.0}, m_h > m_l.
- Network: L=2 stout-type layers, 6 trainable weights (ρ_plaq, ρ_poly,s,
  ρ_poly,4 per layer). Spatial Polyakov weights tied for rotation symmetry.
- Training: SGD, η=1e-7, 1500 trajectories, MSE loss between S and S_θ.
- Result: SLHMC and HMC agree on plaquette (0.7024 vs 0.70257), Polyakov loop,
  and chiral condensate within statistical error.
- Acceptance starts ~0 (loss ~90), reaches usable values after training.

### 2.7 Connection to other concepts (theory bonus)

- **Neural ODE:** treat layer index as continuous time; Eq. (16) becomes
  dU_μ(n)/dt = w2(t) · G^{θ̄(t)}_{μ,n}(U(t)). For Wilson plaquette G this is
  exactly **Lüscher's Wilson flow / gradient flow**.
- **Universality:** since stacking these layers builds rank-2 ResNets with
  rich loop content, they inherit universal-approximation arguments from
  ResNet/MLP literature (small-a expansion shows reduction to a fully
  connected linear layer in U(1)).
- Also called **residual flow** in Abbott et al. 2401.10874.

---

## 3. CASK (Covariant Attention with Stout Kernel) — Nagai, Ohno, Tomiya 2501.16955

### 3.1 Big picture

CASK = **transformer block whose attention matrix is built from gauge-invariant
Wilson loops**, sandwiched on top of stout-smearing-style Q/K/V layers (the
gauge-covariant NN above).

Three innovations vs. a vanilla self-attention block:

1. Q, K, V are not vectors but **smeared link configurations** U^{(Q)},
   U^{(K)}, U^{(V)} — produced by separate gauge-covariant (stout) layers
   acting on the input U.
2. The attention score is the **Frobenius inner product** of two
   gauge-covariant link/Wilson-loop combinations — invariant by cyclicity of
   trace.
3. The output is a **link update** (an exp(iQ) multiplying an input link),
   identical in form to one stout layer, but with the loop coefficients
   replaced by attention scores.

This makes the attention matrix **gauge invariant** and the output **gauge
covariant**, exactly matching the spin-O(3) equivariant transformer of
Nagai-Tomiya 2306.11527 lifted to local gauge symmetry.

### 3.2 Step 1: Q, K, V link fields

Three independent gauge-covariant (stout-type) layers, sharing the same
template but with separate trainable smearing weights ρ:
```
U^{(Q)}_μ(n) = U^{(α)}_μ(n; ρ^{(α)})    with α=Q
U^{(K)}_μ(n) = U^{(α)}_μ(n; ρ^{(α)})    with α=K
U^{(V)}_μ(n) = U^{(α)}_μ(n; ρ^{(α)})    with α=V
```
In the paper's experiments these are single-plaquette stout smearings (the
simplest covariant layer). Each has its own ρ.

### 3.3 Step 2: extended staples (the "kernel")

Define an **extended (1×s) staple** sitting on the link (n, μ), going s
steps along direction ν before closing back:
```
V_{ν, n+μ̂; s}({U})
   =  [ Π_{t=0..s-1}  U_ν(n + μ̂ + t ν̂) ]
     · U†_μ(n + μ̂ + s ν̂)
     · [ Π_{t=0..s-1}  U†_ν(n + (s-1-t) ν̂) ]
```
(Eq. 9 of CASK paper.) For s=1 this is the usual rectangular staple; for
s=2,3 it is an extended rectangular staple.

These staples are gauge-covariant: V → Ω_n · V · Ω†_{n+μ̂}.

### 3.4 Step 3: gauge-invariant attention

Define raw attention score (an s-indexed family) using the Frobenius inner
product:
```
ã_{n,μ,ν,s}  =  Re Tr[ U^{(Q)}_μ(n) · V_{ν,n+μ̂;s}({U^{(K)}}) ]
              − Re Tr[ U_μ(n)       · V_{ν,n+μ̂;s}({U})       ]
```
The second term is the *input* to the l-th CASK layer, included so that with
ρ=0 initialization the block reduces to identity (greedy-layer-wise warmup).

Tangent rescaling (to amplify):
```
a_{n,μ,ν,s}  =  tan( (4 / N_c) · ã_{n,μ,ν,s} )
```

This is a **gauge invariant** real number per (n, μ, ν, s) by Tr cyclicity:
```
Re Tr[ Ω_n A Ω†_n · Ω_n B Ω†_n ] = Re Tr[A B].
```

The attention is **sparse**: in the paper it is computed only for s = 1, 2, 3
(i.e. links connected within 1×1, 1×2, 1×3 rectangular Wilson loops). This
mirrors sparse-transformer / flash-attention philosophy and replaces the
softmax with a cheaper bounded transform — softmax is non-trivial to keep
symmetric.

### 3.5 Step 4: applying attention (covariant output)

Use the V-channel link to build "value staples":
```
C^A_μ(n; {a})  =  Σ_{ν≠μ}  Σ_{s=1..R}  a_{n,μ,ν,s} ·
       (   U^{(V)}_ν(n) · U^{(V)}_μ(n + ν̂) · U^{(V)†}_ν(n + μ̂)
         + U^{(V)†}_ν(n − ν̂) · U^{(V)}_μ(n − ν̂) · U^{(V)}_ν(n − ν̂ + μ̂) )

Ω^A_μ(n; {a})  =  C^A_μ(n; {a}) · U^{(V)†}_μ(n)
```
This is exactly an *attention-weighted sum of staples*, replacing the
constant ρ in stout smearing.

Form the su(N_c) generator (traceless-anti-Hermitian projection):
```
Q^A_μ(n; {a}) = (i/2) ( Ω^A_μ − Ω^A†_μ )
              − (i/(2 N_c)) Tr( Ω^A†_μ − Ω^A_μ ) · 1
```

Apply it to update links (one CASK transformer block):
```
U^{(l+1)}_μ(n) = exp( i Q^A_μ(n; {a_{n,μ,ν,s}}) ) · U^{(l)}_μ(n)
```

This is again a stout-type covariant layer — but with attention-determined
coefficients `a` that are themselves gauge-invariant functions of the entire
configuration. It is therefore both gauge-covariant (output is a link field)
and globally context-aware (attention sees correlations across the lattice
through the staples).

### 3.6 Pseudocode for one CASK block

```
def CASK_block(U):
    U_Q = stout_layer(U, rho_Q)
    U_K = stout_layer(U, rho_K)
    U_V = stout_layer(U, rho_V)

    a = {}                                # (n, μ, ν, s) → real
    for n, mu, nu, s in (lattice_sites x dirs x dirs x [1,2,3]):
        if nu == mu: continue
        V_K = extended_staple(U_K, n+mu_hat, mu, nu, s)
        V_in = extended_staple(U,   n+mu_hat, mu, nu, s)
        a_tilde = Re_Tr( U_Q[n,mu] @ V_K )  -  Re_Tr( U[n,mu] @ V_in )
        a[n,mu,nu,s] = tan( (4/Nc) * a_tilde )

    for n, mu in (lattice_sites x dirs):
        C = sum over (nu != mu, s in [1..R]) of  a[n,mu,nu,s] * staple(U_V, n,mu,nu)
        Omega = C @ dagger(U_V[n,mu])
        Q = traceless_antiherm(Omega)
        U_new[n,mu] = expm(i * Q) @ U[n,mu]

    return U_new
```
Stack multiple blocks → "CASK_n" with n attention blocks.

### 3.7 Why the design choices

- **Frobenius / Tr inner product**: the only natural way to get an O(N²)
  attention from gauge-covariant matrices that is *invariant*.
- **ReLU / tan instead of softmax**: softmax mixes scores in a way that is
  hard to keep symmetric; both choices are explicitly non-negative or
  preserve sign and are cheap.
- **Sparse attention** (s ≤ R, with R=3): for cost reasons. In principle
  all-to-all is fine; the formalism is identical, only s ranges further.
- **Identity at init** (subtracting the input contribution from ã): so that
  ρ ≈ 0 makes the block ≈ identity, enabling greedy layer-wise training and
  stable training of deep stacks.
- **Spin-O(3) precedent**: the architecture is the gauge analogue of an
  earlier equivariant transformer (2306.11527, 2310.13222) where attention
  was the dot product of equivariant block-spin transforms.

### 3.8 Test problem and results

- **Self-Learning HMC (SLHMC)** for SU(2) lattice gauge theory with
  staggered fermions. 4⁴ lattice, β=2.7, target m=0.3, effective m_eff=0.4.
- **Effective fermion action** uses CASK-smeared links instead of thin links;
  CASK absorbs the discrepancy between the effective Dirac operator at m_eff
  and the true one at m.
- Optimizer: Adam.
- Implementation: Gaugefields.jl + LatticeDiracOperators.jl in JuliaQCD.
- **Baseline**: gauge covariant network ("CovNet") of paper 3.

Findings (Fig. 2 of paper):
- All networks reduce loss with training.
- CovNet saturates; **CASK keeps improving** at later epochs and reaches
  lower loss.
- Plaquette histogram from CASK matches the true MC distribution → physics is
  preserved.
- CASK_n with more attention blocks is more expressive than CovNet.

### 3.9 Implementation hints

- Build extended staples once per (n, μ, ν, s) and reuse for both the K and
  the V branch (just feed different smeared links through them).
- `tan` can blow up; `4/N_c` scaling is small enough in practice to keep
  scores bounded, but you should clamp or use `tanh` for stability if needed.
- Initialize ρ ≈ 0 so the first forward pass is close to identity.
- Backprop: use the rank-2 delta rule from paper 3 (Sec. 2.5 of this doc) —
  every layer in CASK is either a stout layer or an attention-weighted stout
  layer.

---

## 4. Side-by-side comparison

| Aspect                  | L-CNN (paper 1)              | Covariant NN (paper 3)        | CASK (paper 2)                                |
|-------------------------|------------------------------|-------------------------------|-----------------------------------------------|
| Architecture            | Convolutional                | ResNet (stout smearing)       | Transformer on top of stout                   |
| Inputs                  | (U, W=plaquettes/Polyakov)   | Links U                       | Links U                                       |
| Outputs                 | Equivariant W → scalar; or updated U via L-Exp | Link field U^{eff}  | Link field U^{eff}                            |
| Layers                  | Plaq, Poly, L-Conv, L-Bilin, L-Act, L-Exp, Trace | Stout-type covariant layer  | Q/K/V stout layers + attention-weighted stout |
| Activation              | L-Act: scalar invariant gain `g(U,W)·W`, e.g. `ReLU(Re Tr W)·W` | none (linear residual + projection) | tan-rescaled invariant attention (replaces softmax) |
| Trainable params        | Complex ω (L-Conv), α (L-Bilin), β (L-Exp) | Real ρ per loop type / layer | Real ρ_Q, ρ_K, ρ_V + smearing weights         |
| Symmetries              | Gauge + lattice trans/rot/refl | Gauge + global trans/rot     | Gauge + lattice symmetries                    |
| Invariance source       | Trace layer at end           | S_g, S_f are gauge-invariant   | Attention via Re Tr; output covariant         |
| Universality            | Arbitrary Wilson loops via L-Conv+L-Bilin (proved); arbitrary non-linear functions of loops with L-Act | Universal as ResNet; expressivity grows with loop types | Inherits CovNet universality + non-local correlations |
| Tested on               | Wilson-loop regression + 3+1D topological charge, SU(2) | SLHMC (SU(2), 4D, dyn. fermions) | SLHMC (SU(2), 4D, dyn. fermions)              |
| Best benchmark vs CNN   | 4–6 orders of magnitude lower MSE; 79% adv. error → ~10⁻⁶ | SLHMC matches HMC observables to 10⁻³ | SLHMC loss < CovNet, doesn't saturate         |
| Practical strength      | Invariant *observables*, transferable across lattice sizes | Trainable smearing & effective actions | Best expressivity for SLHMC effective actions  |

---

## 5. Practical recipe for building your own gauge-invariant NN

If your thesis problem is **regression of an invariant observable**
(action, Wilson loop, topological charge, plaquette mean, etc.) on an SU(N_c)
lattice configuration:

1. Start from L-CNN. Preprocessing: links → Plaq (+ Poly if non-contractible
   loops matter).
2. Stack a few `L-CB(K, N_in, N_out)` blocks. Rule of thumb:
   `n_layers ≥ ⌈log₂(N²)⌉` for an `N × N` target loop.
3. Optionally insert an L-Act between L-CB blocks if you need non-linear
   functions of loops (e.g. `ReLU(Re Tr W)·W`).
4. Trace → small linear layer per site. Don't add a global average pool —
   emit per-site predictions and aggregate downstream if needed.
5. Loss: MSE; optimizer AdamW with lr 1e-3 → 3e-3 for loops, 3e-4 for
   topological-charge density; rescale tiny labels by `C` (e.g. C=100 for
   q^plaq).
6. Validate gauge invariance with random *and* adversarial gauge attacks
   (parametrize Ω = exp(i t^a ρ^a) and optimize ρ to maximize
   `|y_pred − y_trans|`); the L-CNN should drift only at floating-point
   precision.

If you need to **emit a transformed configuration** (classical evolution,
gradient/Wilson flow, normalizing flow, preconditioner inside a solver),
add an **L-Exp** layer: `U → exp(i Σ β [W]_ah) U`. After every L-Exp,
re-run Plaq/Poly so the W-channels are consistent with the new links.

If your problem is **building a covariant *map* between configurations**
(surrogate action for SLHMC, normalizing flow, force shaping):

1. Use the gauge-covariant ResNet of paper 3 as the backbone. Trainable ρ
   weights per loop type per layer; output is U^{eff} = ∏_l exp(iQ^{(l)}) U.
2. Build the loss from gauge-invariant quantities of the *output* links
   (action difference, KL, force matching, MSE of S vs S_θ).
3. Train with the rank-2 delta rule (or autodiff frameworks that already
   support complex matrix Wirtinger derivatives, e.g. Zygote.jl in JuliaQCD).

If you need **long-range correlations** (confining / near-critical regime):

1. Stack CASK blocks: replace constant ρ in some smearing layers by
   attention scores `a_{n,μ,ν,s}`. Q, K, V each have their own stout layer.
2. Use sparse attention with R = 2..3 to keep cost manageable; expand if
   physics demands. `tanh` is a safer rescaling than `tan`.

**Transformer vs GNN framing.** For a transformer-flavored thesis: CASK is
the most directly publishable structure today, and it builds cleanly on the
covariant-NN framework. For a GNN framing: lattice gauge theory is naturally
a graph (sites = nodes, links = edges with SU(N_c) features), and L-CNN's
L-Conv is exactly a message-passing layer with parallel transport on edges
(message = U_{xy} m_y U†_{xy}), L-Bilin is a node-wise edge-feature product,
L-Act is a node-wise gauge-invariant gating, and L-Exp updates edge features
from incident-node messages. All three papers fit into a unified
"gauge-equivariant message passing" framework — useful framing for a thesis
that wants to bridge GDL and lattice physics.

### 5.1 What the L-CNN paper proved is *transferable*

A particularly valuable empirical result for thesis purposes: L-CNNs trained
on small lattices (8·8 in 1+1D, 4·8³ in 3+1D) **generalize without
retraining** to:
- Lattices up to 8× larger in each direction with no MSE degradation —
  because the architecture has no global pooling and is built from local
  operators sharing weights across sites.
- **Different physical regimes** of the same problem: networks trained on
  Monte-Carlo configurations correctly predict topological charge on
  Wilson-flowed configurations they never saw.

That last point is exceptional in machine learning: it is a strong sign
that the network has learned the *physical* relationship rather than
memorizing the data manifold, and is the kind of result that justifies the
structural-prior approach in a thesis defense.

---

## 6. References (arXiv IDs you'll cite)

- **L-CNN:** 2112.11239 (proceedings) and 2012.12901 (full paper +
  supplementary). Authors: Favoni, Ipp, Müller, Schuh.
- **Gauge covariant NN:** 2103.11965. Authors: Nagai, Tomiya. Code:
  JuliaQCD / LatticeQCD.jl.
- **CASK:** 2501.16955. Authors: Nagai, Ohno, Tomiya. Code: JuliaQCD,
  Gaugefields.jl, LatticeDiracOperators.jl.
- Related: equivariant flow models 2003.06413, 2008.05456, 2106.05934;
  gauge-equivariant CNNs on manifolds 1902.04615; geometric DL 2104.13478;
  equivariant continuous flows 2110.02673; ML preconditioner for Dirac
  2302.05419, 2304.10438; fixed-point action 2401.06481; equivariant
  transformer for spins 2306.11527, 2310.13222.
