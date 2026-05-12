# Roadmap — Physics Results with the Gauge-Equivariant Graph Attention Network

This document maps a progressive sequence of results, from sanity checks on
Z₂ to potentially novel research directions. Each phase has:

- **Setup**: lattice, gauge group, β range, what configurations to generate.
- **Tasks**: what the network is trained to do.
- **Observables / comparisons**: what to measure and what to compare against
  (analytic / Monte Carlo / prior ML papers).
- **Pass criteria**: concrete numerical targets.
- **Estimated time** for a master's-level student already comfortable with
  PyTorch and lattice basics.
- **Pitfalls** specific to that phase.

The architecture is the **G-GAT** of `architecture.md` (gauge-equivariant
graph attention with multiplicative value path). All phases reuse the same
codebase; the gauge group, dimension, and head are the only things that
change.

For Monte Carlo data generation you can either write your own (Z₂ and U(1)
are a few hundred lines; pure SU(N_c) is a standard heat-bath / overrelaxation
pseudocode) or use existing tools: **JuliaQCD** (`Gaugefields.jl`,
`LatticeDiracOperators.jl`) for SLHMC, **lge-cnn** (gitlab.com/openpixi)
for L-CNN-compatible data, **Hipparchus / openQCD / Chroma** for serious
SU(3) production. For a master's thesis, JuliaQCD + a small custom Python
MC for Z₂/U(1) is the lightest combination.

---

## Phase 0 — Implementation validation on Z₂ in 2D
**Time: 2–3 weeks**

### 0.1 Setup
- Gauge group **Z₂**: links `U_{x,μ} ∈ {+1, −1}` represented as 1×1 complex
  matrices to keep the architecture's tensor shapes intact.
- 2D lattice, sizes `L = 4, 8, 16, 32`, periodic.
- Action: `S = −β Σ_p U_p` with plaquette `U_p = U_{x,0} U_{x+0̂,1} U_{x+1̂,0} U_{x,1}`.
- 2D Z₂ has **no phase transition** — this is intentional. The point is
  pure code validation, not physics.
- Generate `10^4` configurations per (L, β) at β ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
  with single-site Metropolis. Cheap.

### 0.2 Tasks
1. **Plaquette regression**: predict `<U_p>` from a configuration. The label
   is just the configuration's plaquette mean; the network should achieve
   ≈ machine-precision MSE because the input *contains* the answer trivially.
2. **Wilson-loop regression**: predict `(1/N_c) Re Tr W^{(2×2)}` per site;
   then the configuration mean.
3. **Gauge-invariance unit tests**: random + adversarial Ω attacks (§7 of
   `architecture.md`).

### 0.3 Pass criteria
- Plaquette MSE ≤ 10⁻¹⁰ (single precision).
- 2×2 Wilson loop MSE ≤ 10⁻⁸.
- Random Ω drift < 10⁻⁵; adversarial drift < 10⁻⁵.
- Forward pass on 32² in < 10 ms on a laptop GPU.

### 0.4 Pitfalls
- N_c = 1 makes the trace and the matrix coincide. Make sure your tensor
  shapes still carry the (1, 1) "color" axes — otherwise you'll have to
  refactor when moving to SU(2).
- Z₂ links commute with everything → many of your covariance unit tests
  will pass *even with wrong daggers*. Re-run all tests in Phase 1 with a
  non-abelian-style generic Ω.

---

## Phase 1 — 3D Z₂ gauge: first real physics
**Time: 3–4 weeks**

3D Z₂ pure gauge theory is exactly **dual to the 3D Ising model**
(Wegner 1971): the deconfinement transition of Z₂ gauge maps to the
ferromagnetic transition of Ising. This gives you analytical predictions
to test against, on a problem small enough to fit on a workstation.

### 1.1 Setup
- 3D Z₂ on `L³` lattices, `L = 8, 12, 16, 24, 32`.
- β-range bracketing the critical point `β_c ≈ 0.7613` (dual to Ising
  K_c = 0.2216544...). Sample β ∈ [0.5, 1.0] with 21 values.
- ~5×10³ decorrelated configurations per (L, β); use cluster-update Wolff
  on the dual Ising representation if Metropolis autocorrelation hurts.

### 1.2 Tasks
1. **Order-parameter regression**: predict the **vortex free energy** /
   't Hooft loop (the dual order parameter). On finite volumes this is the
   logarithm of a ratio of partition functions with twisted boundary
   conditions — non-trivial to evaluate by MC, an honest learning target.
2. **Phase classification**: train a binary classifier (confined /
   deconfined). Recover `β_c` from the inflection point of the network's
   confidence as a function of β.
3. **Wilson-loop area-law detection**: predict `−log <W^{(R×T)}>` for
   variable (R, T); fit `σ R T + c (R + T) + d` to extract the **string
   tension σ(β)**. Compare against high-temperature expansion
   `σ ≈ −log tanh(β)` deep in the confined phase.
4. **Critical exponent extraction**: from the network's response on
   different volumes, do finite-size scaling on the order parameter to
   extract `ν` (correlation length exponent). 3D Ising universality class
   predicts `ν = 0.6299...` — a tight target.
5. **(Stretch) Operator scaling dimensions vs. conformal bootstrap.** In
   the FSS regime, fit the scaling dimensions of several primary operators
   simultaneously (energy, magnetization, stress tensor) from the
   network's per-operator response across volumes. 3D Ising bootstrap
   (Kos–Poland–Simmons-Duffin) gives Δ_σ = 0.5181489(10),
   Δ_ε = 1.412625(10), Δ_T = 3 exactly. Matching multiple Δ's in one fit
   is much more discriminating than ν alone, and is a clean way to argue
   the network has learned the *operator content* of the critical theory,
   not just one scale.

### 1.3 Observables to plot
- `<U_p>(β)` predicted vs. MC, all volumes overlaid.
- Σ-extracted vs. β with both ML prediction and MC fit.
- `β_c(L)` from network classifier vs. `1/L`; extrapolate to L → ∞.
- Data collapse: `M(β, L) · L^{β_exp/ν}` vs. `(β − β_c) L^{1/ν}`. The
  collapse quality is the test.

### 1.4 Pass criteria
- `β_c` extracted to within 0.5 % of 0.7613.
- `ν` to within 5 % of 0.6299.
- Cross-volume generalization: train on `L = 8, 12`; test on `L = 24, 32`
  with no MSE degradation.

### 1.5 Pitfalls
- 3D Z₂ duality only gives clean Ising mapping in **infinite volume**.
  Finite-size corrections are real and you must include them in any fit.
- Critical slowing down near β_c → ensure your MC autocorrelation is short
  (cluster-update Ising on the dual is much better than gauge-side
  Metropolis).
- Do **not** train data straddling the transition for regression tasks
  (very different physics on the two sides). Train per-phase; treat the
  phase boundary as the test.

---

## Phase 2 — 4D U(1) compact: continuous abelian gauge
**Time: 3–4 weeks**

This is the smallest continuous gauge group, with a known **first-order
deconfinement transition** at `β_c ≈ 1.0111` (Lautrup-Nauenberg 1980;
later refined). It's the natural step before SU(N).

### 2.1 Setup
- Compact U(1): links `U_{x,μ} = e^{iθ_{x,μ}}`, θ ∈ (−π, π].
- 4D lattice, `L⁴` for L = 6, 8, 10, 12.
- Wilson action `S = −β Σ_p Re U_p`, β ∈ [0.85, 1.15].
- Heat-bath updates (Hattori-Nakajima for compact U(1)). Generate ~5×10³
  configurations per β in each phase; finer sampling near β_c.
- Beware of **metastability** at the first-order transition: thermalize
  separately from cold and hot starts and verify hysteresis.

### 2.2 Tasks
1. **Plaquette regression**: standard sanity check.
2. **Photon-mass / Coulomb-phase test**: in the deconfined phase the
   transverse plaquette correlator decays as `1/r²` (massless photon). The
   network predicts `<P_{0,1}(x) P_{0,1}(0)>(r)` — compare against analytic
   lattice perturbation theory at large β.
3. **Monopole density**: in 4D compact U(1) the deconfinement is driven by
   monopole condensation (DeGrand-Toussaint construction). Train on the MC
   monopole density; test whether the network correctly identifies the
   transition without ever being shown the action — purely from
   configurations.
4. **Latent heat at the first-order transition**: from `<U_p>` jump.

### 2.3 Pass criteria
- Recover β_c to within 0.5 %.
- Distinguish the two phases on configurations from the metastable region
  (training on cold-started + hot-started separately) → the network's
  decision boundary should match the phase actually realized in each
  configuration.
- Monopole density prediction MSE within 1× of the L-CNN baseline if/when
  L-CNN is run on the same task. **Note**: L-CNN was not tested on U(1) in
  4D in the original paper — this is already a small novel data point.

### 2.4 Pitfalls
- The first-order transition in finite volume looks smooth — you need the
  L-dependence of the latent heat to argue for first-order.
- DeGrand-Toussaint monopole charges are integer-valued *labels*; use a
  classification head, not regression.

---

## Phase 3 — 4D SU(2): replicate L-CNN, then beat it
**Time: 4–6 weeks**

This is where you reproduce the published L-CNN benchmarks. Reproducing
them confirms your implementation; **beating them at matched parameter
count** is the first publishable result.

### 3.1 Setup
- Pure SU(2), 4D, Wilson action.
- Lattices: `4 × 8³` (training), `4 × 16³, 8 × 16³, 8 × 24³` (test).
- β ∈ {1.5, 2.0, 2.3, 2.5, 2.7} (β_c ≈ 2.4 for finite-T deconfinement on
  N_t = 4; pick most volumes in the confined phase).
- Heat-bath + overrelaxation (Kennedy-Pendleton). 10³ thermalized
  configurations per (V, β); train/val/test = 80/10/10.

### 3.2 Tasks (replication of L-CNN paper)
1. Wilson-loop regression: `W^{(1×2)}, W^{(2×2)}, W^{(4×4)}` (1+1D from
   the paper, but extend here to 4D).
2. **Topological-charge density** (plaquette discretization, Eq. 13 of
   review). Train at `4 × 8³`, test up to `8 × 16³`.
3. **Wilson-flowed inference**: take the `4 × 8³`-trained network and
   apply it to Wilson-flowed configurations on `8 × 24³` (Δτ = 0.005, 200
   flow steps with cooling). Recover near-integer global topological
   charge `Q_P(τ)`. Reproduce Fig. 5 of the L-CNN Letter.

### 3.3 Pass criteria — replication
- `W^{(2×2)}` MSE ≤ 1.1 × L-CNN-Medium reported (≈ 1.1 × 10⁻⁸).
- Topological-charge MSE ≤ 1.1 × L-CNN-Small reported (≈ 3 × 10⁻⁹).
- Volume generalization: MSE flat ±20 % across volumes.

### 3.4 Tasks (novel — G-GAT vs L-CNN)
1. **Matched-parameter shootout**. For each Wilson loop task, train an
   L-CNN and a G-GAT with the **same number of trainable parameters** (use
   the L-CNN sizes: 35 / 1305 / 13521 / 39905). Plot MSE vs. parameters.
2. **Long-range correlation regime**. Push β closer to the bulk-transition
   crossover (β ≈ 2.3) where the spatial correlation length grows.
   Hypothesis: at fixed parameter count, G-GAT pulls ahead of L-CNN as
   ξ/R_kernel increases, because attention with range R can see structure
   beyond the L-Conv receptive field. Quantify with `ΔMSE(ξ/R)`.
3. **Receptive-field sweep**. Train G-GAT at attention range R ∈ {1, 2, 3, 4};
   show how the optimal R correlates with the spatial correlation length
   measured on the same configurations. This is interpretable: the network
   is *learning to look as far as the physics demands*.
4. **Point-group equivariance ablation.** Train two G-GAT variants:
   (a) translation-equivariant only (default), (b) translation +
   hypercubic point group via tied `b_h(μ, k)` across μ-orbits and ±k
   (cf. `architecture.md` §12). Compare MSE at matched parameter count on
   rotation-symmetric targets (Wilson loops, topological charge). Expected:
   tying gives a free win; the size of the gap measures how much capacity
   the unconstrained network was burning on memorizing the point-group
   action. A clean, small, defensible result.

### 3.5 Pass criteria — novel
- At ≤ 10⁴ parameters and ξ/a ≥ 3, G-GAT beats L-CNN by ≥ 2× on Wilson-loop
  MSE. (If it doesn't, that itself is an interesting null result and you
  should publish it.)
- Optimal R tracks ξ/a within a factor of 2 across β ∈ [2.0, 2.5].

### 3.6 Pitfalls
- Topological charge labels are tiny (~10⁻⁴). Apply the L-CNN scale trick
  (multiply labels by 100; divide back at inference). Forgetting this is
  the most common L-CNN reproduction failure.
- The network has **no GAP**. Predict per-site, sum at the end if you need
  global Q. Do not collapse to a single output before the per-site Trace.
- For the matched-parameter comparison, *count complex parameters as 2
  reals*. Otherwise you'll silently double the G-GAT capacity.

---

## Phase 4 — Dynamical fermions and SLHMC for SU(2)
**Time: 6–8 weeks**

This is the CASK / Nagai-Tomiya territory: train the network as a
**surrogate effective fermion action** so that HMC with a heavy-mass
proposal accepts as if it were running at light mass.

### 4.1 Setup
- 2-color QCD, 4D, `4³ × 4` (matching CASK).
- Staggered fermions, β = 2.7, true mass `m_l = 0.3`, surrogate mass
  `m_h ∈ {0.4, 0.5, 0.75, 1.0}`.
- Build on JuliaQCD (`LatticeDiracOperators.jl`) — porting the architecture
  to Julia's `Zygote.jl` is the tax for getting fermion infrastructure for
  free; budget 1–2 weeks.

### 4.2 Tasks
1. **Surrogate-action regression**. Train `S_θ(U) = S_g(U) + S_f(ϕ, U^{NN}_θ(U); m_h)`
   to match the true action `S_g + S_f(ϕ, U; m_l)` via MSE on a fixed
   ensemble. CASK Eq. 18.
2. **SLHMC run**. Use `S_θ` as the molecular-dynamics Hamiltonian, accept
   with the true action. Measure acceptance rate, autocorrelation of
   plaquette and chiral condensate.
3. **G-GAT vs CASK vs CovNet**. At matched parameter count and matched
   training cost, compare:
   - Final loss (lower is better).
   - HMC acceptance.
   - Autocorrelation time of `<U_p>`, Polyakov loop, `<ψ̄ψ>`.
   - Sensitivity to m_h: a *good* surrogate should still work at m_h = 1.0
     (large mass mismatch).

### 4.3 Pass criteria
- Reproduce CASK figure 2: loss decrease across training, plaquette
  histogram match between SLHMC and ground-truth HMC (KS test p > 0.05).
- G-GAT matches CASK loss within 20 % at matched parameter count, and
  beats CovNet by ≥ 30 %.
- HMC acceptance ≥ 60 % at m_l = 0.3, m_h = 0.5 (CASK reports ≈ similar).

### 4.4 Why this matters
This is the **first place attention beats convolution in a real
physics-impact metric**: SLHMC acceptance is a clean, dimensionless number
that lattice physicists care about. A 5–10 % improvement in autocorrelation
time over CASK on the same setup is publishable on its own.

### 4.5 Pitfalls
- Pseudofermion noise in the gradient is the main training bottleneck. Use
  several stochastic estimators per training step.
- Smearing layer count > 4 starts to over-smear and destroy short-range
  fermion physics. Stick to L = 2–3 G-GAT blocks for the smearing tower.
- CASK uses sparse attention (s ≤ 3); resist the temptation to crank it up
  before validating that the loss benefit isn't a regularization artifact.

---

## Phase 5 — SU(3) pure gauge: real QCD without quarks
**Time: 4–6 weeks**

The same architecture with N_c = 3, on physical pure-gauge configurations.
This is also where you can **call your numbers QCD** in talks.

### 5.1 Setup
- SU(3), 4D, Wilson action, β ∈ {5.7, 6.0, 6.3} (lattice spacings
  a ≈ 0.17, 0.094, 0.06 fm).
- Lattices `8⁴, 12⁴, 16⁴, 16³ × 32`.
- Configurations: borrow from a public ensemble (e.g. ILDG / Gauge Connection
  / community releases) instead of generating from scratch — saves weeks.

### 5.2 Tasks
1. **String tension**. Predict `−log W(R, T)` for static-quark potential;
   fit `V(R) = σ R − π/(12 R) + const`. Compare with published `√σ` ≈
   420 MeV on each ensemble.
2. **Glueball mass**. Predict variational `0⁺⁺` glueball operators
   (smeared plaquettes and rectangles); diagonalize a small correlator
   matrix to extract `m_{0++}`. Reference: ≈ 1.7 GeV in pure SU(3).
3. **Topological susceptibility** `χ_t = <Q²>/V`. Compare with ≈ (180 MeV)⁴
   in pure SU(3) (Del Debbio et al.).
4. **Continuum-limit scaling**. Plot extracted observables vs. `a²`;
   verify standard `O(a²)` scaling and quote a continuum-extrapolated
   number with statistical and systematic errors.

### 5.3 Pass criteria
- `√σ` agrees with published value on the same ensemble within 5 %.
- `m_{0++}` within 10 %.
- `χ_t^{1/4}` within 10 %.
- Continuum extrapolation linear in a² (no a-dependence in residuals
  beyond statistics).

### 5.4 Pitfalls
- N_c = 3 enlarges every per-site matrix from 2² = 4 to 3² = 9 complex
  numbers; memory and time go up by ≥ 5×. Plan ahead for batch sizes.
- Topological charge in SU(3) needs **gradient flow** before measurement
  to suppress UV noise. Apply Wilson flow at inference (you already
  prototyped this in Phase 3.2.3).
- Glueballs are noisy. Use the variational method on a basis of ≥ 5
  smeared operators; the network's role is to predict the *smeared
  operators*, not the mass directly.

---

## Phase 6 — Novel directions
**Time: open-ended; pick one or two**

These are research-grade questions where positive results would be
publishable as a stand-alone paper, and where negative results are still
interesting. Listed roughly in order of risk.

### 6.1 Cross-β generalization through a phase transition

**Question.** Train the network at β values away from `β_c` in *both*
phases. Test on β = β_c. Does the network correctly interpolate physics
through the transition?

**Why interesting.** Standard ML transferability tests use volume; nobody
has cleanly tested across phase boundaries. A positive result argues the
network has learned the *gauge action structure* rather than the
configuration distribution.

**Concrete plan.**
- Use Phase 1 (3D Z₂) or Phase 2 (4D U(1)) as the testbed.
- Train on β ∈ {0.5, 0.6, 0.9, 1.0} with class label "phase".
- Evaluate on β ∈ {0.7, 0.75, 0.8} (straddling β_c).
- Plot prediction confidence vs. β; compare against finite-size scaling.

**Risk.** Likely partially positive: the abelian groups will work; SU(2)
near the bulk crossover is harder.

**Time.** 3–4 weeks.

---

### 6.2 Attention range as a learned correlation length

**Question.** When trained on configurations at varying β, do the learned
attention scores `α_{x→y}` peak at offsets matching the physical
correlation length ξ(β)?

**Why interesting.** This is interpretability with physical content. If
true, the network has *measured the correlation length without being told
the action*, and you can read ξ off `α` directly without ever computing a
two-point function.

**Concrete plan.**
- 3D Z₂ or 4D U(1) configurations across β.
- Train the standard regression task (plaquette, Wilson loop).
- Diagnostic: average `|α_{x→y}|` over (x, h) at fixed offset `(μ, k)`.
  Plot vs. `k`; extract decay length. Compare with MC-measured correlation
  length on the same configurations.

**Risk.** Low. The decay of α is interpretable by construction; the only
risk is that `α` is too sharply peaked at k = 1 (no useful information).
Mitigation: use larger R and a softer init.

**Time.** 2–3 weeks; doubles as a thesis-defense visualization.

---

### 6.3 Trivializing flow / normalizing flow with G-GAT

**Question.** Use stacked G-GAT blocks with L-Exp heads as a normalizing
flow `U → U_eff` such that `U_eff` is distributed according to a target
action. Sample directly without HMC.

**Why interesting.** The big open problem is **topological freezing**: at
small a, HMC can't tunnel between topological sectors. A trivializing
flow that respects gauge invariance and reaches across topology would be a
landmark result. Compare with Albergo, Kanwar, Boyda et al.
(arXiv:2003.06413, 2008.05456, 2305.02402).

**Concrete plan.**
- Phase 3 setup: 4D SU(2) at β = 2.5, `8⁴`.
- Train a flow `U → V_θ(U)` with `V_θ` = stack of G-GAT + L-Exp.
- Loss: reverse KL `D(p_target || p_θ)` via importance sampling (standard
  in the flow literature).
- Diagnostic: effective sample size, autocorrelation of Q across draws,
  tunneling rate between Q = ±1 sectors at fine β.

**Risk.** High. SU(N) flows are notoriously hard to train past `2⁴`.
A *partial* result (flow works at coarse β, fails at fine β) is still
publishable as a benchmark.

**Time.** 8–12 weeks. This is borderline thesis-scale.

**Comparison targets.**
- Boyda et al. (2008.05456): SU(N) flows on small lattices.
- Albergo et al. (2305.02402): trivializing flows for full QCD.
- Abbott et al. (2401.10874): residual flows — your G-GAT *is* a residual
  flow with attention.

---

### 6.4 Continuous-time limit: neural ODE = Wilson flow

**Question.** Take L → ∞, weight scale → 0, and identify your G-GAT stack
with a learned **continuous gradient flow** on configuration space. Train
the flow's vector field directly.

**Why interesting.** Lüscher's Wilson flow is *the* canonical smoothing
operation; replacing it with a learnable gauge-equivariant flow gives a
**task-adapted Wilson flow** that should outperform the fixed one.

**Concrete plan.**
- Take the gauge-covariant ResNet (paper 3 / your G-GAT without softmax),
  treat layer index as time, integrate with `torchdiffeq`.
- Task: scale-setting (`t_0`-flow). Standard Wilson flow defines `t_0` via
  `t² <E>(t) = 0.3`. Train your flow to *also* satisfy this, but with
  fewer steps for the same precision.
- Diagnostic: number of integration steps to reach `t_0` at fixed tolerance;
  compare against vanilla Wilson flow.

**Risk.** Moderate. Adjoint backprop through a long flow is memory-heavy.

**Time.** 6–8 weeks.

---

### 6.5 Topological-sector sampling

**Question.** Augment HMC with a G-GAT-driven proposal that explicitly
biases toward topology change. Reduce topological autocorrelation by ≥ 10×
at fine β.

**Why interesting.** This is one of the most-cited bottlenecks in lattice
QCD and there is a literature of proposals (instanton update, Lüscher's
master-field) without a clear winner. An ML proposal with proper
detailed-balance treatment would make a real impact.

**Concrete plan.**
- Phase 5 setup: SU(3) at β = 6.4 (already in topological-freezing regime
  on `16⁴`).
- Train a G-GAT to predict **the gradient of |Q − Q_target|** with respect
  to U; use this as a Langevin-step bias inside Metropolis.
- Diagnostic: Q autocorrelation time τ_Q; compare against vanilla HMC on
  the same ensemble. Target: τ_Q ↓ by 10× without observable bias on
  unrelated quantities.

**Risk.** High. Detailed balance under a learned proposal is fragile;
proper accept/reject is non-negotiable. Bayer et al. (2306.04388) is the
current state of the art for ML-assisted HMC.

**Time.** 10–12 weeks. PhD-tier.

---

### 6.6 Sign-problem alleviation via contour deformation

**Question.** For QCD with a θ-term or finite chemical potential, the
action becomes complex and standard MC fails. The **Lefschetz-thimble**
program deforms the integration contour to suppress the sign oscillation.
Can a G-GAT parametrize the deformation field?

**Concrete plan.**
- Toy: 1+1D U(1) with θ-term. Known sign problem; known thimble structure.
- G-GAT outputs a covariant link-shift `δU` defining a deformed manifold.
  Train to maximize the average sign.
- Diagnostic: average phase `<e^{iS_imag}>`; compare against thimble
  literature.

**Risk.** Very high. Thimble deformation is a delicate area where most ML
attempts haven't outperformed analytic constructions.

**Time.** 12+ weeks. Probably PhD/postdoc.

---

### 6.7 Variational ground-state / TRG-style ansatz

**Question.** Use the G-GAT as a *variational wavefunctional* on the
gauge field — i.e. parametrize `Ψ_θ(U)` as a gauge-invariant scalar
output of the network and minimize `<H>` directly. This is the lattice
analogue of neural quantum states (Carleo-Troyer 2017).

**Concrete plan.**
- Hamiltonian formulation of 2+1D Z₂ (the simplest non-trivial example).
- `Ψ_θ(U) = exp(network(U))`, network outputs an invariant scalar via
  Trace + MLP head.
- Variational MC: sample `U` ~ |Ψ_θ|², estimate `<H>`, gradient-descend.
- Compare against exact diagonalization on `4 × 4`, then extrapolate to
  larger volumes.

**Risk.** Moderate. The framework is well-established for spin systems; the
gauge-equivariant analogue is open.

**Time.** 8–10 weeks.

---

### 6.8 Beyond hypercubic: gauge theories on emergent lattices

**Question.** Extend to triangular / honeycomb / Kagome lattices for
condensed-matter applications: Z₂ spin liquids, Kitaev-honeycomb-like
models. The G-GAT framework needs only that "neighbors" and "links" be
defined; the gauge structure is identical.

**Concrete plan.**
- Honeycomb lattice with Z₂ gauge field.
- Detect topological order via the Wilson-loop expectation on
  contractible vs. non-contractible loops (the *only* distinguishing
  observable).
- Compare with bond-dimensional DMRG / iPEPS on the same Hamiltonian.

**Risk.** Implementation-heavy (lattice geometry rewiring) but
scientifically clean.

**Time.** 6–8 weeks.

---

### 6.9 θ-vacuum and topological susceptibility at imaginary θ

**Question.** Compute `χ_t(θ)` and the curvature coefficient `b_2` of the
free energy in θ for pure SU(2) or SU(3) using configurations generated
at *imaginary* θ (no sign problem) and the network to reconstruct the
topology-dependent observables.

**Why interesting.** The θ-dependence of QCD-like theories controls the
mass of the η′, the strong-CP problem, and axion-quality bounds. Lattice
calculations at imaginary θ are an active program (Bonati et al.,
Panagopoulos–Vicari). An ML accelerator for the noisiest piece —
topological charge measurement on coarse, low-flow configurations — is
directly useful, and the data side is sign-problem-free.

**Concrete plan.**
- Generate SU(N) ensembles at θ_I ∈ {0, 2, 4, 6} via reweighting or
  direct simulation (`S → S + θ_I Q`, where `Q` is the lattice topological
  charge).
- Train the network to predict `Q(x)` at each θ_I.
- Fit `<Q²>(θ_I) / V = χ_t (1 − b_2 θ_I² + …)` and analytically continue
  to real θ.

**Risk.** Low to moderate. Imaginary θ is sign-problem-free; the only
question is whether the network reduces the variance of Q measurement
relative to plaquette discretization + Wilson flow at the same statistics.

**Time.** 6–8 weeks. Builds directly on Phase 3 / Phase 5 infrastructure;
much lower risk than Phase 6.6.

---

## Suggested thesis arc

A defensible master's-thesis arc with strong-but-realistic novelty:

```
months 1–2:  Phase 0 + Phase 1   (Z₂ implementation, 3D Z₂ critical point)
months 3–4:  Phase 2              (4D U(1), first continuous group)
months 5–7:  Phase 3              (SU(2) replication + matched-param shootout)
months 8–10: Phase 4 OR Phase 6.1+6.2  (SLHMC OR cross-β/interpretability)
months 11–12: writing + defense
```

This produces:
- Independent reimplementation of L-CNN and CASK results (defensive).
- One genuinely new comparison (G-GAT vs L-CNN under controlled scaling).
- One novel small result (3D Z₂ critical exponents from ML, or
  attention-as-correlation-length).

If the student is fast and well-supported, swapping Phase 4 → Phase 6.3
(normalizing flow) is the high-risk / high-reward upgrade that moves the
thesis from "solid" to "potentially first-author paper".

---

## What to publish from the thesis

Based on the above, a single paper from the thesis would most naturally
contain:

- **G-GAT architecture description** (1 figure, 2 pages of equations).
- **L-CNN replication on 1+1D Wilson loops + 3+1D topological charge**
  (1 table, 1 figure).
- **Matched-parameter comparison G-GAT vs L-CNN, with the attention-range
  vs. correlation-length plot** (this is the novelty).
- **Phase 1 result (3D Z₂ critical exponents from ML) as a secondary check**
  on a different gauge group.

That's a complete, defensible arXiv submission and a clean MSc thesis
structure. The Phase 6 directions are then "outlook" sections pointing at
the PhD.

---

## Reference reading order (per phase)

| Phase | First reads                                                          |
|-------|----------------------------------------------------------------------|
| 0     | Wegner 1971; Creutz *Quarks, Gluons, and Lattices* ch. 1–3           |
| 1     | Wegner 1971; Pelissetto-Vicari *Phys. Rep.* 368, 549 (Ising critical exponents) |
| 2     | Lautrup-Nauenberg 1980; DeGrand-Toussaint 1980 (monopoles)           |
| 3     | Favoni et al. 2012.12901; Müller et al. 2112.11239                   |
| 4     | Nagai-Tomiya 2103.11965; Nagai-Ohno-Tomiya 2501.16955                |
| 5     | Necco-Sommer hep-lat/0108008; Del Debbio et al. hep-lat/0407028      |
| 6.1   | Cossu et al. 2301.05216 (ML phase classification reviews)            |
| 6.2   | (mostly your own — no clear precedent)                               |
| 6.3   | Albergo et al. 2003.06413, 2305.02402; Abbott et al. 2401.10874      |
| 6.4   | Lüscher 1006.4518 (Wilson flow); Chen et al. 1806.07366 (neural ODE) |
| 6.5   | Bayer et al. 2306.04388; Foreman et al. 2112.01586                   |
| 6.6   | Cristoforetti et al. 1205.3996 (Lefschetz thimbles)                  |
| 6.7   | Carleo-Troyer 2017; Luo-Clark 2102.09231                             |
| 6.8   | Kitaev cond-mat/0506438 (honeycomb model)                            |
