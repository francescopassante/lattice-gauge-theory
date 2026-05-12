# Monte Carlo sampling for LGT — strategy notes

Working notes from a planning discussion. Scope: pick a sampler that gets us
from 2D Z₂ (Phase 0) through SU(3) (Phase 5) of `roadmap.md` without
becoming the bottleneck or a parallel research project of its own.

## 1. Algorithm landscape

| Algorithm | Pros | Cons |
|---|---|---|
| **Metropolis** | trivial, group-agnostic | autocorrelation explodes at large β and large L; bad critical slowing |
| **Heat-bath** (Creutz / Kennedy–Pendleton for SU(2); **Cabibbo–Marinari** for SU(N)) | exact local sampling from `exp(β Re Tr[U·A])`; τ ~5–10× shorter than Metropolis | per-group implementation |
| **Overrelaxation** | microcanonical reflection; kills autocorrelation when stacked with HB (1 HB + 4–5 OR is the lattice-QCD standard) | also per-group |
| **HMC** | uniform across groups, GPU-friendly, mandatory once dynamical fermions enter (Phase 4) | ~10× slower per autocorr time than HB+OR for pure gauge |
| **Cluster (Wolff, Swendsen–Wang)** | excellent for Z₂ / abelian; beats critical slowing | does **not** generalize to non-abelian groups |

## 2. Build our own vs. external library

Production codes considered: **OpenQCD**, **Grid** (UKQCD), **Chroma**
(USQCD), **HiRep**, **MILC**, plus Python bindings like Lyncs-API to
QUDA.

These are designed for cluster-scale ensembles with a file-based
workflow. Reasons to **not** use them here:

- Heavy integration cost vs. the actual code we'd write (samplers are
  small — see §3).
- Configurations live in proprietary formats; we'd lose GPU
  co-location with the G-GAT and round-trip through disk.
- Obscures the physics — a thesis benefits from owning the sampler.
- Their value is multi-node scaling, which we don't need at the
  volumes in `roadmap.md` Phases 0–5.

Reasons to **roll our own in PyTorch**:

- Produces `torch` tensors natively on the same device as the network.
- All four target groups (Z₂, U(1), SU(2), SU(3)) share one staple-based
  kernel — total LoC across the stack should be < ~600.
- `torch.func.grad` on the Wilson action gives the HMC molecular-dynamics
  force for free when fermions arrive.
- Easy validation against known results (analytic Z₂, plaquette-vs-β
  curves for SU(2)/SU(3)).

## 3. Recommended plan

A staple-based local-update sampler with a per-group plug-in:

1. **Z₂ / Phase 0.** Single-spin Metropolis first (≈ 10 lines, sanity
   check). Optional Wolff cluster for clean β-scans. 2D Z₂ has no bulk
   transition but `⟨P⟩(β) = tanh(β)` is analytic — perfect validation
   target.
2. **U(1) / SU(2) / SU(3).** Heat-bath + overrelaxation against the
   staple interface. Cabibbo–Marinari reduces SU(3) to repeated SU(2)
   heat-bath on three subgroups, so SU(2) is the only genuinely new
   kernel.
3. **Fermions (Phase 4).** Switch to HMC; reuse the staple machinery
   for the gauge part of the MD force, autograd handles the rest.

## 4. Normalizing flows — assessment

Active research line: Albergo, Kanwar, Boyda, Shanahan et al.
(1904.12072 for φ⁴; 2003.06413 for U(1); 2008.05456 for SU(N);
2207.08945 for SU(3)); stochastic normalizing flows from
Caselle–Cellini–Nada–Panero; CRAFT-style approaches.

**Where flows shine.** Small/moderate 2D lattices, abelian groups, and
especially **high-β regimes where HMC suffers topological freezing** —
flows can hop topological sectors HMC cannot. That is the main current
physics motivation.

**Where flows hurt.**
- Training is its own optimization problem. Reverse-KL with
  insufficient mode coverage causes silent loss of ergodicity (flow
  assigns ~zero density to whole sectors and you don't notice). Forward-KL
  needs HMC samples you don't yet have.
- ESS degrades sharply with volume.
- Recent scaling studies (Abbott et al. 2305.02402, "Aspects of scaling
  and scalability for flow-based sampling of lattice QCD") show that
  **current flow architectures don't scale to physically interesting
  4D SU(3) volumes** — training cost grows faster than HMC cost.

**Decision for this thesis.** Do **not** use flows to generate the
G-GAT's training data:
- We'd be debugging two ML systems against each other.
- A flow needs validated MC data to train against — chicken-and-egg.
- Phase 0–3 ensembles are small enough that HB+OR is essentially free.

Flows **are** a legitimate **Phase 6** topic — `roadmap.md` already lists
"trivializing flows" and "topological-sector sampling." An equivariant
G-GAT is a plausible flow *backbone*, so this is a natural follow-on
rather than a substitute for the conventional sampler.

## 5. Staples — what the term means

Standard LGT terminology, not CASK-specific. Predates the ML literature
by ~40 years (Gattringer–Lang §4, Montvay–Münster §3).

The link `U_μ(x)` participates in `2(D−1)` plaquettes. For each one, the
*other three* links form an open chain. The sum of those chains is the
**staple**:

```
A_μ(x) = Σ_{ν ≠ μ} [ U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)               (forward, ν > 0)
                   + U_ν†(x+μ̂−ν̂) · U_μ†(x−ν̂) · U_ν(x−ν̂) ]        (backward)
```

The part of the Wilson action depending on this one link collapses to

```
S_local(U_μ(x)) = −(β/nc) · Re Tr[ U_μ(x) · A_μ(x) ] + const,
```

so every local sampler — Metropolis, heat-bath, overrelaxation — needs
exactly one quantity per link: the staple sum `A_μ(x)`. Heat-bath then
draws `U` directly from `exp(β Re Tr[U · A]) / Z`.

### Relation to CASK

CASK (2501.16955) reuses the word "staple" in an architectural sense:
an open chain of links acting as the **parallel-transport path** between
two lattice points inside its attention block.

| Sense | What it is | Where used |
|---|---|---|
| **Sampling staple** (this doc) | A *sum* of three-link chains; equals the local-action coupling to one link | Metropolis / heat-bath / overrelaxation |
| **CASK staple** | *Individual* open paths used as a transport operator inside a learned layer | NN architecture |

Both are "open chains of links bordering a loop," hence the shared name.
If/when CASK-like transports enter the G-GAT, disambiguate — call the
sampling object `staple_sum`.

## 6. Open implementation questions

- Whether to keep `staple_sum` as a method on `GaugeGroup` (clean) or as
  a free function taking `U` and `(x, μ)` (matches the rest of
  `lattice.py`).
- Vectorisation: even/odd checkerboard updates so a full sweep is two
  `torch.roll`-heavy tensor ops, no Python loop over sites.
- Whether overrelaxation needs its own group-level primitive, or can be
  expressed generically as `U → A† U† A† / |A|²` once `A` is known.
- Storage: do we keep ensembles in memory (fine up to ~10⁴ configs at
  small L) or stream to disk? Tied to whether β becomes part of the
  dataset (see CLAUDE.md "Things to keep in mind").
