"""Validation plots for the Z₂ Metropolis sampler.

Four panels:
  (0,0) Thermalization  — ⟨P⟩ vs sweep at fixed (D=2, L=8) for several β
  (0,1) 2D β-scan       — ⟨P⟩ vs β for L=4,8,16 vs exact tanh(β)
  (1,0) 3D β-scan       — ⟨P⟩ vs β for L=4,6,8; shows 1st-order transition
  (1,1) Autocorrelation — C(t) of the plaquette time series at one (L, D, β)

Run:
    python validate_sampler.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from lgt.lattice import Z2, plaquette_tensor
from lgt.sampler import mcmc_ensemble, _re_tr

group = Z2()

# ── Tunable ───────────────────────────────────────────────────────────────────
N_CONFIGS = 100       # configs per (L, β) point — increase for lower noise
N_THERM   = 100       # thermalisation sweeps
N_SKIP    = 5         # sweeps between collected configs (decorrelation)
LS_2D     = [4, 8, 16]
LS_3D     = [4, 6, 8]
BETAS_2D  = np.linspace(0.0, 2.0, 21)
BETAS_3D  = np.linspace(0.0, 1.5, 21)


def _plaq_stats(configs):
    """(mean, std-err) of the mean plaquette across a batch of configs."""
    vals = np.array(
        [(_re_tr(plaquette_tensor(c, group)).mean() / group.nc).item() for c in configs]
    )
    return vals.mean(), vals.std(ddof=1) / math.sqrt(len(vals))


# ── 1/4  Thermalization histories  (D=2, L=8) ─────────────────────────────────
print("1/4  Thermalization histories …")
therm: dict = {}
for beta in [0.2, 0.5, 0.8, 1.2]:
    _, _, hist = mcmc_ensemble(
        L=8, D=2, group=group, beta=beta, n_configs=1, n_therm=400, n_skip=1
    )
    n_plaq = 8**2
    therm[beta] = np.array([1.0 - s / (beta * n_plaq) for s in hist])

# ── 2/4  2D β-scan ────────────────────────────────────────────────────────────
print("2/4  2D β-scan …")
scan2: dict = {L: ([], []) for L in LS_2D}
for L in LS_2D:
    for b in BETAS_2D:
        cfgs, _, _ = mcmc_ensemble(
            L=L, D=2, group=group, beta=float(b),
            n_configs=N_CONFIGS, n_therm=N_THERM, n_skip=N_SKIP,
        )
        m, e = _plaq_stats(cfgs)
        scan2[L][0].append(m)
        scan2[L][1].append(e)
    print(f"   L={L} ✓")

# ── 3/4  3D β-scan  (first-order transition near β_c ≈ 0.761) ────────────────
print("3/4  3D β-scan …")
scan3: dict = {L: ([], []) for L in LS_3D}
for L in LS_3D:
    for b in BETAS_3D:
        cfgs, _, _ = mcmc_ensemble(
            L=L, D=3, group=group, beta=float(b),
            n_configs=N_CONFIGS, n_therm=N_THERM, n_skip=N_SKIP,
        )
        m, e = _plaq_stats(cfgs)
        scan3[L][0].append(m)
        scan3[L][1].append(e)
    print(f"   L={L} ✓")

# ── 4/4  Plaquette autocorrelation  (D=2, L=8, β=0.8) ────────────────────────
# We run a long chain (n_configs=1000, n_skip=1) so every consecutive sweep is
# stored.  The 1000 is the chain length used to ESTIMATE C(t); MAX_LAG=50 is
# how many lags we actually plot.
print("4/4  Autocorrelation …")
AC_BETA   = 0.8
N_AC      = 1000
MAX_LAG   = 50
_, _, full_hist = mcmc_ensemble(
    L=8, D=2, group=group, beta=AC_BETA,
    n_configs=N_AC, n_therm=200, n_skip=1,
)
n_plaq_ac = 8**2
# production part only (drop the 200-sweep thermalisation prefix)
plaq_ts = np.array([1.0 - s / (AC_BETA * n_plaq_ac) for s in full_hist[200:]])
delta   = plaq_ts - plaq_ts.mean()
var     = np.mean(delta**2)
C_ac    = np.array(
    [np.mean(delta[: len(delta) - t] * delta[t:]) / var for t in range(MAX_LAG + 1)]
)
# Integrated autocorrelation time (Madras–Sokal definition).
# τ_int = 0.5 + Σ_{t=1}^{∞} C(t).  The 0.5 is the C(0)/2 contribution; it
# equals 0.5 exactly when C(0)=1, which is always true by normalisation.
# If C(t)≈0 for all t≥1 (perfectly uncorrelated chain), τ_int = 0.5,
# meaning every sweep produces an independent sample.
# The statistical error on the sample mean then scales as σ/√(N/2τ_int).
tau_int = 0.5 + float(C_ac[1:].sum())

# ── Plotting ──────────────────────────────────────────────────────────────────
COL = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("2D / 3D  Z₂  Metropolis MC  —  Validation", fontsize=13)

# ─ (0,0) Thermalization ──────────────────────────────────────────────────────
ax = axes[0, 0]
w = 15
for i, (beta, curve) in enumerate(sorted(therm.items())):
    ax.plot(curve, color=COL[i], alpha=0.2, lw=0.8)
    smooth = np.convolve(curve, np.ones(w) / w, mode="valid")
    ax.plot(np.arange(w - 1, len(curve)), smooth, color=COL[i], lw=1.8,
            label=f"β = {beta}")
    ax.axhline(math.tanh(beta), color=COL[i], ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("Sweep")
ax.set_ylabel("⟨P⟩  (smoothed, w=15)")
ax.set_title("Thermalization  (D=2, L=8)\nDashed = analytic tanh(β)")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(-0.05, 1.05)

# ─ (0,1) 2D β-scan ───────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(BETAS_2D, np.tanh(BETAS_2D), "k--", lw=2, label="tanh(β) exact", zorder=5)
for i, L in enumerate(LS_2D):
    ms, es = np.array(scan2[L][0]), np.array(scan2[L][1])
    ax.errorbar(BETAS_2D, ms, yerr=es, fmt="o-", ms=3, capsize=2,
                color=COL[i], label=f"L = {L}")
ax.set_xlabel("β")
ax.set_ylabel("⟨P⟩")
ax.set_title("Mean plaquette  —  2D Z₂\nvs. exact tanh(β)")
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 2.05)
ax.set_ylim(-0.05, 1.05)

# ─ (1,0) 3D β-scan ───────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(BETAS_3D, np.tanh(BETAS_3D), "k--", lw=1.0, alpha=0.35,
        label="tanh(β)  [2D ref]")
ax.axvline(0.7613, color="gray", ls=":", lw=1.4, label="β_c ≈ 0.761")
for i, L in enumerate(LS_3D):
    ms, es = np.array(scan3[L][0]), np.array(scan3[L][1])
    ax.errorbar(BETAS_3D, ms, yerr=es, fmt="o-", ms=3, capsize=2,
                color=COL[i], label=f"L = {L}")
ax.set_xlabel("β")
ax.set_ylabel("⟨P⟩")
ax.set_title("Mean plaquette  —  3D Z₂\n1st-order transition at β_c ≈ 0.761")
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 1.55)
ax.set_ylim(-0.05, 1.05)

# ─ (1,1) Autocorrelation ─────────────────────────────────────────────────────
ax = axes[1, 1]
lags = np.arange(MAX_LAG + 1)
ax.bar(lags, C_ac, color=COL[0], alpha=0.7, width=0.8)
ax.axhline(0, color="k", lw=0.8)
# 1/e marks the point where an exponentially decaying autocorrelation
# C(t) = exp(-t/τ_exp) crosses 1/e, i.e. t = τ_exp (the exponential
# autocorrelation time).  It's a visual reference for how fast the chain mixes.
ax.axhline(1 / math.e, color="gray", ls="--", lw=1.2, label="1/e  (τ_exp reference)")
ax.set_xlabel("Lag  t  (sweeps)")
ax.set_ylabel("C(t)")
ax.set_title(
    f"Plaquette autocorrelation  (D=2, L=8, β={AC_BETA})\n"
    f"τ_int ≈ {tau_int:.1f} sweeps  (chain length = {N_AC})"
)
ax.legend(fontsize=8)
ax.set_xlim(-0.5, MAX_LAG + 0.5)

fig.tight_layout()
plt.savefig("sampler_validation.png", dpi=150, bbox_inches="tight")
print("Saved sampler_validation.png")
plt.show()
