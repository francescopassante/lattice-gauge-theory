"""Plots from the previous L-scan runs of the CNN baseline.

These numbers were saved by the pre-refactor pipeline, which reported only the
absolute MSE. Because the Wilson-action labels grow in scale with the lattice
volume, raw MSE plots are not directly interpretable across L. Below we report
the same data normalised by the analytic label variance for Haar-random Z₂.

Analytic variance (Haar-random Z₂, D=2)
---------------------------------------
Each plaquette is a product of four ±1 links → mean 0, variance 1. Two
plaquettes are independent under random ±1 links (any pair of plaquettes
shares either 0 or 2 links; in both cases the expected product vanishes).
With ``n_plaq = L² · D(D-1)/2 = L²`` independent ±1 contributions,

    Var(action) = Var(n_plaq − Σ_p P_p) = n_plaq = L².

So an MSE of order ``L²`` is exactly the variance of the labels, i.e.
no signal at all. ``R² = 1 − MSE / Var(y)`` puts every L on the same
scale.
"""

import matplotlib.pyplot as plt
import numpy as np

# N = 1000, D = 2, channel_dimensions = [in_channels, 16, 32], lr = 1e-3, splits = [0.7, 0.15, 0.15]

Ls = np.array([4, 8, 12, 16, 20, 24, 28, 32], dtype=float)

test_losses_plaquettes = np.array(
    [
        [0.01717679, 0.15430574, 0.59761062, 1.23729314, 4.44196692, 5.67420044, 7.43264852, 10.51627159],
        [0.01297260, 0.17351418, 0.52883530, 1.25842520, 3.92064381, 7.92166319, 8.25286283, 12.16067047],
    ]
)
train_epochs_plaquettes = np.array(
    [
        [171.0, 240.0, 220.0, 244.0, 235.0, 245.0, 244.0, 162.0],
        [222.0, 195.0, 158.0, 226.0, 180.0, 211.0, 244.0, 240.0],
    ]
)

test_losses_links = np.array(
    [
        [16.61995583, 70.14940643, 147.33751221, 192.79042664, 443.58779297, 486.53728027, 764.34353027, 1092.29003906],
        [14.61621370, 63.01112671, 157.82126160, 262.17309875, 455.67015381, 525.73504028, 780.90612183, 1072.42370605],
    ]
)
train_epochs_links = np.array(
    [
        [20.0, 18.0, 22.0, 20.0, 23.0, 24.0, 33.0, 50.0],
        [28.0, 25.0, 27.0, 48.0, 25.0, 56.0, 43.0, 51.0],
    ]
)

# Analytic label variance for Haar-random Z₂ in 2D: Var(action) = n_plaq = L².
analytic_var = Ls**2

# Average across the two repetitions.
test_loss_plaquette_avg = np.mean(test_losses_plaquettes, axis=0)
train_epochs_plaquette_avg = np.mean(train_epochs_plaquettes, axis=0)
test_loss_links_avg = np.mean(test_losses_links, axis=0)
train_epochs_links_avg = np.mean(train_epochs_links, axis=0)

r2_plaquettes = 1.0 - test_loss_plaquette_avg / analytic_var
r2_links = 1.0 - test_loss_links_avg / analytic_var


def _save(fig_name):
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


# Absolute test loss (legacy plots).
plt.figure(figsize=(10, 5))
plt.plot(Ls, test_loss_links_avg, marker="o", label="links input")
plt.plot(Ls, test_loss_plaquette_avg, marker="s", label="plaquettes input")
plt.plot(Ls, analytic_var, "k--", label="Var(y) = L² (chance level)")
plt.xlabel("L")
plt.ylabel("Test MSE")
plt.title("Absolute test MSE vs L (CNN baseline)")
plt.yscale("log")
plt.grid(True, which="both", ls=":")
plt.legend()
_save("Test loss vs L_absolute.png")

# Normalised by label variance: this is the panel that actually says something.
plt.figure(figsize=(10, 5))
plt.plot(Ls, r2_links, marker="o", label="links input")
plt.plot(Ls, r2_plaquettes, marker="s", label="plaquettes input")
plt.axhline(0.0, color="k", ls="--", label="R² = 0 (predicting the mean)")
plt.axhline(1.0, color="g", ls=":", label="R² = 1 (perfect)")
plt.xlabel("L")
plt.ylabel("R² = 1 − MSE / Var(y)")
plt.title("Normalised generalisation: CNN baseline on Haar-random Z₂")
plt.grid(True, ls=":")
plt.legend()
_save("Test R2 vs L.png")

# Epochs plots, kept for completeness.
plt.figure(figsize=(10, 5))
plt.plot(Ls, train_epochs_links_avg, marker="o", label="links input")
plt.plot(Ls, train_epochs_plaquette_avg, marker="s", label="plaquettes input")
plt.xlabel("L")
plt.ylabel("Epochs to early-stop")
plt.title("Training cost vs L (CNN baseline)")
plt.grid(True, ls=":")
plt.legend()
_save("Train epochs vs L.png")
