import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lgt.lattice import Z2
from lgt.cnn_baseline import LatticeCNN
from lgt.train import full_pipeline

if __name__ == "__main__":
    L = 8
    D = 2
    N = 1000
    seed = 0
    hidden_channels = [16, 32]
    lrs = np.logspace(-2, -5, 7)  # 1e-2 … 1e-5, seven points

    test_losses = np.zeros(len(lrs))
    test_label_vars = np.zeros(len(lrs))
    test_r2s = np.zeros(len(lrs))
    train_epochs = np.zeros(len(lrs))
    train_losses_all = []
    val_losses_all = []

    for i, lr in enumerate(tqdm(lrs)):
        torch.manual_seed(seed)
        model = LatticeCNN(L, D, in_channels=1, hidden_channels=hidden_channels)
        result = full_pipeline(
            L=L,
            D=D,
            N=N,
            model=model,
            group=Z2(),
            splits=(0.7, 0.15, 0.15),
            lr=float(lr),
            epochs=2000,
            patience=20,
            plots=False,
            verbose=True,
            input="plaquettes",
            seed=seed,
            checkpoint_path=f"best_model_lr{lr:.0e}.pth",
        )
        test_losses[i] = result["test_loss"]
        test_label_vars[i] = result["test_label_var"]
        test_r2s[i] = result["test_r2"]
        train_epochs[i] = result["epochs"]
        train_losses_all.append(np.array(result["train_losses"]))
        val_losses_all.append(np.array(result["val_losses"]))

    print("lrs:          ", lrs)
    print("test_loss:    ", test_losses)
    print("var(y):       ", test_label_vars)
    print("R²:           ", test_r2s)
    print("epochs:       ", train_epochs)

    def _save(fig_name):
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogx(lrs, test_r2s, marker="o")
    plt.axhline(1.0, color="g", ls=":", label="R² = 1 (perfect)")
    plt.xlabel("Learning rate")
    plt.ylabel("R² = 1 − MSE / Var(y)")
    plt.title("LR scan: test R² (CNN, plaquettes, L=8)")
    plt.grid(True, ls=":")
    plt.legend()
    _save("LR scan R2.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lr, train_l, val_l in zip(lrs, train_losses_all, val_losses_all):
        label = f"lr={lr:.0e}"
        axes[0].plot(train_l, label=label, alpha=0.8)
        axes[1].plot(val_l, label=label, alpha=0.8)

    for ax, title in zip(axes, ("Train loss", "Val loss")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.grid(True, ls=":")
        ax.legend(fontsize=8)

    plt.suptitle("LR scan: convergence curves (CNN, plaquettes, L=8)")
    _save("LR scan curves.png")
