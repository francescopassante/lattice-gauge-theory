import torch

from lgt.lattice import Z2
from lgt.model import LatticeCNN
from lgt.train import full_pipeline

if __name__ == "__main__":
    D = 2
    N = 1000
    L = 32
    seed = 0
    hidden_channels = [16, 32]

    torch.manual_seed(seed)
    model = LatticeCNN(int(L), D, in_channels=1, hidden_channels=hidden_channels)
    result = full_pipeline(
        L=L,
        D=D,
        N=N,
        model=model,
        group=Z2(),
        beta=0.8,
        splits=(0.7, 0.15, 0.15),
        lr=1e-4,
        epochs=400,
        patience=10,
        plots=True,
        verbose=True,
        input="plaquettes",
        seed=seed,
        checkpoint_path=f"best_model_L{int(L)}.pth",
        n_therm=200,
        n_skip=5,
    )
    test_loss = result["test_loss"]
    test_label_var = result["test_label_var"]
    test_r2 = result["test_r2"]
    train_epoch = result["epochs"]

    print("L:            ", L)
    print("test_loss:    ", test_loss)
    print("var(y):       ", test_label_var)
    print("R^2:          ", test_r2)
    print("epochs:       ", train_epoch)
