import numpy as np
from tqdm import tqdm

from train import full_pipeline

if __name__ == "__main__":
    D = 2
    N = 1000
    hidden_channels = [16, 32]
    Ls = np.arange(4, 33, 4, dtype=np.int64)
    print(Ls)

    test_losses = np.zeros(len(Ls))
    test_label_vars = np.zeros(len(Ls))
    test_r2s = np.zeros(len(Ls))
    train_epochs = np.zeros(len(Ls))

    for i, L in enumerate(tqdm(Ls)):
        result = full_pipeline(
            L=int(L),
            D=D,
            N=N,
            hidden_channels=hidden_channels,
            splits=(0.7, 0.15, 0.15),
            lr=1e-3,
            epochs=300,
            patience=10,
            plots=False,
            verbose=True,
            input="links",
            seed=0,
            checkpoint_path=f"best_model_L{int(L)}.pth",
        )
        test_losses[i] = result.test_loss
        test_label_vars[i] = result.test_label_var
        test_r2s[i] = result.test_r2
        train_epochs[i] = result.epochs

    print("Ls:           ", Ls)
    print("test_loss:    ", test_losses)
    print("var(y):       ", test_label_vars)
    print("R^2:          ", test_r2s)
    print("epochs:       ", train_epochs)
