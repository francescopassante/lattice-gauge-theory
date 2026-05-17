import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt import haar_ensemble
from gelt.blocks import GELT

"""
========================================================================================
 This is a minimal training script used for lookup on how to train the architecture
========================================================================================
"""


def evaluate(model, test_loader, criterion, device, save_outputs=False):
    model.eval()

    test_loss = 0.0
    test_count = 0
    if save_outputs:
        all_targets = []
        all_outputs = []
    with torch.no_grad():
        for X, T, y in tqdm(test_loader):
            X, T, y = X.to(device), T.to(device), y.to(device)
            outputs = model(X, T)
            loss = criterion(outputs, y)
            batch_size = y.shape[0]
            test_loss += loss.item() * batch_size
            test_count += batch_size
            if save_outputs:
                all_targets.append(y.cpu())
                all_outputs.append(outputs.cpu())

    test_loss /= test_count
    if save_outputs:
        all_targets = torch.cat(all_targets)
        all_outputs = torch.cat(all_outputs)
        return test_loss, all_targets, all_outputs
    return test_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=5,
    checkpoint_path: str = "best_model.pth",
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for X, T, y in tqdm(train_loader):
            X, T, y = X.to(device), T.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X, T)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            batch_size = y.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size

        train_loss /= train_count
        train_losses.append(train_loss)
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_losses, epoch + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    from gelt import SU, build_plaquette_datasets

    D = 3
    L = 8
    gaugegroup = SU(2)
    R = 2

    dataset_parameters = {
        "N": 100,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": True,
        "structured": True,
        "sampler": haar_ensemble,
        "beta": 1,
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32,
    }

    train_parameters = {
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 300,
        "patience": 30,
        "checkpoint_path": "best_model.pth",
    }

    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 2,
        "gemhsa_layers": 2,
        "d_qkv": None,
        "gate": "softplus",
        "dtype": torch.complex64,
        "mlp_hidden": 2,
        "mlp_out": 1,
    }

    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )

    model = GELT(**model_parameters)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_parameters["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )

    X, T, y = next(iter(train_loader))
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"GELT | params: {n_params:,} | "
        f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype} | "
        f"out {tuple(model(X, T).shape)}"
    )

    # Identity-at-init diagnostic (notes/architecture.html §10): at init the
    # GEMHSA stack should be a small perturbation of the input. We report
    # ||attn(W,T) - W||_F / ||W||_F (relative Frobenius drift) and the
    # untrained val R² so a poor initialization is visible before training.
    with torch.no_grad():
        W_attn = model.attn(X, T)
        drift = (W_attn - X).norm() / X.norm()
        init_pred = model(X, T)
        var_y = y.var(unbiased=False).item()
        init_mse = ((init_pred - y) ** 2).mean().item()
        init_r2 = 1.0 - init_mse / var_y if var_y > 0 else float("nan")
    print(
        f"[init] identity drift ||attn(W,T)-W||/||W|| = {drift.item():.3e} | "
        f"train-batch MSE = {init_mse:.4f} | Var(y) = {var_y:.4f} | R² = {init_r2:.4f}"
    )

    model = model.to(device)

    train_losses, val_losses, full_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=train_parameters["epochs"],
        patience=train_parameters["patience"],
        checkpoint_path=train_parameters["checkpoint_path"],
    )

    # Load best model to evaluate on test set
    model.load_state_dict(
        torch.load(
            train_parameters["checkpoint_path"], map_location=device, weights_only=True
        )
    )

    test_loss, all_targets, all_outputs = evaluate(
        model, test_loader, criterion, device, save_outputs=True
    )

    # Plots and visualizations

    # Population variance of the labels — the natural scale to normalise MSE by.
    test_label_var = all_targets.var(unbiased=False).item()
    test_r2 = 1.0 - test_loss / test_label_var if test_label_var > 0 else float("nan")

    print(
        f"Test Loss: {test_loss:.4f} | Var(y): {test_label_var:.4f} | R²: {test_r2:.4f}"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets.numpy(), all_outputs.numpy(), alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values (Test Set)")
    plt.grid(True)
    plt.show()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
