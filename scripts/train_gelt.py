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
        "N": 1000,
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

    # Matched-capacity hyperparameters with scripts/train_cnn.py. With
    # (nhead=2, d_qkv=8, mlp_hidden=16, gemhsa_layers=2) GELT has 911 numel
    # ≈ 1707 real DOFs (complex Q/K/V/mix weights count for 2 real DOFs each),
    # comparable to the CNN baseline at hidden_channels=[1], fc_hidden=2
    # (1678 numel, all real). Ratio in real DOFs ≈ 1.02×.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 2,
        "gemhsa_layers": 2,
        "d_qkv": 8,
        "gate": "softplus",
        "dtype": torch.complex64,
        "mlp_hidden": 16,
        "mlp_out": 1,
    }

    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )

    model = GELT(**model_parameters)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

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

    # Standardize the target (notes/architecture.html §6.1). Compute (μ_y, σ_y)
    # on the train split, then mutate the shared full-y tensor in place — all
    # three subsets are Subsets of the same TensorDataset, so val/test see the
    # normalized labels too. Paired with the zero-init of the MLP's last layer
    # (gelt/blocks.py), the untrained model is the constant predictor at the
    # normalized mean (= 0), giving R² = 0 — the trivial mean-baseline.
    # Predictions and saved targets are denormalized at the end for plotting.
    y_train = torch.cat([batch[-1] for batch in train_loader])
    mu_y = y_train.mean()
    sigma_y = y_train.std(unbiased=False).clamp_min(1e-12)
    train_dataset.dataset.tensors[-1].sub_(mu_y).div_(sigma_y)
    print(f"target scaler fit: μ_y = {mu_y.item():.4f} | σ_y = {sigma_y.item():.4f}")

    X, T, y = next(iter(train_loader))
    n_params = sum(p.numel() for p in model.parameters())
    n_real_dofs = sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
    )
    print(
        f"GELT | params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
        f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype} | "
        f"out {tuple(model(X, T).shape)}"
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

    # ``test_loss`` and the saved arrays are in normalized space (y was
    # standardized in place above). R² is invariant under linear label
    # transforms, so we can compute it either way. Denormalize to show the
    # scatter plot in physical Wilson-action units.
    all_targets = all_targets * sigma_y + mu_y
    all_outputs = all_outputs * sigma_y + mu_y
    test_label_var = all_targets.var(unbiased=False).item()
    test_mse_physical = ((all_outputs - all_targets) ** 2).mean().item()
    test_r2 = (
        1.0 - test_mse_physical / test_label_var if test_label_var > 0 else float("nan")
    )

    print(
        f"Test Loss (norm): {test_loss:.4f} | "
        f"Test MSE (physical): {test_mse_physical:.4f} | "
        f"Var(y): {test_label_var:.4f} | R²: {test_r2:.4f}"
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
    plt.savefig("gelt_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets.numpy(), all_outputs.numpy(), alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values (Test Set)")
    plt.grid(True)
    plt.savefig("gelt_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
