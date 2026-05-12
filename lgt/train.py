from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lgt.lattice import GaugeGroup


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    verbose=True,
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
        wrap = tqdm if verbose else (lambda x: x)
        for inputs, targets in wrap(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size

        train_loss /= train_count
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in wrap(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_size = targets.shape[0]
                val_loss += loss.item() * batch_size
                val_count += batch_size

        val_loss /= val_count
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_losses, epoch + 1


def full_pipeline(
    L: int,
    D: int,
    N: int,
    model: nn.Module,
    group: GaugeGroup,
    beta: float = 1.0,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    plots: bool = False,
    verbose: bool = True,
    input: str = "plaquette",
    batch_size: int = 32,
    seed: Optional[int] = None,
    checkpoint_path: str = "best_model.pth",
    sampler=None,
    n_therm: int = 200,
    n_skip: int = 5,
) -> dict:
    """
    ``sampler`` : ensemble-generator callable with the same interface as
                  ``mcmc_ensemble``.  ``None`` (default) auto-dispatches
                  to the registered sweep for ``group`` via ``_SWEEP_FN``.
                  Pass ``sampler=haar_ensemble`` for Haar-uniform configurations.
    """
    from lgt.data import build_link_datasets, build_plaquette_datasets

    if seed is not None:
        torch.manual_seed(seed)

    input_key = input.lower()
    if input_key not in {"plaquette", "plaquettes", "link", "links"}:
        raise ValueError(
            "input must be one of: 'plaquette', 'plaquettes', 'link', 'links'."
        )

    builder = (
        build_plaquette_datasets
        if input_key in {"plaquette", "plaquettes"}
        else build_link_datasets
    )
    train_dataset, val_dataset, test_dataset = builder(
        N, D, L, group=group, beta=beta, splits=splits, structured=False,
        sampler=sampler, n_therm=n_therm, n_skip=n_skip,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, full_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    test_loss = 0.0
    test_count = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        wrap = tqdm if verbose else (lambda x: x)
        for inputs, targets in wrap(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.shape[0]
            test_loss += loss.item() * batch_size
            test_count += batch_size
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())

    test_loss /= test_count
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    # Population variance of the labels — the natural scale to normalise MSE by.
    test_label_var = all_targets.var(unbiased=False).item()
    test_r2 = 1.0 - test_loss / test_label_var if test_label_var > 0 else float("nan")

    if verbose:
        print(
            f"Test Loss: {test_loss:.4f} | Var(y): {test_label_var:.4f} | R²: {test_r2:.4f}"
        )

    if plots:
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

    return {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
