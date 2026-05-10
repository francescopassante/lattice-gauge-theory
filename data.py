from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import TensorDataset, random_split

from lattice import (
    GaugeGroup,
    Z2,
    action,
    as_ml_input,
    as_ml_plaquettes,
    plaquette_tensor,
    random_links,
)


def build_link_datasets(
    N: int,
    D: int,
    L: int,
    group: Optional[GaugeGroup] = None,
    beta: float = 1.0,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    """Dataset of (link configuration, action). X shape ``(N, D · nc², *Λ)``."""
    group = group if group is not None else Z2()
    generator = _make_generator(seed)

    sample_x = as_ml_input(random_links(L, D, group, generator, dtype=dtype))
    X = torch.zeros((N,) + sample_x.shape, dtype=sample_x.dtype)
    y = torch.zeros(N, dtype=dtype)

    # Reset generator so iteration 0 is reproducible (we already drew a sample above).
    generator = _make_generator(seed)
    for i in range(N):
        U = random_links(L, D, group, generator, dtype=dtype)
        X[i] = as_ml_input(U)
        y[i] = action(U, group, beta=beta)

    return _split(X, y, splits, save, prefix=f"{group.name.lower()}_link", generator=generator)


def build_plaquette_datasets(
    N: int,
    D: int,
    L: int,
    group: Optional[GaugeGroup] = None,
    beta: float = 1.0,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    """Dataset of (plaquette configuration, action). X shape ``(N, n_pairs · nc², *Λ)``."""
    group = group if group is not None else Z2()
    generator = _make_generator(seed)

    sample_x = as_ml_plaquettes(
        plaquette_tensor(random_links(L, D, group, generator, dtype=dtype), group)
    )
    X = torch.zeros((N,) + sample_x.shape, dtype=sample_x.dtype)
    y = torch.zeros(N, dtype=dtype)

    generator = _make_generator(seed)
    for i in range(N):
        U = random_links(L, D, group, generator, dtype=dtype)
        P = plaquette_tensor(U, group)
        X[i] = as_ml_plaquettes(P)
        y[i] = action(U, group, beta=beta, plaquettes=P)

    return _split(X, y, splits, save, prefix=f"{group.name.lower()}_plaquette", generator=generator)


def _make_generator(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g


def _split(X, y, splits, save, prefix, generator):
    full = TensorDataset(X, y)
    train, val, test = random_split(full, list(splits), generator=generator)
    if save:
        out_dir = Path("datasets")
        out_dir.mkdir(exist_ok=True)
        torch.save(train, out_dir / f"train_dataset_{prefix}.pt")
        torch.save(val, out_dir / f"val_dataset_{prefix}.pt")
        torch.save(test, out_dir / f"test_dataset_{prefix}.pt")
    return train, val, test
