from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import TensorDataset, random_split

from gelt.lattice import (
    GaugeGroup,
    action,
    as_ml_input,
    as_ml_plaquettes,
    plaquette_tensor,
)
from gelt.sampler import mcmc_ensemble


def build_link_datasets(
    N: int,
    D: int,
    L: int,
    gaugegroup: GaugeGroup,
    beta: float = 1.0,
    n_therm: int = 200,
    n_skip: int = 5,
    sampler=None,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    dtype: torch.dtype = torch.float32,
    structured: bool = True,
):
    """Dataset of (link config, action).

    ``sampler`` : ensemble-generator callable with the same signature as
    ``mcmc_ensemble``.  Defaults to ``mcmc_ensemble`` (Metropolis MC).
    Pass ``sampler=haar_ensemble`` for Haar-uniform configurations.

    ``structured=True`` (default): X shape ``(N, D, *Λ, nc, nc)`` — full matrix layout, for G-GAT.
    ``structured=False``         : X shape ``(N, D · nc², *Λ)``    — flattened color axes, for CNN.
    """
    _validate_n_configs(N)
    if sampler is None:
        sampler = mcmc_ensemble
    configs, _, _ = sampler(
        L, D, gaugegroup, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    X = configs if structured else torch.stack([as_ml_input(c) for c in configs])
    y = torch.stack([action(c, gaugegroup, beta=beta) for c in configs])

    prefix = _dataset_prefix(
        gaugegroup.name.lower(), "link", L, D, N, beta, dtype, structured
    )
    return _split(X, y, splits, save, prefix=prefix)


def build_plaquette_datasets(
    N: int,
    D: int,
    L: int,
    gaugegroup: GaugeGroup,
    beta: float = 1.0,
    n_therm: int = 200,
    n_skip: int = 5,
    sampler=None,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    dtype: torch.dtype = torch.float32,
    structured: bool = False,
):
    """Dataset of (plaquette config, action).

    ``sampler`` : ensemble-generator callable with the same signature as
    ``mcmc_ensemble``.  Defaults to ``mcmc_ensemble`` (Metropolis MC).
    Pass ``sampler=haar_ensemble`` for Haar-uniform configurations.

    ``structured=False`` (default): X shape ``(N, n_pairs · nc², *Λ)`` — flattened color axes, for CNN.
    ``structured=True``            : X shape ``(N, n_pairs, *Λ, nc, nc)`` — full matrix layout, for G-GAT.
    """
    _validate_n_configs(N)
    if sampler is None:
        sampler = mcmc_ensemble
    configs, _, _ = sampler(
        L, D, gaugegroup, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    Ps = torch.stack([plaquette_tensor(c, gaugegroup) for c in configs])
    X = Ps if structured else torch.stack([as_ml_plaquettes(p) for p in Ps])
    y = torch.stack(
        [action(configs[i], gaugegroup, beta=beta, plaquettes=Ps[i]) for i in range(N)]
    )

    prefix = _dataset_prefix(
        gaugegroup.name.lower(), "plaquette", L, D, N, beta, dtype, structured
    )
    return _split(X, y, splits, save, prefix=prefix)


def _dataset_prefix(
    group_name: str,
    kind: str,
    L: int,
    D: int,
    N: int,
    beta: float,
    dtype: torch.dtype,
    structured: bool,
) -> str:
    dtype_tag = str(dtype).replace("torch.", "")
    layout = "structured" if structured else "flat"
    return f"{group_name}_{kind}_L{L}_D{D}_N{N}_beta{beta}_dtype{dtype_tag}_{layout}"


def _validate_n_configs(N: int):
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")


def _split(X, y, splits, save, prefix):
    if len(splits) != 3 or any(s <= 0 for s in splits):
        raise ValueError(f"Expected three positive split fractions, got {splits}.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}."
        )
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {splits}.")

    n_samples = len(y)
    lengths = [int(split * n_samples) for split in splits]
    for i in range(n_samples - sum(lengths)):
        lengths[i % len(lengths)] += 1
    if any(length == 0 for length in lengths):
        raise ValueError(
            f"Dataset with N={n_samples} is too small for non-empty splits {splits}; "
            f"computed split lengths {tuple(lengths)}."
        )

    full = TensorDataset(X, y)
    train, val, test = random_split(full, lengths)

    if save:
        out_dir = Path("datasets")
        out_dir.mkdir(exist_ok=True)

        train_ds = TensorDataset(X[train.indices], y[train.indices])
        val_ds = TensorDataset(X[val.indices], y[val.indices])
        test_ds = TensorDataset(X[test.indices], y[test.indices])

        torch.save(train_ds, out_dir / f"train_dataset_{prefix}.pt")
        torch.save(val_ds, out_dir / f"val_dataset_{prefix}.pt")
        torch.save(test_ds, out_dir / f"test_dataset_{prefix}.pt")

    return train, val, test
