"""Lattice gauge theory primitives as pure tensor operations.

Conventions
-----------
- Link tensor shape: ``(D, L, ..., L, nc, nc)`` where the leading axis indexes
  the spatial direction ``μ ∈ {0, ..., D-1}``, the middle ``D`` axes are the
  spatial coordinates, and the trailing ``(nc, nc)`` axes are the matrix
  representation of the link in the gauge group's defining representation.
- Even for Z₂ (where ``nc = 1``) the trailing color axes are kept so that
  every operation generalises verbatim to U(1) / SU(N). The dagger is written
  out explicitly for the same reason.
- Plaquette tensor shape: ``(D(D-1)/2, L, ..., L, nc, nc)``, ordered by
  ``(μ, ν)`` pairs with ``μ < ν`` lexicographically.
- Plaquette convention:
  ``P_{μν}(x) = U_μ(x) · U_ν(x + μ̂) · U_μ†(x + ν̂) · U_ν†(x)``.
- Periodic boundary conditions throughout (``torch.roll`` for shifts).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class GaugeGroup(ABC):
    """Abstract gauge group, parametrised by its defining-representation dimension ``nc``."""

    name: str
    nc: int

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def random(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample group elements as a tensor of shape ``shape + (nc, nc)`` (Haar)."""

    @abstractmethod
    def dagger(self, U: torch.Tensor) -> torch.Tensor:
        """Hermitian conjugate (= group inverse for unitary groups)."""


class Z2(GaugeGroup):
    name = "Z2"
    nc = 1

    def random(self, shape, dtype=torch.float32):
        signs = (torch.randint(0, 2, shape, dtype=torch.int64) * 2 - 1).to(dtype)
        # Add the trailing (nc, nc) = (1, 1) color axes so that the layout matches U(1)/SU(N).
        return signs.unsqueeze(-1).unsqueeze(-1)

    def dagger(self, U):
        # Identity for real 1×1 matrices, but written explicitly for portability.
        return U.conj().transpose(-1, -2)


def random_links(
    L: int,
    D: int,
    group: GaugeGroup,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a Haar-random link configuration of shape ``(D, L, ..., L, nc, nc)``."""
    return group.random((D,) + (L,) * D, dtype=dtype)


def plaquette_tensor(U: torch.Tensor, group: GaugeGroup) -> torch.Tensor:
    """Compute every 1×1 plaquette ``P_{μν}(x)`` for ``μ < ν``.

    Parameters
    ----------
    U
        Links of shape ``(D, *Λ, nc, nc)`` where ``*Λ`` is the spatial shape.
    group
        Gauge group (used for the dagger operation).

    Returns
    -------
    Tensor of shape ``(n_pairs, *Λ, nc, nc)`` with ``n_pairs = D(D-1)/2``.
    """
    D = U.shape[0]
    pairs = [(mu, nu) for mu in range(D) for nu in range(mu + 1, D)]
    plaqs = []
    for mu, nu in pairs:
        # After indexing U[mu] / U[nu], the spatial axis ``mu`` sits at index ``mu``
        # (axes 0..D-1 are spatial, then nc, nc). torch.roll(t, -1, dims=mu) brings
        # the value at lattice position ``x + μ̂`` to index ``x``.
        Umu = U[mu]
        Unu = U[nu]
        Unu_shift_mu = torch.roll(Unu, shifts=-1, dims=mu)  # U_ν(x + μ̂)
        Umu_shift_nu = torch.roll(Umu, shifts=-1, dims=nu)  # U_μ(x + ν̂)
        P = Umu @ Unu_shift_mu @ group.dagger(Umu_shift_nu) @ group.dagger(Unu)
        plaqs.append(P)
    return torch.stack(plaqs, dim=0)


def action(
    U: torch.Tensor,
    group: GaugeGroup,
    beta: float = 1.0,
    plaquettes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wilson action ``S = β Σ_p (1 − Re Tr P_p / N_c)``.

    Parameters
    ----------
    U
        Link tensor (ignored if ``plaquettes`` is provided).
    group
        Gauge group (used to compute plaquettes if needed).
    beta
        Coupling. Default 1.0 reproduces the unnormalised ``n_plaq − Σ P`` form
        used by the original baseline experiments.
    plaquettes
        Pre-computed plaquette tensor; if ``None`` it is computed from ``U``.
    """
    P = plaquettes if plaquettes is not None else plaquette_tensor(U, group)
    re_tr_over_nc = P.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / group.nc
    n_plaq = re_tr_over_nc.numel()
    return beta * (n_plaq - re_tr_over_nc.sum())


def as_ml_input(U: torch.Tensor) -> torch.Tensor:
    """Flatten a link tensor ``(D, *Λ, nc, nc)`` into ML input ``(C, *Λ)``.

    Real groups: ``C = D · nc²``.
    Complex groups: ``C = 2 · D · nc²`` (real and imaginary parts as separate channels).
    For Z₂ (``nc = 1``, real) the output collapses to ``(D, *Λ)``, matching the
    legacy CNN baseline interface.
    """
    D = U.shape[0]
    spatial = U.shape[1:-2]
    nc = U.shape[-1]
    if torch.is_complex(U):
        re_im = torch.stack([U.real, U.imag], dim=1)  # (D, 2, *Λ, nc, nc)
        return re_im.reshape(D * 2 * nc * nc, *spatial)
    return U.reshape(D * nc * nc, *spatial)


def as_ml_plaquettes(P: torch.Tensor) -> torch.Tensor:
    """Same flattening as :func:`as_ml_input` for the plaquette tensor."""
    n_pairs = P.shape[0]
    spatial = P.shape[1:-2]
    nc = P.shape[-1]
    if torch.is_complex(P):
        re_im = torch.stack([P.real, P.imag], dim=1)
        return re_im.reshape(n_pairs * 2 * nc * nc, *spatial)
    return P.reshape(n_pairs * nc * nc, *spatial)
