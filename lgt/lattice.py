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

import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

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


class SU(GaugeGroup):
    def __init__(self, n: int):
        self.name = f"SU({n})"
        self.nc = n

    def random(self, shape, dtype=torch.complex64):
        # Accept either a real or complex dtype as the precision specifier;
        # the result is always complex (real dtypes are promoted).
        complex_dtype = {
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
        }[dtype]
        # Haar sampling on U(N) via QR of a complex Ginibre matrix with the
        # Mezzadri (2007) diagonal-phase correction: PyTorch's QR fixes the
        # phase of Q by a convention that is not Haar-uniform, so we absorb
        # the phases of diag(R) back into Q's columns.
        z = torch.randn(*shape, self.nc, self.nc, dtype=complex_dtype)
        q, r = torch.linalg.qr(z)
        d = torch.diagonal(r, dim1=-2, dim2=-1)
        q = q * (d / d.abs()).unsqueeze(-2)
        # q is Haar on U(N), with det on the unit circle; divide by det^(1/nc)
        # (principal branch) to project onto SU(N).
        det = torch.linalg.det(q)
        return q / det.pow(1 / self.nc).unsqueeze(-1).unsqueeze(-1)

    def dagger(self, U):
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
        Coupling.
    plaquettes
        Pre-computed plaquette tensor; if ``None`` it is computed from ``U``.
    """
    P = plaquettes if plaquettes is not None else plaquette_tensor(U, group)
    re_tr_over_nc = P.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / group.nc
    n_plaq = re_tr_over_nc.numel()
    # equivalent to beta (sum_p 1 - P_p)
    return beta * (n_plaq - re_tr_over_nc.sum())


def as_ml_input(U: torch.Tensor) -> torch.Tensor:
    """Flatten a link tensor ``(D, *Λ, nc, nc)`` into ML input ``(C, *Λ)``.

    Used for non-equivariant models only (breaks group structure by collapsing everything)

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


def gauge_transformation(
    U: torch.Tensor,
    omega: torch.Tensor,
    group: GaugeGroup,
) -> torch.Tensor:
    """Apply a site-local gauge transformation Ω to a link configuration.

    For each direction μ:
        U'_μ(x) = Ω(x) · U_μ(x) · Ω†(x + μ̂)

    Parameters
    ----------
    U
        Link tensor of shape ``(D, *Λ, nc, nc)``.
    omega
        Gauge transformation elements of shape ``(*Λ, nc, nc)``.
    group
        Gauge group (used for the dagger operation).

    Returns
    -------
    Transformed link tensor of the same shape as ``U``.
    """
    D = U.shape[0]
    out = []
    for mu in range(D):
        omega_shifted = torch.roll(omega, shifts=-1, dims=mu)  # Ω(x + μ̂)
        out.append(omega @ U[mu] @ group.dagger(omega_shifted))
    return torch.stack(out, dim=0)


def l1_ball_offsets(D: int, R: int) -> List[Tuple[int, ...]]:
    """All non-zero offsets Δx with |Δx|₁ ≤ R, sorted by L1 norm.

    Parameters
    ----------
    D
        Number of lattice directions.
    R
        Manhattan radius.

    Returns
    -------
    List of offset tuples ``(Δx₀, …, Δx_{D-1})``, sorted so that entries
    with smaller ``|Δx|₁`` come first (ties broken by lexicographic order).
    The ordering guarantees that when building the DP table, every
    sub-step offset ``Δx ± ê_μ`` is already present when ``Δx`` is reached.
    """
    return sorted(
        (
            dx
            for dx in itertools.product(range(-R, R + 1), repeat=D)
            if 0 < sum(abs(d) for d in dx) <= R
        ),
        key=lambda dx: sum(abs(d) for d in dx),
    )


def build_transport_sums(
    U: torch.Tensor,
    R: int,
    group: GaugeGroup,
) -> Dict[Tuple[int, ...], torch.Tensor]:
    """Shortest-path-averaged parallel transports for **every** offset 0 < |Δx|₁ ≤ R.

    For each signed lattice offset Δx in the L1-ball of radius R, the returned
    tensor ``T_Δx`` has shape ``(*Λ, nc, nc)`` and equals the unweighted sum
    over all shortest lattice paths from x to x+Δx:

        T_Δx(x)  =  Σ_{P : x→x+Δx, |P|=|Δx|₁}  U_P

    This is gauge-covariant: under site-local Ω,

        T_Δx(x)  →  Ω(x) · T_Δx(x) · Ω†(x+Δx)

    The DP recursion mixes forward and backward links per component sign:

        T_Δx(x) = Σ_{μ : Δx_μ > 0}  U_μ(x) · T_{Δx − ê_μ}(x + ê_μ)
                + Σ_{μ : Δx_μ < 0}  U†_μ(x − ê_μ) · T_{Δx + ê_μ}(x − ê_μ)

    Sub-offsets ``Δx ∓ ê_μ`` always have strictly smaller L1 norm and the
    same component signs (just one zeroed out, possibly), so ordering the
    iteration by ``|Δx|_1`` guarantees every sub-step is in the table when
    needed.

    The full table covers every offset the G-Attn block iterates over —
    positive, purely-negative, and mixed-sign — uniformly.  The octant trick
    ``T_{−Δx}(x) = dagger(T_Δx(x − Δx))`` still holds as a property of the
    math and is exercised by the test suite, but is not relied on at build
    time: a single auditable surface for the gauge-implementation stress test
    (notes/architecture.md §7.2) is worth more than the 2× memory saving for now,
    later we'll maybe switch to a half table for memory efficiency.

    Parameters
    ----------
    U
        Link tensor of shape ``(D, *Λ, nc, nc)``.
    R
        Manhattan radius.
    group
        Gauge group (used for the backward-link daggers).

    Returns
    -------
    Dict mapping each signed offset tuple with ``0 < |Δx|₁ ≤ R`` to a tensor
    of shape ``(*Λ, nc, nc)``.
    """
    D = U.shape[0]
    spatial_shape = U.shape[1:-2]
    nc = U.shape[-1]

    # Identity is the DP base for the zero offset.
    identity = (
        torch.eye(nc, dtype=U.dtype, device=U.device)
        .expand(*spatial_shape, nc, nc)
        .contiguous()
    )

    # Pre-compute U†_μ(x − ê_μ) once per direction: roll by +1 in dim μ brings
    # the link tensor's value at site x − ê_μ to index x, then dagger.
    U_back: List[torch.Tensor] = [
        group.dagger(torch.roll(U[mu], shifts=1, dims=mu)) for mu in range(D)
    ]

    zero: Tuple[int, ...] = (0,) * D
    table: Dict[Tuple[int, ...], torch.Tensor] = {zero: identity}

    # All signed offsets in the L1-ball, sorted by |Δx|_1 so every sub-step is ready.
    for dx in sorted(
        (
            dx
            for dx in itertools.product(range(-R, R + 1), repeat=D)
            if 0 < sum(abs(d) for d in dx) <= R
        ),
        key=lambda dx: sum(abs(d) for d in dx),
    ):
        t: Optional[torch.Tensor] = None
        for mu in range(D):
            if dx[mu] > 0:
                prev_dx = tuple(v - 1 if i == mu else v for i, v in enumerate(dx))
                # U_μ(x) · T_{Δx−ê_μ}(x+ê_μ): roll by −1 brings x+ê_μ to index x.
                contrib = U[mu] @ torch.roll(table[prev_dx], shifts=-1, dims=mu)
            elif dx[mu] < 0:
                prev_dx = tuple(v + 1 if i == mu else v for i, v in enumerate(dx))
                # U†_μ(x−ê_μ) · T_{Δx+ê_μ}(x−ê_μ): roll by +1 brings x−ê_μ to index x.
                contrib = U_back[mu] @ torch.roll(table[prev_dx], shifts=1, dims=mu)
            else:
                continue
            t = contrib if t is None else t + contrib

        table[dx] = t

    del table[zero]
    return table
