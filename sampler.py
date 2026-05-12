"""Monte Carlo sampler for lattice gauge theory.

Phase 0: single-site Metropolis with checkerboard vectorisation.
The staple interface is designed for extension to heat-bath + overrelaxation
for U(1)/SU(2)/SU(3) without restructuring the sweep loop.
"""

from typing import List, Optional, Tuple

import torch

from lattice import Z2, GaugeGroup, action, plaquette_tensor, random_links


def staple_sum(U: torch.Tensor, mu: int, group: GaugeGroup) -> torch.Tensor:
    """Sum of staples for every site along direction ``mu``.

    The local Wilson action for link U_μ(x) is
        S_local = -(β/nc) Re Tr[ U_μ(x) · A_μ(x) ]
    where the staple sum A_μ(x) is::

        (Using LGT convention where the ordering is reverse (from left to right), cyclicity of trace allows it)
        Σ_{ν≠μ} [  U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)          (forward staple)
                  + U_ν†(x+μ̂-ν̂) · U_μ†(x-ν̂) · U_ν(x-ν̂) ]  (backward staple)

    Parameters
    ----------
    U     : ``(D, *Λ, nc, nc)``
    mu    : direction index
    group : gauge group (used for ``dagger``)

    Returns
    -------
    Tensor of shape ``(*Λ, nc, nc)``.
    """
    D = U.shape[0]
    A = torch.zeros_like(U[mu])
    for nu in range(D):
        if nu == mu:
            continue
        Umu = U[mu]
        Unu = U[nu]
        # Forward: U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)
        Unu_fwd = torch.roll(Unu, shifts=-1, dims=mu)  # U_ν(x + μ̂)
        Umu_nu = torch.roll(Umu, shifts=-1, dims=nu)  # U_μ(x + ν̂)
        A = A + Unu_fwd @ group.dagger(Umu_nu) @ group.dagger(Unu)
        # Backward: U_ν†(x+μ̂-ν̂) · U_μ†(x-ν̂) · U_ν(x-ν̂)
        Unu_bwd = torch.roll(torch.roll(Unu, shifts=-1, dims=mu), shifts=+1, dims=nu)
        Umu_negnu = torch.roll(Umu, shifts=+1, dims=nu)  # U_μ(x - ν̂)
        Unu_negnu = torch.roll(Unu, shifts=+1, dims=nu)  # U_ν(x - ν̂)
        A = A + group.dagger(Unu_bwd) @ group.dagger(Umu_negnu) @ Unu_negnu
    return A


def _re_tr(M: torch.Tensor) -> torch.Tensor:
    """Re Tr for a batch of matrices: ``(*batch, nc, nc)`` → ``(*batch)``."""
    return M.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real


def _site_parity(spatial_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Checkerboard parity (0 or 1) for each site. Shape: ``(*spatial_shape)``."""
    coords = torch.meshgrid(
        *[torch.arange(s, device=device) for s in spatial_shape],
        indexing="ij",
    )
    return sum(coords) % 2


def metropolis_sweep(
    U: torch.Tensor,
    group: GaugeGroup,
    beta: float,
) -> Tuple[torch.Tensor, float]:
    """One full Metropolis sweep (all directions, both checkerboard parities).

    For Z₂ the proposal is the unique non-identity element U' = −U. To extend
    to U(1)/SU(N), replace the ``U_proposed = -U_mu`` line with a group-valued
    random proposal drawn from a neighbourhood of U_mu.

    Checkerboard structure: for a fixed direction μ, sites of the same parity
    do not share any plaquette through same-direction links, so their updates
    commute. The even sweep then the odd sweep is equivalent to a sequential
    site-by-site update but fully vectorised.

    Parameters
    ----------
    U         : ``(D, *Λ, nc, nc)`` — not modified in-place
    group     : gauge group
    beta      : inverse coupling

    Returns
    -------
    (U_new, acceptance_rate)
    """
    D = U.shape[0]
    spatial_shape = U.shape[1:-2]
    nc = group.nc
    device = U.device

    parity = _site_parity(spatial_shape, device)  # (*Λ)

    U = U.clone()
    total_proposed = 0
    total_accepted = 0

    for mu in range(D):
        for par in (0, 1):
            A = staple_sum(U, mu, group)  # (*Λ, nc, nc)
            U_mu = U[mu]
            U_proposed = -U_mu  # Z₂: only proposal is the flip

            # ΔS = (β/nc) Re Tr[(U − U') · A]  > 0 means action increases
            dS = (beta / nc) * _re_tr((U_mu - U_proposed) @ A)  # (*Λ)

            rand = torch.rand(spatial_shape).to(device)
            accept = (dS <= 0) | (rand < torch.exp(-dS.clamp(min=0)))

            site_mask = parity == par  # (*Λ)
            update_mask = accept & site_mask  # (*Λ)

            total_accepted += update_mask.sum().item()
            total_proposed += site_mask.sum().item()

            # [..., None, None] adds two extra dimension to broadcast update_mask with U_mu (or U_proposed)
            U[mu] = torch.where(update_mask[..., None, None], U_proposed, U_mu)

    return U, total_accepted / total_proposed


def generate_ensemble(
    L: int,
    D: int,
    group: GaugeGroup,
    beta: float,
    n_configs: int,
    n_therm: int = 200,
    n_skip: int = 5,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float, List[float]]:
    """Generate a thermalized ensemble of gauge field configurations.

    Starts from a Haar-random configuration, runs ``n_therm`` thermalisation
    sweeps, then collects one configuration every ``n_skip`` sweeps.

    Parameters
    ----------
    L, D      : lattice size and number of dimensions
    group     : gauge group
    beta      : inverse coupling (Boltzmann weight ~ exp(−β S))
    n_configs : number of configurations to collect
    n_therm   : thermalisation sweeps before collection begins
    n_skip    : sweeps between collected configurations (decorrelation)
    dtype, device : passed to ``random_links`` / sweep

    Returns
    -------
    (configs, mean_acceptance, action_history)
        ``configs``        : ``(n_configs, D, *Λ, nc, nc)`` on CPU
        ``mean_acceptance``: mean Metropolis acceptance rate over production run
        ``action_history`` : action S at every sweep (thermalisation + production)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    U = random_links(L, D, group, dtype=dtype).to(device)

    action_history = []

    for _ in range(n_therm):
        U, _ = metropolis_sweep(U, group, beta)
        action_history.append(action(U, group, beta).cpu().item())

    configs: List[torch.Tensor] = []
    acc_rates: List[float] = []
    for i in range(n_configs * n_skip):
        U, acc = metropolis_sweep(U, group, beta)
        action_history.append(action(U, group, beta).cpu().item())
        if (i + 1) % n_skip == 0:
            configs.append(U.cpu())
            acc_rates.append(acc)

    return torch.stack(configs), sum(acc_rates) / len(acc_rates), action_history
