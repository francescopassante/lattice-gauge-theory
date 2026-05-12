"""Unit tests for lattice primitives.

Run with:  pytest test_lattice.py -v
"""

import torch
import pytest

from lgt.lattice import Z2, random_links, plaquette_tensor, action, gauge_transformation


@pytest.fixture
def z2():
    return Z2()


def _random_omega(L: int, D: int, group, dtype, seed: int = 42) -> torch.Tensor:
    """Sample a random gauge transformation Ω of shape (*Λ, nc, nc)."""
    torch.manual_seed(seed)
    return group.random((L,) * D, dtype=dtype)


# ---------------------------------------------------------------------------
# Z₂ plaquette invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L,D", [(4, 2), (6, 2), (4, 3)])
def test_plaquette_bitexact_z2(z2, L, D):
    """Z₂ plaquettes are bit-exact after any gauge transformation (float64)."""
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=1)

    P_before = plaquette_tensor(U, z2)
    U_prime = gauge_transformation(U, omega, z2)
    P_after = plaquette_tensor(U_prime, z2)

    assert torch.equal(P_before, P_after), (
        f"Plaquettes not bit-exact after Z₂ gauge transform (L={L}, D={D}); "
        f"max diff = {(P_before - P_after).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Action invariance (general — holds for all unitary groups)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L,D,beta", [(4, 2, 1.0), (6, 2, 2.5), (4, 3, 0.5)])
def test_action_invariant_z2(z2, L, D, beta):
    """Wilson action is invariant under gauge transformation (float64)."""
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=2)

    S_before = action(U, z2, beta=beta)
    U_prime = gauge_transformation(U, omega, z2)
    S_after = action(U_prime, z2, beta=beta)

    assert torch.equal(S_before, S_after), (
        f"Action not invariant under Z₂ gauge transform "
        f"(L={L}, D={D}, β={beta}); diff = {(S_before - S_after).abs().item()}"
    )


# ---------------------------------------------------------------------------
# Plaquette covariance: P'(x) = Ω(x) P(x) Ω†(x)
# This is the general identity for any unitary group; for Z₂ it reduces
# to the bit-exact test above, but the explicit form guards porting to SU(N).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L,D", [(4, 2), (6, 2)])
def test_plaquette_covariance_z2(z2, L, D):
    """P'(x) = Ω(x) P(x) Ω†(x) holds exactly for Z₂ (float64)."""
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=3)

    P = plaquette_tensor(U, z2)
    U_prime = gauge_transformation(U, omega, z2)
    P_prime = plaquette_tensor(U_prime, z2)

    # Expected: omega[None] @ P @ dagger(omega)[None]
    # P has shape (n_pairs, *Λ, nc, nc); omega has shape (*Λ, nc, nc)
    P_expected = omega @ P @ z2.dagger(omega)  # broadcasts over n_pairs leading dim

    assert torch.allclose(P_prime, P_expected, atol=0.0), (
        f"Plaquette covariance P'=ΩPΩ† violated (L={L}, D={D}); "
        f"max diff = {(P_prime - P_expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

def test_output_shape_preserved(z2):
    """gauge_transformation returns a tensor with the same shape as U."""
    L, D = 5, 2
    torch.manual_seed(0)
    U = random_links(L, D, z2)
    omega = _random_omega(L, D, z2, torch.float32)
    U_prime = gauge_transformation(U, omega, z2)
    assert U_prime.shape == U.shape
