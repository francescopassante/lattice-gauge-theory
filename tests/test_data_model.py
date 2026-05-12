import pytest
import torch

from lgt.data import _split
from lgt.model import LatticeCNN


def test_split_rejects_empty_partition():
    X = torch.arange(2).float().unsqueeze(1)
    y = torch.arange(2).float()

    with pytest.raises(ValueError, match="too small"):
        _split(X, y, (0.7, 0.15, 0.15), save=False, prefix="test")


def test_split_rejects_fractions_that_do_not_sum_to_one():
    X = torch.arange(10).float().unsqueeze(1)
    y = torch.arange(10).float()

    with pytest.raises(ValueError, match="sum to 1.0"):
        _split(X, y, (0.7, 0.2, 0.2), save=False, prefix="test")


def test_lattice_cnn_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="positive odd"):
        LatticeCNN(L=4, D=2, in_channels=1, hidden_channels=[2], kernel_size=2)


def test_lattice_cnn_preserves_shape_for_larger_odd_kernel():
    model = LatticeCNN(L=4, D=2, in_channels=1, hidden_channels=[2], kernel_size=5)
    out = model(torch.randn(3, 1, 4, 4))

    assert out.shape == (3,)
