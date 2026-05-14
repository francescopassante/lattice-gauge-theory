from itertools import product as iproduct
from typing import Sequence

import torch
import torch.nn as nn


class _RollConvND(nn.Module):
    """Circular convolution for any D via roll + linear.

    Equivalent to a ConvNd with circular padding. Parameter count:
    ``out_channels * in_channels * kernel_size**D + out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, D: int, kernel_size: int):
        super().__init__()
        p = kernel_size // 2
        # Generate all offsets in the D-dimensional kernel, e.g. for D=2 and p=1:
        # [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self._offsets = list(iproduct(range(-p, p + 1), repeat=D))
        # Actual lattice coordinates (first 2 dims are batch and channel)
        self._spatial_dims = tuple(range(2, 2 + D))
        self.linear = nn.Linear(in_channels * len(self._offsets), out_channels)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, *spatial)
        ndim = len(self._spatial_dims)
        # creates K^D copies of the lattice, each shifted by a kernel offset, then stacks them up -> (N, K^D, C, *spatial)
        parts = [
            torch.roll(x, shifts=off, dims=self._spatial_dims) for off in self._offsets
        ]
        # (N, K^D, C_in, *spatial) → flatten kernel and channel axes
        gathered = torch.stack(parts, dim=1).flatten(1, 2)  # (N, K^D * C_in, *spatial)
        # permute so spatial is in the middle for nn.Linear: (N, *spatial, K^D * C_in)
        perm = (0,) + tuple(range(2, 2 + ndim)) + (1,)
        out = self.linear(gathered.permute(*perm))  # (N, *spatial, C_out)
        # restore channel-second layout: (N, C_out, *spatial)
        perm_back = (0, ndim + 1) + tuple(range(1, ndim + 1))
        return out.permute(*perm_back).contiguous()


def _make_conv(in_ch: int, out_ch: int, D: int, kernel_size: int) -> nn.Module:
    p = kernel_size // 2
    if D == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size, padding=p, padding_mode="circular")
    if D == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size, padding=p, padding_mode="circular")
    return _RollConvND(in_ch, out_ch, D, kernel_size)


class LatticeCNN(nn.Module):
    """CNN baseline with circular convolutions to enforce periodic BCs.

    Supports D=2 (``Conv2d``), D=3 (``Conv3d``), and D≥4 (roll-based circular
    convolution with identical parameter count to a hypothetical ``ConvNd``).

    NOT gauge-equivariant — used as a reference against which the equivariant
    architecture is compared.
    """

    def __init__(
        self,
        L: int,
        D: int,
        in_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.L = L
        self.D = D
        self.in_channels = in_channels

        channels = [in_channels, *hidden_channels]
        layers = []
        for chan_in, chan_out in zip(channels, channels[1:]):
            layers.append(_make_conv(chan_in, chan_out, D, kernel_size))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(self.L**self.D * channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}. "
                f"Did you mix link-input vs. plaquette-input datasets?"
            )
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(-1)


if __name__ == "__main__":
    from torchsummary import summary

    L = 5
    D = 2
    in_channels = 2  # D=2 link channels for Z₂
    model = LatticeCNN(L, D, in_channels=in_channels, hidden_channels=[16, 32])
    summary(model, (in_channels, L, L))
