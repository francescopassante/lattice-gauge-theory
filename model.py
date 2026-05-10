from typing import Sequence

import torch
import torch.nn as nn
from torchsummary import summary


class LatticeCNN(nn.Module):
    """CNN baseline with circular-padded ``Conv2d`` to enforce periodic BCs.

    NOT gauge-equivariant — used as a reference against which the equivariant
    architecture is compared.
    """

    def __init__(self, L: int, D: int, in_channels: int, hidden_channels: Sequence[int]):
        super().__init__()
        if D != 2:
            raise NotImplementedError("LatticeCNN currently uses Conv2d (D=2 only).")
        self.L = L
        self.D = D
        self.in_channels = in_channels

        channels = [in_channels, *hidden_channels]
        layers = []
        for chan_in, chan_out in zip(channels, channels[1:]):
            layers.append(
                nn.Conv2d(chan_in, chan_out, kernel_size=3, padding=1, padding_mode="circular")
            )
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
    L = 5
    D = 2
    in_channels = 2  # D=2 link channels for Z₂
    model = LatticeCNN(L, D, in_channels=in_channels, hidden_channels=[16, 32])
    summary(model, (in_channels, L, L))
