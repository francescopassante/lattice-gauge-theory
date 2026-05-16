import math

import torch
import torch.nn as nn

from gelt import build_transport_sums, l1_ball_offsets


class GEMHSA(nn.Module):
    """Gauge-equivariant multi-head self-attention block (G-Attn).

    Implements §3 of ``notes/architecture.html``. The input is a batched
    covariant W-field of shape ``(B, C, *Λ, nc, nc)``; each Q/K/V channel
    that comes out is locally covariant at the same site.
    """

    def __init__(
        self,
        linktensor,
        gaugegroup,
        L,
        D,
        R,
        d_input,
        nhead,
        d_qkv=None,
        dropout=0.1,
        dtype=torch.complex64,
    ):
        super(GEMHSA, self).__init__()
        self.gaugegroup = gaugegroup
        self.linktensor = linktensor
        self.L = L
        self.D = D
        self.R = R
        self.H = nhead
        self.C = d_input
        self.d_qkv = d_input // nhead if d_qkv is None else d_qkv
        # Channel augmentation (notes/architecture.html §2.3) expands
        # C -> C̃ = 2C + 1 by prepending the identity and appending daggers.
        self.C_tilde = 2 * d_input + 1
        # §5: small Gaussian init with σ ≈ 0.02 / √C, real and imaginary
        # parts independent. Together with the residual + small w^V this
        # makes the block ≈ identity at init.
        sigma = 0.02 / math.sqrt(self.C_tilde)
        self.w_Q = self._init_projection(
            (self.H, self.d_qkv, self.C_tilde), sigma, dtype
        )
        self.w_K = self._init_projection(
            (self.H, self.d_qkv, self.C_tilde), sigma, dtype
        )
        self.w_V = self._init_projection(
            (self.H, self.d_qkv, self.C_tilde), sigma, dtype
        )

    @staticmethod
    def _init_projection(shape, sigma, dtype):
        """Small-Gaussian init for a complex (or real) projection weight.

        For complex ``dtype`` we draw real and imaginary parts independently
        from ``N(0, σ²)`` — using ``torch.randn(..., dtype=complex)`` would
        give each part variance ``σ²/2``, halving the spec's std.
        """
        if dtype.is_complex:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            re = torch.randn(*shape, dtype=real_dtype) * sigma
            im = torch.randn(*shape, dtype=real_dtype) * sigma
            return nn.Parameter(torch.complex(re, im))
        return nn.Parameter(torch.randn(*shape, dtype=dtype) * sigma)

    def _augment(self, W):
        # Batched form of gelt.lattice.augment:
        #   (B, C, *Λ, nc, nc) -> (B, 2C+1, *Λ, nc, nc)
        # Prepend the site-local identity, append the daggered channels.
        spatial = W.shape[2:-2]
        nc = W.shape[-1]
        identity = torch.eye(nc, dtype=W.dtype, device=W.device).expand(
            W.shape[0], 1, *spatial, nc, nc
        )
        return torch.cat([identity, W, self.gaugegroup.dagger(W)], dim=1)

    def _aggregate(self, W):
        """
        For each site in the lattice, aggregates the neighboring W-fields
        by parallel transporting them to the site and summing.
        """

        offsets = l1_ball_offsets(self.D, self.R)
        T = build_transport_sums(
            U=self.linktensor, R=self.R, gaugegroup=self.gaugegroup
        )

        Wprime = torch.zeros_like(W)
        for offset in offsets:
            transport_tensor = T[offset]  # (*Λ, nc, nc)
            Wprime += torch.matmul(transport_tensor, W)  # (B, C, *Λ, nc, nc)

        return Wprime

    def forward(self, U):
        # U shape: (batch_size, channel, *Λ, nc, nc)
        T = build_transport_sums(U)  # (D, R, nc, nc)

        # augment W, then mix channels to build Q, K, V of shape
        # (B, H, d_qkv, *Λ, nc, nc).

        U_aug = self._augment(U)  # (B, C, *Λ, nc, nc), contiguous
        B = U_aug.shape[0]
        trailing = U_aug.shape[2:]  # (*Λ, nc, nc)

        # Collapse dimensions so that matmul broadcasts correctly
        # (last two dimensions interpreted as matrix dimensions)
        #   (H·d, C) @ (B, C, N) -> (B, H·d, N).

        U_aug_flat = U_aug.view(B, self.C_tilde, -1)
        w_Q_flat = self.w_Q.view(self.H * self.d_qkv, self.C_tilde)
        w_K_flat = self.w_K.view(self.H * self.d_qkv, self.C_tilde)
        w_V_flat = self.w_V.view(self.H * self.d_qkv, self.C_tilde)

        Q = torch.matmul(w_Q_flat, U_aug_flat).view(B, self.H, self.d_qkv, *trailing)
        K = torch.matmul(w_K_flat, U_aug_flat).view(B, self.H, self.d_qkv, *trailing)
        V = torch.matmul(w_V_flat, U_aug_flat).view(B, self.H, self.d_qkv, *trailing)

        return U


if __name__ == "__main__":
    from gelt import SU, plaquette_tensor, random_links

    # Timing test
    input = random_links(L=5, D=3, gaugegroup=SU(3)).unsqueeze(0)
    U = plaquette_tensor(input, gaugegroup=SU(3))
    print(U.shape)
