"""Optimised GEMHSA / GELT — numerically identical to gelt.blocks, but with
two algorithmic rewrites of the hot path identified by scripts/profile_gelt.py:

  (c) Transport T·X·T†:
      original — launches (B·H·d·n_off·|Λ|) ≈ 5·10⁷ tiny (2,2)@(2,2) cgemm's
                 and broadcast-replicates T 16× over (H, d_qkv).
      here     — folds (H, d_qkv, col) into the right-multiplicand's column
                 dimension, so we issue (B·n_off·|Λ|) matmuls of
                 (nc, nc)@(nc, H·d·nc) — 16× fewer launches, 16× more useful
                 work per launch, and T is no longer broadcast-replicated.

  (d) Value path Σ_n α_n · (Q† · V_tilde_n):
      original — one matmul per offset, then α-weighted sum.
      here     — α-weights V_tilde over offsets *first*, then a single
                 matmul. Mathematically identical (matmul is linear); 24×
                 fewer matmul launches at the benchmark shape.

Everything else (QKV projection, score, residual + L-Act gate, MLP, Trace)
is byte-for-byte the original implementation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gelt.lattice import l1_ball_offsets


class GEMHSA(nn.Module):
    def __init__(
        self,
        gaugegroup,
        L,
        D,
        R,
        d_input,
        nhead,
        d_qkv=None,
        gate="relu",
        dtype=torch.complex64,
    ):
        super(GEMHSA, self).__init__()
        self.gaugegroup = gaugegroup
        self.D = D
        self.R = R
        self.H = nhead
        self.C = d_input
        self.d_qkv = d_input // nhead if d_qkv is None else d_qkv
        if self.d_qkv < 1:
            raise ValueError(
                f"d_qkv must be >= 1, got {self.d_qkv} "
                f"(d_input={d_input}, nhead={nhead})."
            )
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate!r}")
        self.gate = gate

        self.offsets = l1_ball_offsets(D, R)
        self.n_offsets = len(self.offsets)

        offset_tensor = torch.tensor(self.offsets, dtype=torch.long)
        coords = torch.meshgrid(
            *[torch.arange(L) for _ in range(D)], indexing="ij"
        )
        nbr_idx = torch.stack(
            [
                (coords[d].unsqueeze(0) + offset_tensor[:, d].view(-1, *([1] * D))) % L
                for d in range(D)
            ],
            dim=0,
        )
        self.register_buffer("_nbr_idx", nbr_idx)

        sigs = [
            tuple(sorted((abs(d) for d in dx), reverse=True)) for dx in self.offsets
        ]
        unique_sigs = sorted(set(sigs))
        sig_to_idx = {s: i for i, s in enumerate(unique_sigs)}
        self.n_orbits = len(unique_sigs)
        orbit_idx = torch.tensor([sig_to_idx[s] for s in sigs], dtype=torch.long)
        self.register_buffer("_orbit_idx", orbit_idx)
        self.b_h = nn.Parameter(torch.zeros(self.H, self.n_orbits))

        self.C_tilde = 2 * d_input + 1

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

        sigma_mix = 0.02 / math.sqrt(self.H * self.d_qkv)
        self.w_mix = self._init_projection(
            (self.C, self.H, self.d_qkv), sigma_mix, dtype
        )

        # ReZero scalar; init to 0 so the untrained block is exactly identity.
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _init_projection(shape, sigma, dtype):
        if dtype.is_complex:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            re = torch.randn(*shape, dtype=real_dtype) * sigma
            im = torch.randn(*shape, dtype=real_dtype) * sigma
            return nn.Parameter(torch.complex(re, im))
        return nn.Parameter(torch.randn(*shape, dtype=dtype) * sigma)

    def _augment(self, W):
        spatial = W.shape[2:-2]
        nc = W.shape[-1]
        identity = torch.eye(nc, dtype=W.dtype, device=W.device).expand(
            W.shape[0], 1, *spatial, nc, nc
        )
        return torch.cat([identity, W, self.gaugegroup.dagger(W)], dim=1)

    def _transport_folded(self, X_nb, T, T_dag):
        """Compute T(x) · X_nb(x, n) · T†(x) for every (h, d, n, x) with
        (H, d_qkv) folded into the column dim of the right-multiplicand.

        Inputs:
            X_nb : (B, H, d_qkv, n_off, *Λ, nc, nc)
            T    : (B, n_off, *Λ, nc, nc)
            T_dag: (B, n_off, *Λ, nc, nc)  -- precomputed once per layer
        Returns:
            (B, H, d_qkv, n_off, *Λ, nc, nc), bit-equivalent to the naive
            two-matmul broadcast version up to floating-point reassociation.
        """
        B = X_nb.shape[0]
        H = self.H
        d = self.d_qkv
        n = self.n_offsets
        nc = X_nb.shape[-1]
        spatial = X_nb.shape[4:-2]
        Dsp = len(spatial)

        # Move (H, d) to sit just after the row index 'i'. Source axes:
        #   0=B, 1=H, 2=d, 3=n, 4..3+Dsp=spatial, 4+Dsp=i, 5+Dsp=j
        # Target:
        #   0=B, 1=n, 2..1+Dsp=spatial, 2+Dsp=i, 3+Dsp=H, 4+Dsp=d, 5+Dsp=j
        perm = (0, 3) + tuple(range(4, 4 + Dsp)) + (4 + Dsp, 1, 2, 5 + Dsp)
        Xp = X_nb.permute(*perm)
        # Flatten (H, d, j) -> wide column. reshape forces a contiguous copy.
        X_flat = Xp.reshape(B, n, *spatial, nc, H * d * nc)
        # Left-multiply: one big matmul per (B, n_off, x).
        L = T @ X_flat  # (B, n, *Λ, nc, H*d*nc)

        # Now right-multiply by T†. Unflatten then re-flatten so (i, H, d)
        # become rows and j stays the contraction axis:
        L = L.reshape(B, n, *spatial, nc, H, d, nc)
        L_flat = L.reshape(B, n, *spatial, nc * H * d, nc)
        R = L_flat @ T_dag  # (B, n, *Λ, nc*H*d, nc)

        # Reshape and permute back to (B, H, d, n, *Λ, nc, nc).
        out = R.reshape(B, n, *spatial, nc, H, d, nc)
        inv_perm = (0, 3 + Dsp, 4 + Dsp, 1) + tuple(range(2, 2 + Dsp)) + (2 + Dsp, 5 + Dsp)
        return out.permute(*inv_perm).contiguous()

    def _attend(self, Q, K, V, T):
        nc = Q.shape[-1]

        # 1. Neighbour gather — same as the original.
        idx = tuple(self._nbr_idx[k] for k in range(self.D))
        nb_indexer = (slice(None),) * 3 + idx + (slice(None), slice(None))
        K_nb = K[nb_indexer]
        V_nb = V[nb_indexer]

        # 2. Adjoint transport with (H, d_qkv) folded into the matmul column.
        T_dag = self.gaugegroup.dagger(T)
        K_tilde = self._transport_folded(K_nb, T, T_dag)
        V_tilde = self._transport_folded(V_nb, T, T_dag)

        # 3. Score = Tr[Q† K̃]/sqrt(nc·d_qkv) as Frobenius product (cheap).
        Q_e = Q.unsqueeze(3)  # (B, H, d, 1, *Λ, nc, nc)
        score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real
        score = score / math.sqrt(self.d_qkv * nc)

        bias = self.b_h[:, self._orbit_idx]
        if torch.is_complex(bias):
            bias = bias.real
        score = score + bias.view(1, self.H, self.n_offsets, *([1] * self.D))

        # 4. Softmax over offsets.
        alpha = torch.softmax(score, dim=2)

        # 5. Value path. Sum V_tilde over n_off with α weights BEFORE the
        # Q† matmul — matmul is linear, so this is mathematically identical
        # to the original's "matmul then weight then sum" but issues 24×
        # fewer matmul calls at R=2 in D=3.
        # alpha: (B, H, n, *Λ) -> (B, H, 1, n, *Λ, 1, 1) to broadcast over
        # (d_qkv, nc, nc).
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        V_weighted = (alpha_b * V_tilde).sum(dim=3)  # (B, H, d, *Λ, nc, nc)
        Q_dag = self.gaugegroup.dagger(Q)  # (B, H, d, *Λ, nc, nc)
        return torch.matmul(Q_dag, V_weighted)  # (B, H, d, *Λ, nc, nc)

    def forward(self, W, T):
        assert T.shape[1] == self.n_offsets, (
            f"Expected T.shape[1] == {self.n_offsets}, got {T.shape[1]}"
        )

        nc = W.shape[-1]
        W_aug = self._augment(W)
        B = W_aug.shape[0]
        trailing = W_aug.shape[2:]

        W_aug_flat = W_aug.view(B, self.C_tilde, -1)
        w_Q_flat = self.w_Q.view(self.H * self.d_qkv, self.C_tilde)
        w_K_flat = self.w_K.view(self.H * self.d_qkv, self.C_tilde)
        w_V_flat = self.w_V.view(self.H * self.d_qkv, self.C_tilde)

        Q = torch.matmul(w_Q_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)
        K = torch.matmul(w_K_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)
        V = torch.matmul(w_V_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)

        out = self._attend(Q, K, V, T)

        W_mix = torch.einsum("iha,bha...->bi...", self.w_mix, out)

        W_res = W + W_mix
        trace_per_chan = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
        if self.gate == "relu":
            g = F.relu(trace_per_chan)
        else:
            g = F.softplus(trace_per_chan)
        W_act = g.unsqueeze(-1).unsqueeze(-1) * W_res
        return W + self.alpha * (W_act - W)


class Trace(nn.Module):
    def forward(self, W):
        trace = W.diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.cat([trace.real, trace.imag], dim=1)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = x.movedim(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GELT(nn.Module):
    def __init__(
        self,
        gaugegroup,
        L,
        D,
        R,
        nhead,
        gemhsa_layers,
        d_qkv=None,
        gate="softplus",
        dtype=torch.complex64,
        mlp_hidden=32,
        mlp_out=1,
    ):
        d_input = D * (D - 1) // 2
        super(GELT, self).__init__()
        self.gemhsa_models = nn.ModuleList(
            [
                GEMHSA(gaugegroup, L, D, R, d_input, nhead, d_qkv, gate, dtype)
                for _ in range(gemhsa_layers)
            ]
        )

        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.mlp = MLP(2 * d_input, mlp_hidden, mlp_out).to(real_dtype)
        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)

    def attn(self, W, T):
        for layer in self.gemhsa_models:
            W = layer(W, T)
        return W

    def forward(self, W, T):
        W_attn = self.attn(W, T)
        trace = self.trace(W_attn)
        site_out = self.mlp(trace)
        spatial_dims = tuple(range(1, site_out.ndim - 1))
        return site_out.sum(dim=spatial_dims).squeeze(-1)
