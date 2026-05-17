import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gelt.lattice import l1_ball_offsets


class GEMHSA(nn.Module):
    """Gauge-equivariant multi-head self-attention block (G-Attn).

    The input is a batched covariant W-field of shape ``(B, C, *Λ, nc, nc)`` and the output has the
    same shape, so blocks chain. Every channel of W transforms in the
    adjoint representation, ``W → Ω W Ω†``.

    Pipeline (one G-Attn block):

    1. **augment + project.** Append the on-site identity and daggers
    (``C → C̃ = 2C + 1``), then project to per-head Q, K, V of
       shape ``(B, H, d_qkv, *Λ, nc, nc)``.
    2. **adjoint transport.** For every Δx in the L1-ball, gather
       the neighbour fields ``K(x+Δx)`` / ``V(x+Δx)`` and apply the
       shortest-path-averaged transport in the adjoint representation:
       ``Kprime = T(x) · K(x+Δx) · T†(x)``.
    3. **score.** ``s = (1/√(nc·d)) · Re Σ_a Tr[Q†_a · K̃_a]`` per offset
       and per head, plus a learnable real bias tied across the lattice
       point-group orbit of Δx (sign flips + axis permutations).
    4. **softmax** over the offset axis (normalizes over neighbours per
       site, per head).
    5. **multiplicative value path.** Output of the attention head is
       ``Σ_i α_i · Q†(x) · Ṽ_i(x)`` — both factors are covariant at x, so the
       product is covariant; this is the L-Bilin-baked-in step that gives the
       loop-doubling expressivity argument.
    6. **channel mix back to C** via a complex linear ``(H, d_qkv) → C``.
    7. **residual + L-Act gate.** ``W_act = g(W_res) · W_res`` with
       ``W_res = W_in + W_mix`` and ``g(W) = ReLU(Re Tr[W]/nc)`` (default)
       or ``softplus(Re Tr[W]/nc)``.

    The transport ``T`` is precomputed by the dataset builder (it is a
    function of the link configuration only, see
    :func:`gelt.lattice.build_transport_sums`). ``forward(W, T)`` takes it as
    a tensor of shape ``(B, n_offsets, *Λ, nc, nc)``
    """

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
                f"(d_input={d_input}, nhead={nhead}). "
                f"Pass d_qkv explicitly when d_input < nhead — this happens "
                f"e.g. with GELT in D=2, where d_input = D(D-1)/2 = 1."
            )
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate!r}")
        self.gate = gate

        # offsets is a list of the Δx_i in the L1 ball of radius R
        self.offsets = l1_ball_offsets(D, R)
        self.n_offsets = len(self.offsets)

        # _nbr_idx[d, i, x] are the coords of the neighbor of x at offset Δx_i
        # = (x[d] + Δx_i[d]) mod L.
        offset_tensor = torch.tensor(self.offsets, dtype=torch.long)  # (n_off, D)
        coords = torch.meshgrid(
            *[torch.arange(L) for _ in range(D)], indexing="ij"
        )  # (*Λ)
        nbr_idx = torch.stack(
            [
                (coords[d].unsqueeze(0) + offset_tensor[:, d].view(-1, *([1] * D))) % L
                for d in range(D)
            ],
            dim=0,
        )  # (D, n_offsets, *Λ)
        self.register_buffer("_nbr_idx", nbr_idx)

        # Orbit-tied score bias. Offsets in the same point-group orbit
        # (sign flips + axis permutations) share a single learnable scalar
        # per head. Sigs are essentially the offsets in the positive octant,
        # all the other bias are tied to these by symmetry.
        sigs = [
            tuple(sorted((abs(d) for d in dx), reverse=True)) for dx in self.offsets
        ]
        unique_sigs = sorted(set(sigs))
        sig_to_idx = {s: i for i, s in enumerate(unique_sigs)}
        # n_orbits = number of unique biases
        self.n_orbits = len(unique_sigs)
        # orbit_idx = indices of sigs, where two sigs that are symmetry-tied have the same index (n_sigs)
        orbit_idx = torch.tensor([sig_to_idx[s] for s in sigs], dtype=torch.long)
        self.register_buffer("_orbit_idx", orbit_idx)  # (n_offsets,)
        self.b_h = nn.Parameter(torch.zeros(self.H, self.n_orbits))

        # Channel augmentation expands
        # C -> Ctilde = 2C + 1 by prepending the identity and appending daggers.
        self.C_tilde = 2 * d_input + 1

        # Small Gaussian init with σ ≈ 0.02 / √Ctilde, real and imaginary
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
        # channel mix back to C output channels.
        sigma_mix = 0.02 / math.sqrt(self.H * self.d_qkv)
        self.w_mix = self._init_projection(
            (self.C, self.H, self.d_qkv), sigma_mix, dtype
        )

        # ReZero / LayerScale: per-block learnable scalar α, init to 0 so the
        # block is *exactly* identity at init regardless of gate choice (the
        # gate g_softplus(0) = ln 2 ≠ 1 would otherwise rescale W on the first
        # forward, breaking the §5 "identity-at-init" property). α is real and
        # gauge-invariant; the convex combination of two equivariant terms
        # remains equivariant. See notes/architecture.html §3.8.
        self.alpha = nn.Parameter(torch.zeros(1))

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
        # Channel augmentation: (B, C, *Λ, nc, nc) -> (B, 2C+1, *Λ, nc, nc).
        # Prepend the site-local identity, append the daggered channels.
        spatial = W.shape[2:-2]
        nc = W.shape[-1]
        identity = torch.eye(nc, dtype=W.dtype, device=W.device).expand(
            W.shape[0], 1, *spatial, nc, nc
        )
        return torch.cat([identity, W, self.gaugegroup.dagger(W)], dim=1)

    def _attend(self, Q, K, V, T):
        """Fully batched gauge-equivariant attention over the L1-ball.

        Single fused pass — no Python loop over offsets. Pipeline:
          1. Gather K(x+Δx_i), V(x+Δx_i)
          2. Adjoint transport: K̃ = T(x) · K(x+Δx) · T†(x), one fused matmul
             chain that broadcasts T over H and d_qkv.
          3. Scalar score Re Σ_c Tr[Q_c† · K̃_c] / √(d_qkv·nc) computed as a
             Frobenius product, plus the §3.4 orbit-tied bias.
          4. Softmax over the offset axis.
          5. Value path Q† · Ṽ as one batched matmul, weighted by α and
             reduced over the offset axis.
        """
        nc = Q.shape[-1]

        # 1. Neighbour gather.
        idx = tuple(
            self._nbr_idx[d] for d in range(self.D)
        )  # (n_off, *Λ) D dimensional vectors
        # nb_indexer = (:, :, :, ?, :, :) -> ? across dimension *Λ selects neighbors for each lattice site
        nb_indexer = (slice(None),) * 3 + idx + (slice(None), slice(None))
        K_nb = K[
            nb_indexer
        ]  # (B, H, d_qkv, n_off, *Λ, nc, nc) -> for each lattice site and neighbor, a (B, H, d, nc, nc) K tensor
        V_nb = V[nb_indexer]  # same

        # 2. Adjoint transport: T · X · T†. T broadcasts over H and d_qkv.
        T_b = T.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, n_off, *Λ, nc, nc)
        T_b_dag = self.gaugegroup.dagger(T_b)  # same shape (.conj().transpose)
        K_tilde = T_b @ K_nb @ T_b_dag  # (B, H, d_qkv, n_off, *Λ, nc, nc)
        V_tilde = T_b @ V_nb @ T_b_dag

        # 3. Score = Tr[Q_c† K̃_c]/sqrt(Nc d_qkv); Implementable via Frobenius product instead of expensive matmul
        Q_e = Q.unsqueeze(3)  # (B, H, d_qkv, 1, *Λ, nc, nc)
        score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real
        score = score / math.sqrt(self.d_qkv * nc)
        # score: (B, H, n_off, *Λ)

        # 4. Orbit-tied bias: b_h[(Δx_i)], broadcast over (B, *Λ).
        # ``.real`` is defensive: ``b_h`` is semantically real, but a user
        # calling ``block.to(torch.complex128)`` would cast it to complex —
        # we don't want the imaginary part (always 0 with zero init) to
        # contaminate the score and break softmax (no complex softmax kernel).

        # with this slicing we copy the same bias for symmetry-tied offsets.
        # we go from (H, n_unique_sigs) -> (H, n_off)
        bias = self.b_h[:, self._orbit_idx]  # (H, n_off)
        if torch.is_complex(bias):
            bias = bias.real
        score = score + bias.view(1, self.H, self.n_offsets, *([1] * self.D))

        # 5. Softmax over offsets.
        alpha = torch.softmax(score, dim=2)

        # 6. Value path, fully batched over offsets.
        Q_dag = self.gaugegroup.dagger(Q).unsqueeze(3)  # (B, H, d_qkv, 1, *Λ, nc, nc)
        QdagV = torch.matmul(Q_dag, V_tilde)  # (B, H, d_qkv, n_off, *Λ, nc, nc)
        # α: (B, H, n_off, *Λ) → (B, H, 1, n_off, *Λ, 1, 1) to broadcast over
        # the d_qkv and the two color axes.
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        return (alpha_b * QdagV).sum(dim=3)  # (B, H, d_qkv, *Λ, nc, nc)

    def forward(self, W, T):
        """Run the block.

        ``W`` : covariant input field, ``(B, C, *Λ, nc, nc)``.
        ``T`` : precomputed transports, ``(B, n_offsets, *Λ, nc, nc)`` with
        offset axis ordered by ``self.offsets``.

        Returns a tensor of the same shape as ``W``.
        """

        assert T.shape[1] == self.n_offsets, (
            f"Expected T.shape[1] == {self.n_offsets} (number of offsets), got {T.shape[1]}"
        )

        nc = W.shape[-1]

        # Augment, then mix channels to build Q, K, V of shape
        # (B, H, d_qkv, *Λ, nc, nc).
        W_aug = self._augment(W)  # (B, C̃, *Λ, nc, nc), contiguous
        B = W_aug.shape[0]
        trailing = W_aug.shape[2:]  # (*Λ, nc, nc)

        # Collapse dimensions so that matmul broadcasts correctly:
        #   (H·d, C̃) @ (B, C̃, N) -> (B, H·d, N).
        W_aug_flat = W_aug.view(B, self.C_tilde, -1)
        w_Q_flat = self.w_Q.view(self.H * self.d_qkv, self.C_tilde)
        w_K_flat = self.w_K.view(self.H * self.d_qkv, self.C_tilde)
        w_V_flat = self.w_V.view(self.H * self.d_qkv, self.C_tilde)

        Q = torch.matmul(w_Q_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)
        K = torch.matmul(w_K_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)
        V = torch.matmul(w_V_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)

        # Transport, score, softmax, multiplicative value.
        out = self._attend(Q, K, V, T)  # (B, H, d_qkv, *Λ, nc, nc)

        # Channel mix back to C output channels.
        W_mix = torch.einsum("iha,bha...->bi...", self.w_mix, out)  # (B, C, *Λ, nc, nc)

        # Residual + L-Act gate. The gate is a real scalar per (B, C, x).
        W_res = W + W_mix
        trace_per_chan = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
        if self.gate == "relu":
            g = F.relu(trace_per_chan)
        else:
            g = F.softplus(trace_per_chan)
        W_act = g.unsqueeze(-1).unsqueeze(-1) * W_res
        # ReZero: blend toward the L-Act output with a per-block scalar α
        # (zero-init). At α=0 the block is bit-exactly the identity W → W;
        # during training α grows and the gate/mix path takes over.
        return W + self.alpha * (W_act - W)


class Trace(nn.Module):
    """Trace block: outputs the trace of the input field as a scalar per site.

    This is a gauge-invariant quantity, so it can be used for supervised
    regression tasks or as a readout head for classification.
    """

    def forward(self, W):
        # W: (B, C, *Λ, nc, nc) -> trace over color
        trace = W.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, C, *Λ)
        out = torch.cat([trace.real, trace.imag], dim=1)  # (B, 2C, *Λ)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # (B, 2C, *Λ) -> (B, *Λ, 2C) so nn.Linear acts on the channel axis.
        # reshape() would reinterpret memory and scramble the per-site vectors;
        # movedim is the permutation we actually want.
        x = x.movedim(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GELT(nn.Module):
    """Full GELT model:
    Pipeline:
      1. Compute Plaq (+ optional Poly)
      2. GEMHSA blocks with H heads and d_qkv channels per head.
      3. Trace block to get Re, Im parts of the trace as scalar per site.
      4. MLP with one hidden layer to mix the trace features and output a scalar per site for regression or classification.
    """

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
        # Plaquette input -> D(D-1)/2 plaquettes per site.
        d_input = D * (D - 1) // 2
        super(GELT, self).__init__()
        # ModuleList so the GEMHSA parameters are registered with PyTorch
        # and picked up by .parameters() / .to() / .state_dict().
        self.gemhsa_models = nn.ModuleList(
            [
                GEMHSA(gaugegroup, L, D, R, d_input, nhead, d_qkv, gate, dtype)
                for i in range(gemhsa_layers)
            ]
        )

        # Trace produces real values, so the MLP must live in the matching
        # real dtype — not the complex `dtype` of the GEMHSA stack. Blanket
        # `.to(complex_dtype)` on GELT would otherwise miscast the MLP.
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.mlp = MLP(2 * d_input, mlp_hidden, mlp_out).to(real_dtype)

    def attn(self, W, T):
        for layer in self.gemhsa_models:
            W = layer(W, T)
        return W

    def forward(self, W, T):
        W_attn = self.attn(W, T)
        trace = self.trace(W_attn)
        site_out = self.mlp(trace)  # (B, *Λ, mlp_out)
        # Sum the site-local readout to an extensive scalar per config
        # (matches the Wilson action target). squeeze(-1) handles mlp_out=1.
        spatial_dims = tuple(range(1, site_out.ndim - 1))
        return site_out.sum(dim=spatial_dims).squeeze(-1)
