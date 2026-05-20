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
        
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate!r}")
        self.gate = gate

        self.offsets = l1_ball_offsets(D, R)
        self.n_offsets = len(self.offsets)

        offset_tensor = torch.tensor(self.offsets, dtype=torch.long)
        coords = torch.meshgrid(*[torch.arange(L) for _ in range(D)], indexing="ij")
        nbr_idx = torch.stack([
            (coords[d].unsqueeze(0) + offset_tensor[:, d].view(-1, *([1] * D))) % L
            for d in range(D)
        ], dim=0)
        self.register_buffer("_nbr_idx", nbr_idx)

        sigs = [tuple(sorted((abs(d) for d in dx), reverse=True)) for dx in self.offsets]
        unique_sigs = sorted(set(sigs))
        sig_to_idx = {s: i for i, s in enumerate(unique_sigs)}
        self.n_orbits = len(unique_sigs)
        orbit_idx = torch.tensor([sig_to_idx[s] for s in sigs], dtype=torch.long)
        self.register_buffer("_orbit_idx", orbit_idx)
        self.b_h = nn.Parameter(torch.zeros(self.H, self.n_orbits))

        self.C_tilde = 2 * d_input + 1
        sigma = 0.02 / math.sqrt(self.C_tilde)
        self.w_Q = self._init_projection((self.H, self.d_qkv, self.C_tilde), sigma, dtype)
        self.w_K = self._init_projection((self.H, self.d_qkv, self.C_tilde), sigma, dtype)
        self.w_V = self._init_projection((self.H, self.d_qkv, self.C_tilde), sigma, dtype)
        
        sigma_mix = 0.02 / math.sqrt(self.H * self.d_qkv)
        self.w_mix = self._init_projection((self.C, self.H, self.d_qkv), sigma_mix, dtype)
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _init_projection(shape, sigma, dtype):
        if dtype.is_complex:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            re = torch.randn(*shape, dtype=real_dtype) * sigma
            im = torch.randn(*shape, dtype=real_dtype) * sigma
            return nn.Parameter(torch.complex(re, im))
        return nn.Parameter(torch.randn(*shape, dtype=dtype) * sigma)

    def _attend(self, Q, K, V, T):
        # Fully batched version for speed, with aggressive contiguity for torch.compile
        nc = Q.shape[-1]
        B, H, d_qkv = Q.shape[:3]
        spatial = Q.shape[3:-2]
        
        # 1. Neighbour gather
        idx = tuple(self._nbr_idx[d] for d in range(self.D))
        nb_indexer = (slice(None), slice(None), slice(None)) + idx + (slice(None), slice(None))
        
        # Ensure K, V are contiguous before slicing to keep inductor happy
        K_nb = K.contiguous()[nb_indexer].contiguous()
        V_nb = V.contiguous()[nb_indexer].contiguous()

        # 2. Adjoint transport
        T_b = T.unsqueeze(1).unsqueeze(1).contiguous()
        T_b_dag = self.gaugegroup.dagger(T_b).contiguous()
        
        K_tilde = (T_b @ K_nb @ T_b_dag).contiguous()
        V_tilde = (T_b @ V_nb @ T_b_dag).contiguous()

        # 3. Score calculation
        Q_e = Q.unsqueeze(3).contiguous()
        score = (Q_e.conj().contiguous() * K_tilde).sum(dim=(2, -2, -1)).real
        score = score / math.sqrt(self.d_qkv * nc)

        # 4. Bias
        bias = self.b_h[:, self._orbit_idx].contiguous()
        if torch.is_complex(bias):
            bias = bias.real
        score = score + bias.view(1, self.H, self.n_offsets, *([1] * self.D))

        # 5. Softmax
        alpha = torch.softmax(score, dim=2)

        # 6. Value path
        Q_dag = self.gaugegroup.dagger(Q).unsqueeze(3).contiguous()
        QdagV = torch.matmul(Q_dag, V_tilde).contiguous()
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).contiguous()
        
        return (alpha_b * QdagV).sum(dim=3).contiguous()

    def forward(self, W, T):
        nc = W.shape[-1]
        B = W.shape[0]
        spatial = W.shape[2:-2]
        trailing = W.shape[2:]

        # Optimized QKV: avoid torch.cat, use fused weights
        W = W.contiguous()
        W_flat = W.view(B, self.C, -1)
        W_dag = self.gaugegroup.dagger(W).contiguous()
        W_dag_flat = W_dag.view(B, self.C, -1)
        
        w_QKV = torch.stack([
            self.w_Q.reshape(-1, self.C_tilde),
            self.w_K.reshape(-1, self.C_tilde),
            self.w_V.reshape(-1, self.C_tilde)
        ]).contiguous()
        
        qkv_id = w_QKV[:, :, 0].view(3, 1, self.H, self.d_qkv, *([1] * (len(spatial) + 2)))
        eye = torch.eye(nc, dtype=W.dtype, device=W.device)
        eye = eye.view(1, 1, 1, 1, *([1] * len(spatial)), nc, nc)
        QKV = (qkv_id * eye).contiguous()
        
        w_W = w_QKV[:, :, 1 : self.C + 1].unsqueeze(1)
        w_W_dag = w_QKV[:, :, self.C + 1 :].unsqueeze(1)
        
        QKV_W = torch.matmul(w_W, W_flat.unsqueeze(0)) + torch.matmul(w_W_dag, W_dag_flat.unsqueeze(0))
        QKV_W = QKV_W.view(3, B, self.H, self.d_qkv, *trailing).contiguous()
        
        QKV = QKV + QKV_W
        Q, K, V = QKV[0].contiguous(), QKV[1].contiguous(), QKV[2].contiguous()

        out = self._attend(Q, K, V, T)

        W_mix = torch.einsum("iha,bha...->bi...", self.w_mix, out).contiguous()

        # Residual + Gate: be very careful with contiguity for Inductor complex add decomposition
        W_res = (W + W_mix).contiguous()
        trace_per_chan = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
        g = F.softplus(trace_per_chan) if self.gate == "softplus" else F.relu(trace_per_chan)
        W_act = (g.unsqueeze(-1).unsqueeze(-1) * W_res).contiguous()
        
        # ReZero update
        update = (W_act - W).contiguous()
        return (W + self.alpha * update).contiguous()

class Trace(nn.Module):
    def forward(self, W):
        W = W.contiguous()
        trace = W.diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.cat([trace.real, trace.imag], dim=1).contiguous()

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = x.movedim(1, -1).contiguous()
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
        self.gemhsa_models = nn.ModuleList([
            GEMHSA(gaugegroup, L, D, R, d_input, nhead, d_qkv, gate, dtype)
            for _ in range(gemhsa_layers)
        ])
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.mlp = MLP(2 * d_input, mlp_hidden, mlp_out).to(real_dtype)
        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)

    def forward(self, W, T):
        for layer in self.gemhsa_models:
            W = layer(W, T)
        trace = self.trace(W)
        site_out = self.mlp(trace)
        spatial_dims = tuple(range(1, site_out.ndim - 1))
        return site_out.sum(dim=spatial_dims).squeeze(-1)
