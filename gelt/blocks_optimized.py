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
        self.w_Q = nn.Parameter(self._init_projection((self.H, self.d_qkv, self.C_tilde), sigma, dtype))
        self.w_K = nn.Parameter(self._init_projection((self.H, self.d_qkv, self.C_tilde), sigma, dtype))
        self.w_V = nn.Parameter(self._init_projection((self.H, self.d_qkv, self.C_tilde), sigma, dtype))
        
        sigma_mix = 0.02 / math.sqrt(self.H * self.d_qkv)
        self.w_mix = nn.Parameter(self._init_projection((self.C, self.H, self.d_qkv), sigma_mix, dtype))
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _init_projection(shape, sigma, dtype):
        if dtype.is_complex:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            re = torch.randn(*shape, dtype=real_dtype) * sigma
            im = torch.randn(*shape, dtype=real_dtype) * sigma
            return torch.complex(re, im)
        return torch.randn(*shape, dtype=dtype) * sigma

    def _resolve_dagger(self, U):
        return U.transpose(-1, -2).conj().resolve_conj().contiguous()

    def _attend(self, Q, K, V, T):
        nc = Q.shape[-1]
        B, H, d_qkv = Q.shape[:3]
        spatial_shape = Q.shape[3:-2]
        
        idx = tuple(self._nbr_idx[d] for d in range(self.D))
        nb_indexer = (slice(None), slice(None), slice(None)) + idx + (slice(None), slice(None))
        
        K_nb = K.contiguous()[nb_indexer].contiguous()
        V_nb = V.contiguous()[nb_indexer].contiguous()

        T_b = T.unsqueeze(1).unsqueeze(1).contiguous()
        T_b_dag = self._resolve_dagger(T_b)
        
        K_tilde = (T_b @ K_nb @ T_b_dag).contiguous()
        V_tilde = (T_b @ V_nb @ T_b_dag).contiguous()

        # Score calculation: Tr[Q† @ K_tilde]
        Q_conj = Q.conj().resolve_conj().contiguous()
        # Frobenius product via matmul + diagonal
        # Q_conj: (B, H, d, *Λ, nc, nc), K_tilde: (B, H, d, n, *Λ, nc, nc)
        Q_conj_e = Q_conj.unsqueeze(3).contiguous()
        score = (Q_conj_e @ K_tilde).diagonal(dim1=-2, dim2=-1).sum(-1).real
        # Sum over head dimension d
        score = score.sum(dim=2) / math.sqrt(self.d_qkv * nc) # (B, H, n, *Λ)

        bias = self.b_h[:, self._orbit_idx].contiguous()
        if torch.is_complex(bias): bias = bias.real
        score = score + bias.view(1, self.H, self.n_offsets, *([1] * self.D))

        alpha = torch.softmax(score, dim=2).to(Q.dtype) # (B, H, n, *Λ)

        # Value path: alpha * (Q† @ V_tilde)
        QdagV = (Q_conj_e @ V_tilde).contiguous() # (B, H, d, n, *Λ, nc, nc)
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1) # (B, H, 1, n, *Λ, 1, 1)
        return (alpha_b * QdagV).sum(dim=3).contiguous() # (B, H, d, *Λ, nc, nc)

    def forward(self, W, T):
        nc = W.shape[-1]
        B = W.shape[0]
        spatial = W.shape[2:-2]
        trailing = W.shape[2:]
        
        W = W.contiguous()
        W_dag = self._resolve_dagger(W)
        
        # Fused QKV Projections via matmul
        # Identity part
        qkv_id = torch.stack([self.w_Q[:,:,0], self.w_K[:,:,0], self.w_V[:,:,0]]) # (3, H, d)
        eye = torch.eye(nc, dtype=W.dtype, device=W.device)
        QKV_id = qkv_id.view(3, 1, self.H, self.d_qkv, 1, 1).matmul(eye.view(1, 1, 1, 1, nc, nc))
        QKV_id = QKV_id.view(3, 1, self.H, self.d_qkv, *([1]*len(spatial)), nc, nc).contiguous()
        
        # W and W_dag part
        # w_W: (3, H, d, C), W: (B, C, N*nc*nc)
        w_W = torch.stack([self.w_Q[:,:,1:self.C+1], self.w_K[:,:,1:self.C+1], self.w_V[:,:,1:self.C+1]])
        w_W_dag = torch.stack([self.w_Q[:,:,self.C+1:], self.w_K[:,:,self.C+1:], self.w_V[:,:,self.C+1:]])
        
        W_flat = W.view(B, self.C, -1)
        W_dag_flat = W_dag.view(B, self.C, -1)
        
        # (3, H, d, C) @ (B, C, N*nc*nc) -> (3, B, H, d, N*nc*nc)
        QKV_W = torch.matmul(w_W.unsqueeze(1), W_flat.unsqueeze(0)) + \
                torch.matmul(w_W_dag.unsqueeze(1), W_dag_flat.unsqueeze(0))
        QKV_W = QKV_W.view(3, B, self.H, self.d_qkv, *trailing).contiguous()
        
        QKV = QKV_id + QKV_W
        Q, K, V = QKV[0], QKV[1], QKV[2]

        out = self._attend(Q, K, V, T) # (B, H, d, *Λ, nc, nc)

        # Output Mix: (C, H, d) @ (B, H, d, *Λ, nc, nc) -> (B, C, *Λ, nc, nc)
        # Reshape for matmul: (H*d, C) and (B, H*d, N*nc*nc)
        w_mix_flat = self.w_mix.view(self.C, self.H * self.d_qkv)
        out_flat = out.view(B, self.H * self.d_qkv, -1)
        W_mix = torch.matmul(w_mix_flat, out_flat).view(B, self.C, *trailing).contiguous()

        # Residual + Gate
        W_res = (W + W_mix).contiguous()
        trace = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
        g = F.softplus(trace) if self.gate == "softplus" else F.relu(trace)
        W_act = (g.to(W.dtype).unsqueeze(-1).unsqueeze(-1) * W_res).contiguous()
        return (W + self.alpha.to(W.dtype) * (W_act - W)).contiguous()

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
