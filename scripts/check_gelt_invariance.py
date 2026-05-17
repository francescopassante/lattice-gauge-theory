"""Quick gauge-invariance check on the full GELT model: forward(W_g, T_g) ≈ forward(W, T)."""

import torch

from gelt import (
    SU,
    build_transport_sums,
    l1_ball_offsets,
    link_gauge_transformation,
    plaquette_tensor,
    random_links,
)
from gelt.blocks import GELT

torch.manual_seed(0)
L, D, R, nc, H, layers = 4, 2, 2, 2, 2, 2
gg = SU(nc)

U = random_links(L=L, D=D, gaugegroup=gg, dtype=torch.complex64)
raw = torch.randn(L**D, nc, nc) + 1j * torch.randn(L**D, nc, nc)
omega, _ = torch.linalg.qr(raw)
omega = omega.reshape(*([L] * D), nc, nc).to(torch.complex64)

U_g = link_gauge_transformation(U, omega, gg)
P = plaquette_tensor(U, gg).unsqueeze(0)
P_g = plaquette_tensor(U_g, gg).unsqueeze(0)

offsets = l1_ball_offsets(D, R)
T = torch.stack([build_transport_sums(U, R=R, gaugegroup=gg)[o] for o in offsets], dim=0).unsqueeze(0)
T_g = torch.stack([build_transport_sums(U_g, R=R, gaugegroup=gg)[o] for o in offsets], dim=0).unsqueeze(0)

model = GELT(gaugegroup=gg, L=L, D=D, R=R, nhead=H, gemhsa_layers=layers, d_qkv=2)

out = model(P, T)
out_g = model(P_g, T_g)
drift = (out_g - out).abs().max().item()
print(f"|out_g - out| max = {drift:.3e}    (out = {out.item():.6e}, out_g = {out_g.item():.6e})")
