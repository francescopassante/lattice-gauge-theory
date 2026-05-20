import time
import torch
import torch.nn as nn
import torch.optim as optim

from gelt import SU, build_plaquette_datasets, haar_ensemble
from gelt.blocks import GELT as OriginalGELT
from gelt.blocks_optimized import GELT as OptimizedGELT

def run_bench(model_class, model_parameters, train_loader, val_loader, device, name="Model", compile=False, use_amp=False):
    print(f"\n--- Benchmarking {name} (compile={compile}, amp={use_amp}) ---")
    model = model_class(**model_parameters).to(device)
    
    if compile and hasattr(torch, "compile") and device.type != "mps":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[{name}] model compiled")
        except Exception as e:
            print(f"[{name}] model compilation failed: {e}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # GradScaler does not support ComplexFloat gradients.
    # We use autocast but skip the scaler if model has complex params.
    has_complex = any(p.is_complex() for p in model.parameters())
    scaler_enabled = use_amp and not has_complex
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    
    if use_amp and has_complex:
        print(f"[{name}] Note: GradScaler disabled (ComplexFloat not supported). Autocast remains active.")

    X, T, y = next(iter(train_loader))
    Xd, Td, yd = X.to(device, non_blocking=True), T.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(Xd, Td)
            loss = criterion(out, yd)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Single Step timing
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(Xd, Td)
        loss = criterion(out, yd)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = time.perf_counter() - t0
    print(f"[{name}] single fwd+bwd+step (B={Xd.shape[0]}): {t_step * 1000:.1f} ms")

    # Epoch timing
    epochs = 2
    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        n_batches = 0
        for Xb, Tb, yb in train_loader:
            Xb, Tb, yb = Xb.to(device, non_blocking=True), Tb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(Xb, Tb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n_batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[{name}] epoch {epoch+1} train: {dt:.2f}s ({dt/n_batches*1000:.0f} ms/batch)")

def main():
    torch.manual_seed(0)
    D, L, R = 3, 8, 2
    gaugegroup = SU(2)

    dataset_parameters = {
        "N": 100,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": False,
        "structured": True,
        "sampler": haar_ensemble,
        "beta": 1,
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32,
    }

    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 2, 
        "gemhsa_layers": 2, 
        "d_qkv": 8, 
        "gate": "softplus",
        "dtype": torch.complex64,
        "mlp_hidden": 16,
        "mlp_out": 1,
    }

    t0 = time.perf_counter()
    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )
    print(f"[data] build: {time.perf_counter() - t0:.2f}s")

    # Bump batch size for V100
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")
    
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_bench(OriginalGELT, model_parameters, train_loader, val_loader, device, name="Original")
    run_bench(OptimizedGELT, model_parameters, train_loader, val_loader, device, name="Optimized")
    run_bench(OptimizedGELT, model_parameters, train_loader, val_loader, device, name="Optimized+AMP", use_amp=True)
    
    if device.type != "mps":
        run_bench(OptimizedGELT, model_parameters, train_loader, val_loader, device, name="Optimized+Compiled", compile=True)
        run_bench(OptimizedGELT, model_parameters, train_loader, val_loader, device, name="Optimized+AMP+Compiled", compile=True, use_amp=True)

if __name__ == "__main__":
    main()
