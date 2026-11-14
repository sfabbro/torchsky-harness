"""Cost of the item rasterisation (forward + backward) on CPU, one thread, at the synthetic size and at real-data sizes (seeded lattices,
initial widths; widths change during a fit, so the real-data numbers are for the seeding state). Also counts the patch nodes (nnz) that set
the GPU memory: about nnz x (K + 4) x 4 bytes of transient tensors per chunk-free pass (chunks of ITEM_CHUNK elements bound it).
Run: python tests/bench_items_raster.py"""
import json, math, os, sys, time
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.sky import ItemSky, RBFBasis, ITEM_R

def bench(N, pitch, n_items, K=12, reps=2):
    A0 = torch.rand(K, N, N) * 0.01
    s = ItemSky(A0, pitch, RBFBasis(K), n_items)
    with torch.no_grad():
        w = s.widths(0); q = pitch ** 2 / 12
        h = torch.ceil(ITEM_R * torch.sqrt(w.max(1).values ** 2 + q) / pitch)
        nnz = float(((2 * h + 1) ** 2).sum())
    W = torch.randn(K, N, N)
    t = time.perf_counter()
    for _ in range(reps):
        L = (W * s.coeffs()).sum(); L.backward()
    dt = (time.perf_counter() - t) / reps
    return {"N": N, "pitch": pitch, "items": n_items, "K": K, "patch_nodes_nnz": nnz, "cpu_s_fwd_bwd": dt,
            "grid_numbers": K * N * N, "item_numbers": s.n_fitted()}

out = [bench(95, 0.06, 1500), bench(613, 0.04, 60000, reps=1)]
for r in out: print(json.dumps(r))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "items_results", "bench_raster.json"), "w"), indent=1)
