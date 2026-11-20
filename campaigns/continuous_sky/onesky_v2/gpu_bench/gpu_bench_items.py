"""GPU timing of the item rasterisation (forward + backward), direct against pyramid, for several chunk sizes.
Run: python gpu_bench_items.py <folder holding fieldop/> <out.json>.  States as tests/bench_items_pyramid.py (seed lattice; 'widened' = both
log widths drawn independently, uniform in log between the floor and 0.32 arcsec; 'widened_iso' = one draw for both widths)."""
import json
import math
import sys
import time
import torch
sys.path.insert(0, sys.argv[1])
import fieldop.sky as SK
from fieldop.sky import ItemSky, RBFBasis
dev = "cuda"
K, n_items = 12, 60000
out = []
def build(N, pitch, state, mode, patch):
    torch.manual_seed(0)
    A0 = (torch.rand(K, N, N) * 0.01).to(dev)
    s = ItemSky(A0, pitch, RBFBasis(K).to(dev), n_items, max_width=0.32, n_levels=3, raster=mode, patch=patch)
    with torch.no_grad():
        if state.startswith("widened"):
            g = torch.Generator().manual_seed(11)
            w = torch.exp(math.log(s.floor) + torch.rand(n_items, 2, generator=g, dtype=torch.float64) * math.log(0.32 / s.floor))
            if state == "widened_iso":
                w = w[:, :1].repeat(1, 2)
            s.lw.data[0] = (0.5 * torch.log((w ** 2 - s.floor ** 2).clamp(min=(0.05 * s.floor) ** 2))).float().to(dev)
            s.ang.data[0] = (torch.rand(n_items, generator=g, dtype=torch.float64) * math.pi).float().to(dev)
        s._mut += 1
    return s
def timeit(s, N, reps=3):
    W = torch.randn(K, N, N, device=dev)
    ts = []
    for r in range(reps + 1):
        for t in s.parameters():
            t.grad = None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        A = s.coeffs(); torch.cuda.synchronize(); t1 = time.perf_counter()
        (W * A).sum().backward(); torch.cuda.synchronize(); t2 = time.perf_counter()
        if r > 0:
            ts.append((t1 - t0, t2 - t1))
    f = sorted(x[0] for x in ts)[len(ts) // 2]; b = sorted(x[1] for x in ts)[len(ts) // 2]
    return f, b
for N, pitch in ((613, 0.04), (1226, 0.02)):
    for state in ("seed", "widened", "widened_iso"):
        for mode, patch in (("direct", "square"), ("pyramid", "square"), ("pyramid", "sheared")):
            for chunk in (4_000_000, 32_000_000, 128_000_000):
                SK.ITEM_CHUNK = chunk
                try:
                    torch.cuda.reset_peak_memory_stats()
                    s = build(N, pitch, state, mode, patch)
                    f, b = timeit(s, N)
                    rec = dict(N=N, pitch=pitch, state=state, mode=mode, patch=patch, chunk=chunk, fwd_s=f, bwd_s=b,
                               peak_gb=torch.cuda.max_memory_allocated() / 1e9)
                except Exception as e:  # noqa: BLE001
                    rec = dict(N=N, pitch=pitch, state=state, mode=mode, patch=patch, chunk=chunk, error=repr(e)[:200])
                print(json.dumps(rec), flush=True); out.append(rec)
                del s; torch.cuda.empty_cache()
# where does the time go in the slowest sensible case: operator table
from torch.profiler import profile, ProfilerActivity
SK.ITEM_CHUNK = 4_000_000
for mode, patch in (("direct", "square"), ("pyramid", "sheared")):
    s = build(613, 0.04, "widened", mode, patch); W = torch.randn(K, 613, 613, device=dev)
    (W * s.coeffs()).sum().backward()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        (W * s.coeffs()).sum().backward(); torch.cuda.synchronize()
    print(mode, patch); print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=14, max_name_column_width=40))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10, max_name_column_width=40))
json.dump(out, open(sys.argv[2], "w"), indent=1)
