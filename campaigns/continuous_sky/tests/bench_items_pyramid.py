"""Cost of the item rasterisation, direct against pyramid (F40): forward + backward CPU time (process CPU seconds, one thread, float32 as the
GPU runs) and the number of patch nodes actually evaluated (bucketed patch sizes, summed over items; for the pyramid also the dense nodes of
the coarse levels that the cascade touches).

States: 'seed' = the data-blind seeding lattice (--item-levels 3); 'widened' = both log widths of every item drawn independently and
uniformly in log between the floor and 0.32 arcsec (seed 11), angles uniform; 'widened_iso' = one such draw per item used for both
widths (round items). --item-max-width 0.32 in both (the recommended SMACS arguments).
Optional subsampling: --sub f keeps a random fraction f of the items (same draw for both modes) and the times and node counts are scaled
by 1 / f (stated in the output).
Run: python tests/bench_items_pyramid.py N pitch state mode [--sub f] [--reps r]   (one configuration per process, appends to
tests/items_results/bench_pyramid.jsonl)"""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.sky import ItemSky, RBFBasis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("N", type=int)
ap.add_argument("pitch", type=float)
ap.add_argument("state", choices=["seed", "widened", "widened_iso"])
ap.add_argument("mode", choices=["direct", "pyramid"])
ap.add_argument("--items", type=int, default=60000)
ap.add_argument("--sub", type=float, default=1.0)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--patch", choices=["square", "sheared"], default="square")
ap.add_argument("--max-width", type=float, default=0.32)
a = ap.parse_args()

K = 12
torch.manual_seed(0)
A0 = torch.rand(K, a.N, a.N) * 0.01
s = ItemSky(A0, a.pitch, RBFBasis(K), a.items, max_width=a.max_width, n_levels=3, raster=a.mode, patch=a.patch)
with torch.no_grad():
    if a.state.startswith("widened"):
        g = torch.Generator().manual_seed(11)
        w = torch.exp(math.log(s.floor) + torch.rand(a.items, 2, generator=g, dtype=torch.float64) * math.log(0.32 / s.floor))
        if a.state == "widened_iso":
            w = w[:, :1].repeat(1, 2)          # one draw per item, both widths equal
        s.lw.data[0] = (0.5 * torch.log((w ** 2 - s.floor ** 2).clamp(min=(0.05 * s.floor) ** 2))).float()
        s.ang.data[0] = (torch.rand(a.items, generator=g, dtype=torch.float64) * math.pi).float()
    if a.sub < 1:
        keep = torch.randperm(a.items, generator=torch.Generator().manual_seed(5))[:int(round(a.sub * a.items))]
        n = keep.numel()
        s.pos = torch.nn.Parameter(s.pos.data[:, keep].clone())
        s.lw = torch.nn.Parameter(s.lw.data[:, keep].clone())
        s.ang = torch.nn.Parameter(s.ang.data[:, keep].clone())
        s.lc = torch.nn.Parameter(s.lc.data[:, keep].clone())
        s.n_items = n
    s._mut += 1
scale = 1.0 / a.sub if a.sub < 1 else 1.0

# patch nodes
with torch.no_grad():
    wid = s.widths(0)
    nodes = {}
    if a.mode == "direct":
        nodes["level0"] = sum((2 * h + 1) ** 2 * int(i.numel()) for h, i in s._buckets(wid)) * scale
        dense = 0
        levels = [s.n_items]
    else:
        pl = s.plan()
        lev = pl.assign(wid)
        levels = torch.bincount(lev, minlength=pl.L + 1).tolist()
        for l in range(pl.L + 1):
            sel = torch.nonzero(lev == l).squeeze(1)
            if sel.numel() == 0:
                continue
            obj = pl.level_grid(l, K, shear=a.patch == "sheared") if (l > 0 or a.patch == "sheared") else s
            hh = lambda h: (2 * h[0] + 1) * (2 * h[1] + 1) if isinstance(h, tuple) else (2 * h + 1) ** 2  # noqa: E731
            nodes[f"level{l}"] = sum(hh(h) * int(i.numel()) for h, i in obj._buckets(wid[sel], s.ang[0][sel])) * scale
        dense = sum(pl.n[l] ** 2 for l in range(1, int(lev.max()) + 1))
W = torch.randn(K, a.N, a.N)
ts = []
for _ in range(a.reps):
    for t in (s.pos, s.lw, s.ang, s.lc):
        t.grad = None
    t0 = time.process_time()
    L = (W * s.raster(0)).sum()
    t1 = time.process_time()
    L.backward()
    t2 = time.process_time()
    ts.append((t1 - t0, t2 - t1))
fw = min(t[0] for t in ts) * scale
bw = min(t[1] for t in ts) * scale
out = dict(N=a.N, pitch=a.pitch, state=a.state, mode=a.mode, patch=a.patch, items=a.items, subsample=a.sub, scaled_by=scale, reps=a.reps, K=K,
           items_per_level=[int(x * scale) for x in levels], patch_nodes=nodes, patch_nodes_total=sum(nodes.values()),
           cascade_dense_nodes_per_coefficient=dense, cpu_s_forward=fw, cpu_s_backward=bw, cpu_s_total=fw + bw)
print(json.dumps(out))
with open(os.path.join(os.path.dirname(__file__), "items_results", "bench_pyramid.jsonl"), "a") as f:
    f.write(json.dumps(out) + "\n")
