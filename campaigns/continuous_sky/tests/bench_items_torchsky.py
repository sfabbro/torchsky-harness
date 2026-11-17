"""Time and peak memory of ONE forward + backward of the item raster, ItemSky's own code against GaussianItemScene.
   python tests/bench_items_torchsky.py --impl {prototype,torchsky} --raster {direct,pyramid} [--items 60000] [--n 601] [--k 12] [--reps 2] [--chunk 4000000]
Run each combination in its own process: the peak is the process's resident-set maximum (kernel counter VmHWM), which includes the 12 x N x N coefficient
planes and the gradient.  Items: ItemSky's lattice seeding (76 / 19 / 5 per cent on three lattices), then widths multiplied by a log-normal factor
(sigma 0.7 in the log) and random angles, so that the sizes spread as they do after some fitting; one thread; float32 (the prototype's training dtype)."""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def vmhwm_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="prototype")
    ap.add_argument("--raster", default="direct")
    ap.add_argument("--items", type=int, default=60000)
    ap.add_argument("--n", type=int, default=601)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--pitch", type=float, default=0.04)
    ap.add_argument("--max-width", type=float, default=2.56)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from fieldop import sky as S
    if a.chunk > 0:
        S.ITEM_CHUNK = a.chunk
    g = torch.Generator().manual_seed(0)
    A0 = torch.rand(a.k, a.n, a.n, generator=g) + 0.1
    base = vmhwm_mib()
    sky = S.ItemSky(A0, a.pitch, S.RBFBasis(a.k), a.items, max_width=a.max_width, raster=a.raster, impl=a.impl)
    with torch.no_grad():
        sky.lw.data[0] += 0.7 * torch.randn(a.items, 2, generator=g)
        sky.lw.data[0].clamp_(max=math.log(a.max_width))
    wgt = torch.randn(a.k, a.n, a.n, generator=g)
    ts, fs = [], []
    res = {}
    for r in range(a.reps + 1):                      # the first repetition is a warm-up (caches, allocator)
        for t in (sky.pos, sky.lw, sky.ang, sky.lc):
            t.grad = None
        t0 = time.perf_counter()
        out = sky.raster(0)
        t1 = time.perf_counter()
        (out * wgt).sum().backward()
        t2 = time.perf_counter()
        if r > 0:
            fs.append(t1 - t0)
            ts.append(t2 - t0)
    res = {"impl": a.impl, "raster": a.raster, "items": a.items, "grid": a.n, "K": a.k, "chunk_elements": S.ITEM_CHUNK, "forward_s": min(fs), "forward_backward_s": min(ts),
           "all_forward_backward_s": ts, "peak_rss_MiB": vmhwm_mib(), "rss_before_build_MiB": base,
           "width_px_median": float(torch.exp(sky.lw.data[0]).median() / a.pitch), "width_px_p99": float(torch.quantile(torch.exp(sky.lw.data[0]).flatten(), 0.99) / a.pitch)}
    print(json.dumps(res))
    if a.out:
        open(a.out, "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
