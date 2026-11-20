"""Checksum of the direct item rasteriser (values and gradients), to show that --item-raster direct is bit-identical before and after
the pyramid change. Run: python tests/checksum_direct.py  (prints one sha256 per case)."""
import hashlib
import math
import os
import sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.sky import ItemSky, RBFBasis


def case(dt, widen, max_width):
    torch.manual_seed(0)
    K, N, p = 12, 95, 0.06
    A0 = torch.rand(K, N, N, dtype=dt) * 0.01
    s = ItemSky(A0, p, RBFBasis(K), 1500, max_width=max_width)
    if widen:
        g = torch.Generator().manual_seed(7)
        w = torch.exp(math.log(s.floor) + torch.rand(1500, 2, generator=g, dtype=torch.float64) * math.log(max_width / s.floor))
        s.lw.data[0] = (0.5 * torch.log((w ** 2 - s.floor ** 2).clamp(min=1e-12))).to(dt)
        s.ang.data[0] = (torch.rand(1500, generator=g, dtype=torch.float64) * math.pi).to(dt)
    W = torch.randn(K, N, N, dtype=dt, generator=torch.Generator().manual_seed(3))
    A = s.coeffs()
    (W * A).sum().backward()
    h = hashlib.sha256()
    for t in (A, s.pos.grad, s.lw.grad, s.ang.grad, s.lc.grad):
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


for dt in (torch.float64, torch.float32):
    for widen, mw in ((False, 0.0), (True, 0.06 * 40)):
        print(dt, "widened" if widen else "seed", mw, case(dt, widen, mw))
