"""Is the fine structure of the fitted sky determined by the data? Three fits that differ only by the random seed (same exposures, same
settings) are compared, per filter, as images on the sky grid (0.04 arcsec), unconvolved and after a Gaussian blur.

Metric: unreproduced fraction U(s) = sum((A_s - B_s)^2) / sum(A_s^2 + B_s^2), A and B the two skies blurred with a Gaussian of standard
deviation s. 0 = identical, 1 = unrelated. Also the share of each sky's total flux that sits in pixels where the other seed has less than
a tenth of it ("flux with no counterpart"), at s = 0. Only the central 16 arcsec (edges are constrained by fewer exposures)."""
import sys
import json
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torchsky.map.exposure_fit import SkyFit
S = sys.argv[1]; runs = sys.argv[2:]
torch.set_num_threads(2)
imgs = {}
for r in runs:
    fit = SkyFit.load(f"{S}/fits/{r}", device="cpu"); op = fit.operator
    with torch.no_grad():
        A = op.sky.coeffs()                                   # (K, N, N)
        imgs[r] = torch.einsum("bk,kyx->byx", op.Phi_band.to(A.dtype), A).float()
    bands, pitch = list(fit.bands), float(op.sky.pitch)
N = imgs[runs[0]].shape[-1]; c = N // 2; h = int(8.0 / pitch)
def blur(x, s_px):
    if s_px <= 0: return x
    n = int(4 * s_px) + 1; g = torch.exp(-0.5 * (torch.arange(-n, n + 1) / s_px) ** 2); g = (g / g.sum()).to(x)
    x = F.conv2d(x[:, None], g[None, None, :, None], padding=(n, 0)); x = F.conv2d(x, g[None, None, None, :], padding=(0, n)); return x[:, 0]
scales = [0, 20, 40, 80, 160]
out = {"scales_mas": scales, "bands": bands, "U": {}, "orphan": {}}
print(f"sky grid {N} x {N} at {1e3 * pitch:.0f} mas; bands {bands}; runs {runs}")
print("metric: unreproduced fraction U(s) between seeds, median over the seed pairs; s = Gaussian blur [mas]")
for b, name in enumerate(bands):
    row = []
    for s in scales:
        u = []
        for a_, b_ in itertools.combinations(runs, 2):
            A = blur(imgs[a_][b:b + 1], s / (1e3 * pitch))[0, c - h:c + h, c - h:c + h]; B = blur(imgs[b_][b:b + 1], s / (1e3 * pitch))[0, c - h:c + h, c - h:c + h]
            u.append(float(((A - B) ** 2).sum() / (A ** 2 + B ** 2).sum()))
        row.append(float(np.median(u)))
    orph = []
    for a_, b_ in itertools.permutations(runs, 2):
        A = imgs[a_][b, c - h:c + h, c - h:c + h]; B = imgs[b_][b, c - h:c + h, c - h:c + h]
        orph.append(float(A[B < 0.1 * A].sum() / A.sum()))
    out["U"][name] = row; out["orphan"][name] = float(np.median(orph))
    print(f"  {name}: " + "  ".join(f"s={s:3d}: {v:.3f}" for s, v in zip(scales, row)) + f"   flux with no counterpart in the other seed (s=0): {100 * np.median(orph):.1f}%")
json.dump(out, open(S + "/../summary/diag_seeds.json", "w"), indent=1)
