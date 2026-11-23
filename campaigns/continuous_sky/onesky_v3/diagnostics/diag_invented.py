"""Metrics of structure that one fit has and another does not, for two fits A and B of the same sky.

    python diag_invented.py SCRATCH label=runA,runB[,register] ...

Images: each fit's sky on its grid (0.04 arcsec), per filter common to both, central 16 arcsec. With `register` the frame of B is first
shifted onto A's (the shift that maximises their correlation after an 80 mas blur, searched over +-0.4 arcsec in steps of 8 mas).
Noise scale of the comparison: sigma_d = NMAD(A - B) / sqrt(2) after a 40 mas blur.

Metrics (named in the output):
  unreproduced fraction U(s) = sum((A_s - B_s)^2) / sum(A_s^2 + B_s^2), images blurred by a Gaussian of standard deviation s
  unmatched-peak fraction F(k) = share of the local maxima of A above k sigma_d (40 mas blur, maxima over +-80 mas) for which B has less than
      30% of the peak value anywhere within 120 mas; the same from B to A; the mean of the two is reported
  unmatched flux fraction = sum of max(A - B, 0) + max(B - A, 0) over pixels where the larger is above 5 sigma_d and the smaller is below a
      third of it, divided by sum(A + B) over pixels where either is above 5 sigma_d
"""
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchsky.map.exposure_fit import SkyFit
S = sys.argv[1]
torch.set_num_threads(2)
_cache = {}
def sky(run):
    if run not in _cache:
        fit = SkyFit.load(f"{S}/canfar/fits/{run}", device="cpu"); op = fit.operator
        with torch.no_grad():
            A = op.sky.coeffs(); img = torch.einsum("bk,kyx->byx", op.Phi_band.to(A.dtype), A).float()
        _cache[run] = (dict(zip(fit.bands, img)), float(op.sky.pitch))
    return _cache[run]
def blur(x, s_px):
    if s_px <= 0: return x
    n = int(4 * s_px) + 1; g = torch.exp(-0.5 * (torch.arange(-n, n + 1) / s_px) ** 2); g = (g / g.sum()).to(x)
    x = F.conv2d(x[None, None], g[None, None, :, None], padding=(n, 0)); return F.conv2d(x, g[None, None, None, :], padding=(0, n))[0, 0]
nmad = lambda x: float(1.4826 * (x - x.median()).abs().median())
def shift(img, dx_px, dy_px):
    n = img.shape[-1]; g = torch.linspace(-1, 1, n); yy, xx = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([xx + 2 * dx_px / (n - 1), yy + 2 * dy_px / (n - 1)], -1)[None]
    return F.grid_sample(img[None, None], grid, mode="bicubic", align_corners=True, padding_mode="zeros")[0, 0]
scales = [0, 20, 40, 80, 160]
out = {}
for tok in sys.argv[2:]:
    label, rest = tok.split("=", 1); parts = rest.split(","); ra, rb = parts[0], parts[1]; reg = len(parts) > 2
    (A_, pitch), (B_, _) = sky(ra), sky(rb)
    bands = [b for b in A_ if b in B_]; N = A_[bands[0]].shape[-1]; c = N // 2; h = int(8.0 / pitch); cut = (slice(c - h, c + h), slice(c - h, c + h)); mas = 1e3 * pitch
    sh = (0.0, 0.0)
    if reg:
        a0, b0 = blur(sum(A_[b] for b in bands), 80 / mas), blur(sum(B_[b] for b in bands), 80 / mas); best = -2
        for dy in np.arange(-10, 10.01, 0.2):
            for dx in np.arange(-10, 10.01, 0.2):
                if abs(dx * 5) % 1 > 1e-6 and False: continue
        # coarse (1 px) then fine (0.2 px) search
        def corr(dx, dy):
            x, y = a0[cut].flatten(), shift(b0, dx, dy)[cut].flatten(); x, y = x - x.mean(), y - y.mean(); return float((x * y).sum() / (x.norm() * y.norm()))
        cand = [(corr(dx, dy), dx, dy) for dy in range(-10, 11) for dx in range(-10, 11)]; _, dx0, dy0 = max(cand)
        cand = [(corr(dx0 + ddx, dy0 + ddy), dx0 + ddx, dy0 + ddy) for ddy in np.arange(-1, 1.01, 0.2) for ddx in np.arange(-1, 1.01, 0.2)]; best, dxb, dyb = max(cand); sh = (dxb, dyb)
    rec = {"bands": bands, "shift_mas": [sh[0] * mas, sh[1] * mas], "U": {}, "F": {}, "unmatched_flux": {}}
    print(f"\n== {label}: A = {ra}, B = {rb}; filters {bands}" + (f"; B shifted by ({sh[0] * mas:+.0f}, {sh[1] * mas:+.0f}) mas, correlation {best:.3f}" if reg else ""))
    for b in bands:
        A = A_[b]; B = shift(B_[b], *sh) if reg else B_[b]
        U = []
        for s in scales:
            a, bb = blur(A, s / mas)[cut], blur(B, s / mas)[cut]; U.append(float(((a - bb) ** 2).sum() / (a ** 2 + bb ** 2).sum()))
        a, bb = blur(A, 40 / mas), blur(B, 40 / mas); sd = nmad((a - bb)[cut]) / 2 ** 0.5
        def unmatched(x, y, k):
            loc = (x == F.max_pool2d(x[None, None], 5, 1, 2)[0, 0]) & (x > k * sd); near = F.max_pool2d(y[None, None], 7, 1, 3)[0, 0]
            m = torch.zeros_like(loc); m[cut] = True; loc &= m
            n = int(loc.sum()); return (int((loc & (near < 0.3 * x)).sum()), n)
        Fk = {}
        for k in (5, 10, 30):
            u1, n1 = unmatched(a, bb, k); u2, n2 = unmatched(bb, a, k); Fk[k] = ((u1 + u2) / max(n1 + n2, 1), n1 + n2)
        hi, lo = torch.maximum(a, bb)[cut], torch.minimum(a, bb)[cut]
        um = float((hi - lo)[(hi > 5 * sd) & (lo < hi / 3)].sum() / (hi + lo)[hi > 5 * sd].sum())
        rec["U"][b] = U; rec["F"][b] = {str(k): v[0] for k, v in Fk.items()}; rec["unmatched_flux"][b] = um
        print(f"  {b}: unreproduced fraction U(s), s = {scales} mas: " + " ".join(f"{v:.3f}" for v in U) + "  | unmatched-peak fraction F(k), k = 5, 10, 30: "
              + " ".join(f"{100 * Fk[k][0]:.1f}% of {Fk[k][1]}" for k in (5, 10, 30)) + f"  | unmatched flux fraction {100 * um:.2f}%")
    out[label] = rec
json.dump(out, open(S + "/summary/diag_invented.json", "w"), indent=1)
