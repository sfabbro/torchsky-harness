"""Width of compact sources in a fitted sky against the official mosaic: python sharpness.py <state.pt> <item max width or 0 for a grid sky>
Sources: the 12 most concentrated local maxima of the F200W mosaic inside the central 18 arcsec (evaluation only). Width = full width at
half maximum from the azimuthally averaged profile about the local peak (linear interpolation of the half-maximum crossing), after
subtracting the median in an annulus of 0.6 to 0.9 arcsec. Approximations: nearest-pixel peak, no deblending, one common source list."""
import sys
import numpy as np
import torch
from astropy.io import fits
from scipy.ndimage import maximum_filter, uniform_filter
sys.argv = sys.argv[:1] + sys.argv[1:]
from flux_ratio import sky_coeffs, FILTS
torch.set_num_threads(1)
def fwhm(img, pitch, cx, cy):
    n = img.shape[0]; g = (np.arange(n) - (n - 1) / 2) * pitch
    j0, i0 = np.argmin(abs(g - cx)), np.argmin(abs(g - cy)); h = int(round(1.0 / pitch))
    sub = img[i0 - h:i0 + h + 1, j0 - h:j0 + h + 1]
    if sub.shape != (2 * h + 1, 2 * h + 1) or not np.isfinite(sub).all(): return np.nan
    w = int(round(0.12 / pitch)); c = sub[h - w:h + w + 1, h - w:h + w + 1]; pi, pj = np.unravel_index(c.argmax(), c.shape); pi += h - w; pj += h - w
    yy, xx = np.mgrid[0:2 * h + 1, 0:2 * h + 1]; r = np.hypot(yy - pi, xx - pj) * pitch
    bg = np.median(sub[(r > 0.6) & (r < 0.9)]); pk = sub[pi, pj] - bg
    if pk <= 0: return np.nan
    edges = np.arange(0, 0.5, pitch / 2); prof = []; rad = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.any(): prof.append(sub[m].mean() - bg); rad.append(r[m].mean())
    prof, rad = np.array(prof), np.array(rad); k = np.nonzero(prof < pk / 2)[0]
    if not len(k) or k[0] == 0: return np.nan
    k = k[0]; return 2 * (rad[k - 1] + (prof[k - 1] - pk / 2) / (prof[k - 1] - prof[k]) * (rad[k] - rad[k - 1]))
if __name__ == "__main__":
    s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    if len(sys.argv) > 2 and float(sys.argv[2]) > 0: sys.argv = [sys.argv[0], sys.argv[1], "x", sys.argv[2]]
    A, pitch = sky_coeffs(s)
    mos = fits.getdata("/home/claude/work/kernel_check/dl/cut_F200W.fits").astype(float); pm = fits.getheader("/home/claude/work/kernel_check/dl/cut_F200W.fits")["CDELT1"] * 3600
    n = mos.shape[0]; g = (np.arange(n) - (n - 1) / 2) * pm; X, Y = np.meshgrid(g, g)
    z = np.nan_to_num(mos); loc = (z == maximum_filter(z, 15)) & (abs(X) < 9) & (abs(Y) < 9)
    conc = z / np.maximum(uniform_filter(z, 11), 1e-9); cand = np.argwhere(loc & (z > np.percentile(z, 99)))
    cand = sorted(cand, key=lambda ij: -conc[ij[0], ij[1]])[:12]
    src = [(X[i, j], Y[i, j]) for i, j in cand]
    print("filter  median width in the fitted sky  median width in the official mosaic  (arcsec; sources measured)")
    for f, name in enumerate(FILTS):
        sky = torch.einsum("k,khw->hw", s["Phi_band"][f].to(A.dtype), A).numpy()
        m2 = fits.getdata(f"/home/claude/work/kernel_check/dl/cut_{name}.fits").astype(float); p2 = fits.getheader(f"/home/claude/work/kernel_check/dl/cut_{name}.fits")["CDELT1"] * 3600
        a = np.array([fwhm(sky, pitch, x, y) for x, y in src]); b = np.array([fwhm(m2, p2, x, y) for x, y in src]); ok = np.isfinite(a) & np.isfinite(b)
        print(f"{name}   {np.median(a[ok]):.3f}   {np.median(b[ok]):.3f}   ratio {np.median(a[ok] / b[ok]):.2f}   ({ok.sum()} sources; smallest sky width {np.nanmin(a):.3f}, smallest mosaic width {np.nanmin(b):.3f})")
