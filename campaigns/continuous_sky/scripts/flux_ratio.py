"""Flux scale of a fitted fieldop sky against the official mosaic (SMACS box): python flux_ratio.py <fieldop_state.pt> [label]
Fitted sky band image = sum_k Phi_band[f, k] x coefficients (MJy/sr on the sky nodes); mosaic cut-outs (MJy/sr) from
../kernel_check/dl/cut_<FILTER>.fits (official stage-3 mosaic of all exposures, same tangent frame). Measures, per filter:
  (a) the brightest extended object within the central 14 arcsec (found independently in each image after a 0.4 arcsec box smoothing):
      flux inside radius 1 arcsec and 0.3 arcsec, background = median in the annulus 2 to 3 arcsec;
  (b) total flux in the central 20 arcsec box above the 20th percentile level.
Approximations: one object for (a); annulus background in a crowded cluster field; no astrometric matching beyond the common frame;
item skies are rasterised by the package (needs the package on sys.path) - grid skies are read directly."""
import sys, json
import numpy as np, torch
from astropy.io import fits
from scipy.ndimage import uniform_filter
torch.set_num_threads(1)
FILTS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
def sky_coeffs(s, pkg=None):
    sk = s["skies"][0]
    if "A" in sk:
        return sk["A"][0] * sk["pre"], 0.04 if sk["A"].shape[-1] == 613 else 24.48 / (sk["A"].shape[-1] - 1)
    sys.path.insert(0, pkg or "/home/claude/work/fieldop_items")
    from fieldop.sky import ItemSky, RBFBasis
    K = sk["lc"].shape[-1]; N = sk["resid"].shape[-1] if "resid" in sk else 613; pitch = 24.48 / (N - 1)   # states after 17 Nov have no resid buffer: pitch 0.04 assumed
    it = ItemSky(torch.zeros(K, N, N), pitch, RBFBasis(K), sk["pos"].shape[1], max_width=float(sys.argv[3]) if len(sys.argv) > 3 else 0.32, raster="pyramid")
    with torch.no_grad():
        for k in ("pos", "lw", "ang", "lc"):
            getattr(it, k).data[:] = sk[k]
        if "S" in sk and it.S is not None:
            it.S.data[:] = sk["S"]
        it._mut += 1
        return it.coeffs().detach(), pitch
def ap(img, pitch, cx, cy, r, r1, r2):
    n = img.shape[0]; g = (np.arange(n) - (n - 1) / 2) * pitch
    X, Y = np.meshgrid(g, g); d = np.hypot(X - cx, Y - cy)
    bg = np.nanmedian(img[(d > r1) & (d < r2)])
    return np.nansum(img[d < r] - bg) * pitch ** 2
def measure(img, pitch):
    n = img.shape[0]; g = (np.arange(n) - (n - 1) / 2) * pitch; X, Y = np.meshgrid(g, g)
    sm = uniform_filter(np.nan_to_num(img), int(round(0.4 / pitch))); sm[(abs(X) > 7) | (abs(Y) > 7)] = -1
    iy, ix = np.unravel_index(sm.argmax(), sm.shape); cx, cy = X[iy, ix], Y[iy, ix]
    box = (abs(X) < 10) & (abs(Y) < 10); blank = np.nanpercentile(img[box], 20)
    return dict(f1=ap(img, pitch, cx, cy, 1.0, 2.0, 3.0), f03=ap(img, pitch, cx, cy, 0.3, 2.0, 3.0), tot=np.nansum((img - blank)[box]) * pitch ** 2, blank=blank)
if __name__ == "__main__":
    s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    A, pitch = sky_coeffs(s)
    out = {}
    PIV = dict(F090W=0.902, F150W=1.501, F200W=1.989, F277W=2.776, F356W=3.565, F444W=4.421)   # micrometres
    piv = s["piv"].flatten().tolist() if "piv" in s else None                       # 18 Nov: states with several instruments hold more filters, by pivot
    for f, name in enumerate(FILTS):
        if piv is not None and len(piv) != len(FILTS):
            f = min(range(len(piv)), key=lambda i: abs(piv[i] - PIV[name]))
            assert abs(piv[f] - PIV[name]) < 0.03 * PIV[name], (name, piv)
        sky = torch.einsum("k,khw->hw", s["Phi_band"][f].to(A.dtype), A).numpy()
        hd = fits.getheader(f"/home/claude/work/kernel_check/dl/cut_{name}.fits")
        mos = fits.getdata(f"/home/claude/work/kernel_check/dl/cut_{name}.fits").astype(float)
        a, b = measure(sky, pitch), measure(mos, hd["CDELT1"] * 3600)
        out[name] = {k: float(v) for k, v in dict(ratio_r1=a["f1"] / b["f1"], ratio_r03=a["f03"] / b["f03"], ratio_box_total=a["tot"] / b["tot"], sky_level=a["blank"], mosaic_level=b["blank"]).items()}
        print(f"{name}: sky/mosaic  r=1\": {out[name]['ratio_r1']:.2f}  r=0.3\": {out[name]['ratio_r03']:.2f}  box total: {out[name]['ratio_box_total']:.2f} | 20th-percentile level sky {a['blank']:.3f} mosaic {b['blank']:.3f} MJy/sr")
    json.dump(out, open((sys.argv[2] if len(sys.argv) > 2 else "flux_ratio") + ".json", "w"), indent=1)
