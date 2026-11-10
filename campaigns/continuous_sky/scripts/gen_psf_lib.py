"""Physics PSF libraries for experiment E2 (psf-geometry-audit.md): warm starts only, never constraints.

NIRCam: STPSF monochromatic PSFs at the model's bandpass quadrature nodes, per detector (centre of NRCB1-4 for SW bands, NRCB5 for
LW), measured OPD closest to the observation date, oversample 4, extension OVERDIST (optics + geometric distortion + charge
diffusion; no IPC, which STPSF applies only at detector sampling). The effective PSF (ePSF) = OVERDIST convolved with the detector
pixel box (4 x 4 oversampled samples), normalised to unit sum on the oversampled grid. APPROXIMATIONS: one field position per
detector (its centre); IPC and brighter-fatter left to the learned residual; STPSF array orientation checked against the data
(see orientation_check) rather than assumed.

WFC3/IR: Anderson's empirical STDPSF ePSFs (flt frame, 4x supersampled, 3x3 grid over the detector, include pixel response,
IPC and charge diffusion as observed) for F105W, F125W, F140W, F160W. APPROXIMATION: one ePSF per filter, used for every
quadrature node of the band (no within-band chromatic change).

Output: $OUT/nircam_epsf.npz (key "<DET>/<BAND>" -> float32 (nodes, N, N)), $OUT/wfc3ir/*.fits, $OUT/wfc3ir_epsf.npz
(key "<BAND>" -> (9, N, N) the 3x3 grid, centre = index 4), $OUT/meta.json."""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
from astropy.io import fits

OUT = sys.argv[1]
CAL = sys.argv[2]  # directory of the NIRCam _cal files (date and orientation check)
os.makedirs(OUT, exist_ok=True)
NODES = {"F090W": [0.8122, 0.8475, 0.8822, 0.9175, 0.9525, 0.9884], "F150W": [1.3565, 1.4128, 1.4682, 1.5248, 1.5824, 1.6403],
         "F200W": [1.7897, 1.8651, 1.9418, 2.0205, 2.1008, 2.1849], "F277W": [2.4704, 2.5793, 2.6889, 2.8031, 2.9227, 3.0596],
         "F356W": [3.2065, 3.3499, 3.4929, 3.6308, 3.7679, 3.9094], "F444W": [3.9578, 4.1177, 4.2832, 4.4593, 4.6524, 4.8787]}
SW, LW = ["F090W", "F150W", "F200W"], ["F277W", "F356W", "F444W"]
OS = 4
FOV = 4.0  # arcsec
meta = {"oversample": OS, "fov_arcsec": FOV, "nodes_um": NODES, "nircam": {}, "wfc3ir": {}, "approximations": __doc__}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def box(k, n=OS):
    """Convolve an oversampled PSF with the n x n detector-pixel box (same size, centred, even n: half-sample shift avoided by
    averaging the four n-shifts so the result stays centred on the same sample)."""
    from scipy.ndimage import uniform_filter
    out = uniform_filter(k.astype(np.float64), size=n, mode="constant")
    out2 = uniform_filter(k.astype(np.float64), size=n, mode="constant", origin=-1)
    out = 0.5 * (out + out2)  # even box: average of the two centrings, symmetric to first order
    return out / out.sum()


# ------------------------------------------------------------------ WFC3/IR STDPSF
cands = ["https://www.stsci.edu/~jayander/HST1PASS/LIB/PSFs/STDPSFs/WFC3IR/STDPSF_WFC3IR_{f}.fits",
         "https://www.stsci.edu/~jayander/HST1PASS/LIB/PSFs/STDPSFs/WFC3IR/PSFSTD_WFC3IR_{f}.fits",
         "https://www.stsci.edu/~jayander/STDPSFs/WFC3IR/PSFSTD_WFC3IR_{f}.fits",
         "https://www.stsci.edu/~jayander/STDPSFs/WFC3IR/STDPSF_WFC3IR_{f}.fits"]
os.makedirs(f"{OUT}/wfc3ir", exist_ok=True)
wl = {}
for f in ["F105W", "F125W", "F140W", "F160W"]:
    got = None
    for c in cands:
        u = c.format(f=f)
        dst = f"{OUT}/wfc3ir/{os.path.basename(u)}"
        try:
            urllib.request.urlretrieve(u, dst)
            d = fits.getdata(dst)
            got = (u, dst, d)
            break
        except Exception as exc:  # noqa: BLE001
            log("WFC3IR", f, "failed", u, repr(exc)[:120])
    if got is None:
        meta["wfc3ir"][f] = "NOT FOUND"
        continue
    u, dst, d = got
    d = np.asarray(d, np.float32)
    log("WFC3IR", f, u, d.shape)
    wl[f"WFC3IR_{f}"] = d / d.sum(axis=(-2, -1), keepdims=True)
    meta["wfc3ir"][f] = {"url": u, "shape": list(d.shape), "header": {k: str(v) for k, v in fits.getheader(dst).items() if k not in ("COMMENT", "HISTORY")}}
np.savez_compressed(f"{OUT}/wfc3ir_epsf.npz", **wl)

# ------------------------------------------------------------------ NIRCam STPSF
import stpsf  # noqa: E402

cals = sorted(glob.glob(f"{CAL}/*_cal.fits"))
h0 = fits.getheader(cals[0])
date = f"{h0['DATE-OBS']}T{h0['TIME-OBS'][:8]}"
meta["nircam"]["date_obs"] = date
meta["nircam"]["stpsf_version"] = getattr(stpsf, "__version__", "?")
lib = {}
for dets, bands in ((["NRCB1", "NRCB2", "NRCB3", "NRCB4"], SW), (["NRCB5"], LW)):
    for det in dets:
        for b in bands:
            nrc = stpsf.NIRCam()
            nrc.filter = b
            nrc.detector = det
            if "opd" not in meta["nircam"]:
                nrc.load_wss_opd_by_date(date, plot=False)  # closest measured OPD in time (STPSF default)
                meta["nircam"]["opd"] = str(nrc.pupilopd if isinstance(nrc.pupilopd, str) else getattr(nrc.pupilopd, "filename", "hdulist"))
                opd = nrc.pupilopd
            else:
                nrc.pupilopd = opd
            stack = []
            for lam in NODES[b]:
                t = time.time()
                hdul = nrc.calc_psf(monochromatic=lam * 1e-6, oversample=OS, fov_arcsec=FOV)
                k = hdul["OVERDIST"].data
                stack.append(box(k).astype(np.float32))
                log(det, b, lam, k.shape, f"{time.time() - t:.1f}s")
            lib[f"{det}/{b}"] = np.stack(stack)
            meta["nircam"].setdefault("pixelscale", {})[det] = float(nrc.pixelscale)
np.savez_compressed(f"{OUT}/nircam_epsf.npz", **lib)

# ------------------------------------------------------------------ orientation check against the data (diagnostic only)
# For a few cal images: brightest unsaturated compact peak; compare its 15x15 stamp with the library ePSF (detector-sampled at the
# best of 4x4 sub-pixel phases) under the 8 dihedral transforms; least-squares flux + background. Reported, not used to fit.
def dihedral(k):
    out = []
    for fl in (False, True):
        kk = k[:, ::-1] if fl else k
        for r in range(4):
            out.append((f"flip{int(fl)}_rot{90 * r}", np.rot90(kk, r)))
    return out


chk = {}
for det, b in (("NRCB1", "F150W"), ("NRCB2", "F200W"), ("NRCB5", "F444W")):
    files = [f for f in cals if det.lower().replace("nrcb5", "nrcblong") in f.lower() and fits.getheader(f).get("FILTER") == b]
    if not files:
        continue
    with fits.open(files[0]) as h:
        d, dq = h["SCI"].data.astype(float), h["DQ"].data
    d = np.where(np.isfinite(d) & (dq & 1 == 0), d, np.nan)
    from scipy.ndimage import maximum_filter
    dd = np.nan_to_num(d - np.nanmedian(d))
    sat = (dq & 2) > 0
    pk = (dd == maximum_filter(dd, 7)) & ~maximum_filter(sat, 9)
    ys, xs = np.nonzero(pk)
    ok = (ys > 20) & (xs > 20) & (ys < d.shape[0] - 20) & (xs < d.shape[1] - 20)
    ys, xs = ys[ok], xs[ok]
    o = np.argsort(-dd[ys, xs])[:5]
    k = lib[f"{det}/{b}"][len(NODES[b]) // 2]
    c = k.shape[0] // 2
    res = {}
    for name, kt in dihedral(k):
        tot = 0.0
        for i in o:
            st = dd[ys[i] - 7:ys[i] + 8, xs[i] - 7:xs[i] + 8]
            best = np.inf
            for py in range(OS):
                for px in range(OS):
                    m = kt[c - 7 * OS + py - OS // 2:c + 8 * OS + py - OS // 2:OS, c - 7 * OS + px - OS // 2:c + 8 * OS + px - OS // 2:OS]
                    if m.shape != st.shape:
                        continue
                    A = np.stack([m.ravel(), np.ones(m.size)], 1)
                    coef, *_ = np.linalg.lstsq(A, st.ravel(), rcond=None)
                    r = st.ravel() - A @ coef
                    best = min(best, float((r ** 2).sum() / (st.ravel() ** 2).sum()))
            tot += best
        res[name] = tot / len(o)
    chk[f"{det}/{b}"] = dict(sorted(res.items(), key=lambda kv: kv[1]))
    log("orientation", det, b, list(chk[f"{det}/{b}"].items())[:3])
meta["orientation_check"] = chk
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
log("done")
