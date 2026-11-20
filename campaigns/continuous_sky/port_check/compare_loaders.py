"""Real-file check of piece 2: torchsky's load_exposures + conversion against the prototype's own loader, exposure by exposure.

    PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src python compare_loaders.py

The prototype side (fieldop_min: astropy readers, data.load_exposure_file) runs only here, outside both repositories.
"""
import glob
import math
import sys
import time

import numpy as np
import torch

PROTO = "/home/claude/work/fieldop_min"
sys.path.insert(0, PROTO)
torch.set_num_threads(2)

from fieldop import data as pdata, instruments as pins  # noqa: E402

from torchsky.instrument import instrument_description  # noqa: E402
from torchsky.map.exposure_fit import load_exposures  # noqa: E402
from torchsky.map.exposure_fit._bandpass import box_filter  # noqa: E402
from torchsky.map.exposure_fit._observations import observation_to_record  # noqa: E402

# SMACS box of the prototype's protocol (splits_SMACS.json): centre and size; data region = size/2 + data_margin (1.0)
BOX = dict(ra=110.8268, dec=-73.4544, size_arcsec=20.48)
HALF = BOX["size_arcsec"] / 2 + 1.0

CASES = [
    ("JWST NIRCam", "NIRCAM", sorted(glob.glob("/home/claude/work/dl_full/*_cal.fits")), (BOX["ra"], BOX["dec"])),
    ("HST WFC3/IR", "WFC3_IR", sorted(glob.glob("/home/claude/work/dl_hst/*_flt.fits"))[:4], None),
    ("HST ACS/WFC", "ACS_WFC", sorted(glob.glob("/home/claude/work/dl_hst/*_flc.fits")), None),
]


def centre_of(path, key):
    """Box centre from the file's own a priori header: the sky position of the middle of the first chip (used when the SMACS box is not covered)."""
    raw = pins.INSTRUMENTS[key].read(path)[0]
    ny, nx = raw["sci"].shape
    w = raw["w"]
    r = w.pixel_to_world_differentiable(torch.tensor([[nx / 2.0, ny / 2.0]], dtype=torch.float64), origin=0).numpy()[0]
    return float(r[0]), float(r[1])


def compare(label, key, files, centre):
    desc = instrument_description(key)
    for path in files:
        c = centre or centre_of(path, key)
        box = dict(ra=c[0], dec=c[1])
        t0 = time.perf_counter()
        ref, raws = [], []
        for raw in pins.INSTRUMENTS[key].read(path):
            r = pdata.load_exposure_file(path, box, HALF, raw=raw)
            if r is not None:
                r.update({k: raw[k] for k in ("chip", "name_suffix", "to_sky", "filter") if k in raw})
                ref.append(r)
                raws.append(raw)
        t1 = time.perf_counter()
        obs = load_exposures([path], desc, (c[0], c[1], HALF))
        t2 = time.perf_counter()
        name = path.split("/")[-1]
        if len(ref) != len(obs):
            print(f"{label:12s} {name}: chips with data: prototype {len(ref)}, torchsky {len(obs)}   <-- DIFFERENT")
            continue
        if not ref:
            print(f"{label:12s} {name}: box not covered by either loader (centre {c})")
            continue
        for r, o in zip(ref, obs):
            rec = observation_to_record(o, filt=0, inst=0, chan=0, det=0, bandpass=box_filter("x", key, 1.0, 2.0, 3))
            same_shape = tuple(r["d"].shape) == tuple(rec.d.shape) and tuple(r["cut0"]) == tuple(rec.cut0) and tuple(r["det_shape"]) == tuple(rec.det_shape)
            if not same_shape:
                print(f"{label:12s} {name}: cut differs: prototype {r['d'].shape} at {r['cut0']}, torchsky {tuple(rec.d.shape)} at {tuple(rec.cut0)}")
                continue
            v0, v1 = r["valid"], rec.valid.numpy()
            both = v0 & v1
            d0, d1 = r["d"].astype(np.float64), rec.d.numpy().astype(np.float64)
            scale = max(np.abs(d0[both]).max(), 1e-30)
            val_abs = float(np.abs(d0[both] - d1[both]).max())
            val_rel_f32 = float(np.max(np.abs(d0[both] - d1[both]) / (np.abs(d0[both]) * 1.2e-7 + 1e-37)))   # in units of float32 round-off of the value
            s0, s1 = r["sig"].astype(np.float64), rec.sig.numpy().astype(np.float64)
            sig_rel = float(np.max(np.abs(s0 - s1) / np.maximum(np.abs(s0), 1e-37)))
            nd = int((v0 != v1).sum())
            only0, only1 = int((v0 & ~v1).sum()), int((~v0 & v1).sum())
            dX = float(np.abs(r["X"].astype(np.float64) - rec.X.numpy())[both].max()) if both.any() else float("nan")
            dY = float(np.abs(r["Y"].astype(np.float64) - rec.Y.numpy())[both].max()) if both.any() else float("nan")
            print(f"{label:12s} {name} chip {r['chip']}: cut {r['d'].shape} at {r['cut0']}, valid {int(v0.sum())}/{int(v1.sum())}")
            print(f"    factor to MJy/sr: prototype {r['to_sky']!r} torchsky {o.header['TSTOSKY']!r}  (relative {abs(r['to_sky'] - o.header['TSTOSKY']) / max(abs(r['to_sky']), 1e-300):.1e})")
            print(f"    values on commonly valid pixels: max |diff| {val_abs:.3e} (max value {scale:.3e}); worst {val_rel_f32:.2f} float32 round-offs of the value")
            print(f"    sigma (all pixels): max relative diff {sig_rel:.3e}")
            print(f"    mask: {nd} pixels differ ({only0} valid only in the prototype, {only1} valid only in torchsky)")
            if nd:
                print(f"       differing pixel rows/cols (first 5): {[tuple(int(a) for a in p) for p in np.argwhere(v0 != v1)[:5]]}")
            # float64 positions (before the float32 storage that both sides use), whole cut
            from torchsky.map.exposure_fit._observations import tangent_positions as ts_tangent
            from torchsky.wcs import WCS as TWCS
            size = int(math.ceil(2 * HALF / 0.01)) + 64
            rr, cc = np.mgrid[0:rec.d.shape[0], 0:rec.d.shape[1]]
            raw_r = raws[ref.index(r)]
            X64, Y64 = pdata.tangent_positions(raw_r["w"], TWCS(header=pdata.target_header(c[0], c[1], size, 0.01)),
                                               (cc + r["cut0"][1]).astype(np.float64), (rr + r["cut0"][0]).astype(np.float64), 0.01, size)
            mX, mY = ts_tangent(o.header, tuple(rec.d.shape), c[0], c[1])
            d64 = max(float(np.abs(X64 - mX).max()), float(np.abs(Y64 - mY).max()))
            print(f"    positions in float64 (whole cut): max |difference| {d64:.3e} arcsec")
            if nd:
                dist = np.maximum(np.abs(r["X"].astype(np.float64)), np.abs(r["Y"].astype(np.float64)))
                print(f"       differing pixels: distance of max(|X|,|Y|) from the box edge {HALF}: "
                      f"{sorted(float(x) for x in (dist[v0 != v1] - HALF))[:3]} ... (arcsec; all within the position difference above if tiny)")
            print(f"    positions: max |dX| {dX:.3e} arcsec, max |dY| {dY:.3e} arcsec   (load: prototype {t1 - t0:.1f} s, torchsky {t2 - t1:.1f} s)")


for label, key, files, centre in CASES:
    if len(sys.argv) > 1 and sys.argv[1] not in label:
        continue
    print(f"=== {label}: {len(files)} files, box half side {HALF} arcsec")
    compare(label, key, files, centre)
