"""Acceptance A of section E: one shared detector-pixel shift per (detector, filter) pair, fitted by least squares so that the telescope-frame
count-rate solution (--rate-wcs telescope) matches the pipeline's calibrated solution; files only.
    python tests/check_filter_shift_files.py --rate-dir DIR --cal-dir DIR [--cal-dir DIR2] --out FILE
Model of the shift: position of pixel p = W_tel(p + s), s in detector pixels (x = column, y = row, zero based), the same convention as --filter-shift.
Lattice: 65 x 65 zero-based pixels from 0 to 2047.  Differences are great-circle-free tangent-plane offsets (east, north) about the calibrated position,
in mas (exact to well below 1 microarcsecond at these sizes).  Gauss-Newton on s with the Jacobian of W_tel by central differences of 0.5 pixel."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAS = 180 / np.pi * 3600e3


def tangent(w_a, w_ref):
    """(east, north) offsets of world positions w_a about w_ref, radians; both (n, 2) degrees."""
    ra1, d1, ra2, d2 = [torch.deg2rad(x) for x in (w_ref[:, 0], w_ref[:, 1], w_a[:, 0], w_a[:, 1])]
    de = (ra2 - ra1 + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.stack([de * torch.cos(d1), d2 - d1], -1)


def fit_shift(tw, pc, w_cal, iters=3):
    """Least-squares s (2,) with W_tel(pc + s) ~ w_cal; returns s, the residual (n, 2) in mas."""
    s = torch.zeros(2, dtype=torch.float64)
    h = 0.5
    for _ in range(iters):
        w0 = tw.pixel_to_world_differentiable(pc + s, origin=0)
        r = tangent(w0, w_cal)
        cols = []
        for k in range(2):
            d = torch.zeros(2, dtype=torch.float64)
            d[k] = h
            cols.append((tangent(tw.pixel_to_world_differentiable(pc + s + d, origin=0), w_cal) - tangent(tw.pixel_to_world_differentiable(pc + s - d, origin=0), w_cal)) / (2 * h))
        A = torch.stack(cols, -1)                                  # (n, 2 sky, 2 pixel)
        s = s - torch.linalg.lstsq(A.reshape(-1, 2), r.reshape(-1, 1)).solution[:, 0]
    r = tangent(tw.pixel_to_world_differentiable(pc + s, origin=0), w_cal)
    return s, r * MAS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-dir", required=True)
    ap.add_argument("--cal-dir", action="append", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torchfits
    from fieldop.tsky_backends import TelescopeRateWCS
    from torchsky.wcs import WCS
    g = torch.linspace(0.0, 2047.0, 65, dtype=torch.float64)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pc = torch.stack([gx.ravel(), gy.ravel()], -1)
    ex = {}
    for rp in sorted(glob.glob(os.path.join(a.rate_dir, "*_rate.fits"))):
        base = os.path.basename(rp).replace("_rate.fits", "")
        cal = [os.path.join(d, base + "_cal.fits") for d in a.cal_dir if os.path.exists(os.path.join(d, base + "_cal.fits"))]
        if not cal:
            continue
        sh = dict(torchfits.read_header(cal[0], "SCI"))
        hd = {k: v for k, v in sh.items() if k and k not in ("COMMENT", "HISTORY", "")}
        h0 = dict(torchfits.read_header(rp, 0))
        key = f"{str(h0.get('DETECTOR')).strip()}/{str(h0.get('FILTER')).strip()}"
        ex.setdefault(key, []).append((base, TelescopeRateWCS(rp), WCS(header=hd).pixel_to_world_differentiable(pc, origin=0)))
    rows, _res = [], {}
    for key, lst in sorted(ex.items()):
        # one shared shift: stack the exposures' normal equations by fitting the concatenated lattices
        s = torch.zeros(2, dtype=torch.float64)
        h = 0.5
        for _ in range(3):
            Rs, As = [], []
            for _, tw, wc in lst:
                Rs.append(tangent(tw.pixel_to_world_differentiable(pc + s, origin=0), wc).reshape(-1, 1))
                cols = []
                for k in range(2):
                    d = torch.zeros(2, dtype=torch.float64)
                    d[k] = h
                    cols.append(((tangent(tw.pixel_to_world_differentiable(pc + s + d, origin=0), wc) - tangent(tw.pixel_to_world_differentiable(pc + s - d, origin=0), wc)) / (2 * h)).reshape(-1))
                As.append(torch.stack(cols, -1))
            s = s - torch.linalg.lstsq(torch.cat(As), torch.cat(Rs)).solution[:, 0]
        per = []
        for base, tw, wc in lst:
            r0 = tangent(tw.pixel_to_world_differentiable(pc, origin=0), wc) * MAS
            r = tangent(tw.pixel_to_world_differentiable(pc + s, origin=0), wc) * MAS
            own, rown = fit_shift(tw, pc, wc)
            sep0, sep, sepo = r0.norm(dim=-1), r.norm(dim=-1), rown.norm(dim=-1)
            per.append({"exposure": base, "rms_before_mas": float(sep0.pow(2).mean().sqrt()), "max_before_mas": float(sep0.max()),
                        "rms_shared_mas": float(sep.pow(2).mean().sqrt()), "max_shared_mas": float(sep.max()),
                        "own_shift_px": own.tolist(), "rms_own_mas": float(sepo.pow(2).mean().sqrt()), "max_own_mas": float(sepo.max())})
        allsep = torch.cat([(tangent(tw.pixel_to_world_differentiable(pc + s, origin=0), wc) * MAS).norm(dim=-1) for _, tw, wc in lst])
        own = np.array([p["own_shift_px"] for p in per])
        pair = {"pair": key, "n_exposures": len(lst), "shared_shift_px": s.tolist(), "rms_mas": float(allsep.pow(2).mean().sqrt()), "max_mas": float(allsep.max()),
                "own_shift_difference_px": (own.max(0) - own.min(0)).tolist() if len(lst) > 1 else None,
                "own_shift_difference_norm_px": float(np.linalg.norm(own[0] - own[1])) if len(lst) == 2 else None, "exposures": per}
        rows.append(pair)
        print(json.dumps({k: v for k, v in pair.items() if k != "exposures"}), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
