"""Sky-position difference between the telescope-frame count-rate WCS (--rate-wcs telescope) and the calibrated file's own header WCS (--rate-wcs cal)
over the detector, for every sample exposure.   python tests/check_rate_wcs_telescope.py --rate-dir DIR --cal-dir DIR [--cal-dir DIR2] --out FILE
Lattice: 65 x 65 zero-based pixels from 0 to 2047.  Separation = great-circle angle in milliarcseconds.  'after mean offset' removes the mean
tangent-plane offset (diagnostic only; not used by the package)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


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
    rows = []
    for rp in sorted(glob.glob(os.path.join(a.rate_dir, "*_rate.fits"))):
        base = os.path.basename(rp).replace("_rate.fits", "")
        cal = [os.path.join(d, base + "_cal.fits") for d in a.cal_dir if os.path.exists(os.path.join(d, base + "_cal.fits"))]
        if not cal:
            continue
        sh = dict(torchfits.read_header(cal[0], "SCI"))
        hd = {k: v for k, v in sh.items() if k and k not in ("COMMENT", "HISTORY", "")}
        wc = WCS(header=hd).pixel_to_world_differentiable(pc, origin=0)
        tw = TelescopeRateWCS(rp)
        wt = tw.pixel_to_world_differentiable(pc, origin=0)
        ra1, d1, ra2, d2 = [torch.deg2rad(x) for x in (wc[:, 0], wc[:, 1], wt[:, 0], wt[:, 1])]
        # tangent-plane offsets (east, north) of the telescope solution relative to the calibrated one, small-angle but exact enough below 1 arcsec
        de = (ra2 - ra1 + torch.pi) % (2 * torch.pi) - torch.pi
        east = de * torch.cos(d1)
        north = d2 - d1
        sep = torch.sqrt(east ** 2 + north ** 2) * (180 / torch.pi) * 3600e3
        me, mn = east.mean(), north.mean()
        sep0 = torch.sqrt((east - me) ** 2 + (north - mn) ** 2) * (180 / torch.pi) * 3600e3
        det = str(dict(torchfits.read_header(rp, 0)).get("DETECTOR"))
        rows.append({"exposure": base, "detector": det, "filter": str(dict(torchfits.read_header(rp, 0)).get("FILTER")), "aperture": tw.aperture,
                     "max_mas": float(sep.max()), "rms_mas": float(sep.pow(2).mean().sqrt()), "median_mas": float(sep.median()),
                     "mean_offset_east_mas": float(me) * 180 / np.pi * 3600e3, "mean_offset_north_mas": float(mn) * 180 / np.pi * 3600e3,
                     "max_after_mean_offset_mas": float(sep0.max()), "rms_after_mean_offset_mas": float(sep0.pow(2).mean().sqrt()),
                     "va_scale": tw.va_scale, "pixar_a2_telescope": tw.pixar_a2, "pixar_a2_cal": float(sh.get("PIXAR_A2", float("nan")))})
        print(json.dumps(rows[-1]))
    by = {}
    for r in rows:
        by.setdefault(r["detector"], []).append(r)
    summ = {d: {"n": len(v), "max_mas": max(x["max_mas"] for x in v),
                "rms_mas": float(np.sqrt(np.mean([x["rms_mas"] ** 2 for x in v]))),
                "rms_below_5_mas": all(x["rms_mas"] < 5.0 for x in v)} for d, v in by.items()}
    print(json.dumps(summ, indent=1))
    if a.out:
        json.dump({"rows": rows, "per_detector": summ}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
