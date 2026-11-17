"""End-to-end check of the --rate-units sky convention on real exposures, without fitting (AUDIT_2026-11-17.md section F, change 2 (b)).

For each local _rate file and its matching _cal file:
    ratio(p) = rate(p) x PHOTMJSR / (a(p) / PIXAR_A2) / cal(p)
on the pixels with DQ == 0 in both files (in the _rate file only when the _cal copy has no DQ: the cal SCI is then NaN where the pipeline
flags DO_NOT_USE), finite values and |cal| > 0, where a(p) is the header pixel area [arcsec^2] from central differences of
the _cal SIP WCS (what the model's alpha is at zero learned geometry). If the convention is right, the median ratio is about 1; its pixel-to-pixel
scatter is what calwebb_image2 divides by besides PHOTMJSR (the flat field, with any area term the flat carries) -- the part the fit's learned
throughput and gain terms are meant to absorb. The same with the pipeline's own relative area map (AREA extension) in place of a / PIXAR_A2 is
reported for comparison, and with no area term at all.

    python tests/check_rate_photom.py --rate-dir /home/claude/work/fieldop_gain/dl/rate --cal-dir /home/claude/work/score_paired/tests/data/cal
"""
import argparse
import glob
import json
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom


def header_area(w, shape, step=32):
    """Header pixel area [arcsec^2] on a grid every `step` pixels (central differences of 1 pixel, local tangent plane), upsampled bilinearly."""
    ny, nx = shape
    ys, xs = np.arange(0, ny + step, step).clip(max=ny - 1), np.arange(0, nx + step, step).clip(max=nx - 1)
    X, Y = np.meshgrid(xs.astype(float), ys.astype(float))

    def sky(x, y):
        r, d = w.all_pix2world(x, y, 0)
        return np.radians(r), np.radians(d)
    r0, d0 = sky(X, Y)

    def offs(dx, dy):
        r, d = sky(X + dx, Y + dy)
        return np.degrees((r - r0) * np.cos(d0)) * 3600, np.degrees(d - d0) * 3600
    ax, ay = offs(0.5, 0)
    bx, by = offs(-0.5, 0)
    cx, cy = offs(0, 0.5)
    ex, ey = offs(0, -0.5)
    A = np.abs((ax - bx) * (cy - ey) - (ay - by) * (cx - ex))
    up = zoom(A, (ny / A.shape[0], nx / A.shape[1]), order=1)
    return up[:ny, :nx]


def robust(v):
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-dir", required=True)
    ap.add_argument("--cal-dir", required=True, nargs="+", help="one or more folders; the largest matching file is used")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for rp in sorted(glob.glob(os.path.join(a.rate_dir, "*_rate.fits"))):
        cands = [os.path.join(d, os.path.basename(rp).replace("_rate.fits", "_cal.fits")) for d in a.cal_dir]
        cands = [c for c in cands if os.path.exists(c)]
        cp = max(cands, key=os.path.getsize) if cands else ""
        if not cp:
            print(f"{os.path.basename(rp)}: no matching _cal file, skipped")
            continue
        with fits.open(rp, memmap=False) as hr, fits.open(cp, memmap=False) as hc:
            rate, dqr = hr["SCI"].data.astype(np.float64), hr["DQ"].data
            cal, sh = hc["SCI"].data.astype(np.float64), hc["SCI"].header
            names = [x.name for x in hc]
            dqc = hc["DQ"].data if "DQ" in names else None          # the test copies of the _cal files are cut after SCI (no DQ, no AREA)
            area_ext = hc["AREA"].data.astype(np.float64) if "AREA" in names else None
            det, filt = str(hc[0].header["DETECTOR"]).upper(), str(hc[0].header["FILTER"]).upper()
        C, a_nom = float(sh["PHOTMJSR"]), float(sh["PIXAR_A2"])
        rel = header_area(WCS(sh), cal.shape) / a_nom
        ok = (dqr == 0) & ((dqc == 0) if dqc is not None else True) & np.isfinite(rate) & np.isfinite(cal) & (np.abs(cal) > 0) & np.isfinite(rel)
        r_hdr = rate[ok] * C / rel[ok] / cal[ok]
        r_none = rate[ok] * C / cal[ok]
        row = {"file": os.path.basename(rp), "cal_file": cp, "cal_dq_used": dqc is not None, "detector": det, "filter": filt, "PHOTMJSR": C, "PIXAR_A2": a_nom, "n_pixels": int(ok.sum()),
               "relative_header_area_median": float(np.median(rel[ok])), "relative_header_area_p1_p99": np.percentile(rel[ok], [1, 99]).tolist()}
        row["ratio_header_area_median"], row["ratio_header_area_scatter"] = robust(r_hdr)
        row["ratio_no_area_median"], row["ratio_no_area_scatter"] = robust(r_none)
        if area_ext is not None:
            r_ext = rate[ok] * C / area_ext[ok] / cal[ok]
            row["ratio_area_ext_median"], row["ratio_area_ext_scatter"] = robust(r_ext)
            row["area_ext_vs_header_median"] = float(np.median(area_ext[ok] / rel[ok]))
        rows.append(row)
        print(f"{row['file']:45s} {det:9s} {filt:6s} C {C:.4f} n {row['n_pixels']:8d}  ratio (header area) median {row['ratio_header_area_median']:.4f} "
              f"scatter {row['ratio_header_area_scatter']:.4f} | no area {row['ratio_no_area_median']:.4f} ({row['ratio_no_area_scatter']:.4f})"
              + (f" | AREA ext {row['ratio_area_ext_median']:.4f} ({row['ratio_area_ext_scatter']:.4f})" if area_ext is not None else "")
              + f" | rel. header area median {row['relative_header_area_median']:.4f} [{row['relative_header_area_p1_p99'][0]:.4f}, "
              f"{row['relative_header_area_p1_p99'][1]:.4f}]")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
