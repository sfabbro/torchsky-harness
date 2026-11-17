"""--loader torchfits against --loader astropy on the 12 sample calibrated files and the 12 count-rate files.
   python tests/check_loader_equivalence.py --rate-dir DIR --cal-dir DIR [--cal-dir DIR2] [--out FILE]
Compared (for dq_ignore_bits 0 and 2): science and error at usable pixels (bit equal), the data-quality plane read both ways (exact), the usable mask
against the prototype's own validity expression (exact), AREA (bit equal), every header key and value of the WCS header used (equal, with type),
PIXAR_A2, DETECTOR, INSTRUME.  Also counts the pixels where science / error differ outside the usable set (by construction: zero fill)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def compare(path, rate_path, ignore, rate_wcs):
    import torchfits
    from fieldop import data
    data.DQ_IGNORE_BITS = ignore
    data.RATE_WCS = rate_wcs
    data.LOADER = "astropy"
    a = data.read_exposure_file(path, rate_path)
    data.LOADER = "torchfits"
    b = data.read_exposure_file(path, rate_path)
    src = rate_path or path
    ref_valid = np.isfinite(a["sci"]) & np.isfinite(a["err"]) & (a["err"] > 0) & ((a["dq"].astype(np.int64) & ~ignore) == 0)
    u = b["usable"]
    out = {"file": os.path.basename(src), "ignore": ignore, "shape": list(a["sci"].shape), "n_usable": int(u.sum())}
    out["usable_mask_identical"] = bool((u == ref_valid).all())
    out["n_mask_differences"] = int((u != ref_valid).sum())
    out["sci_bit_equal_on_usable"] = bool(np.array_equal(a["sci"][u], b["sci"][u]))
    out["err_bit_equal_on_usable"] = bool(np.array_equal(a["err"][u], b["err"][u]))
    out["sci_max_abs_diff_usable"] = float(np.abs(a["sci"][u] - b["sci"][u]).max())
    out["err_max_abs_diff_usable"] = float(np.abs(a["err"][u] - b["err"][u]).max())
    nu = ~u
    out["n_unusable"] = int(nu.sum())
    out["sci_differs_on_unusable"] = int((~((a["sci"][nu] == b["sci"][nu]) | (np.isnan(a["sci"][nu]) & np.isnan(b["sci"][nu])))).sum())
    out["err_differs_on_unusable"] = int((~((a["err"][nu] == b["err"][nu]) | (np.isnan(a["err"][nu]) & np.isnan(b["err"][nu])))).sum())
    ihdu = 1 + [str(torchfits.read_extname(str(src), i) or "").strip().upper() for i in range(1, int(torchfits.read_num_hdus(str(src))))].index("DQ")
    dq_t = torchfits.read(str(src), hdu=ihdu).numpy().astype(np.int64)
    out["dq_plane_identical"] = bool(np.array_equal(dq_t, a["dq"].astype(np.int64)))
    out["area_identical"] = (a["area"] is None and b["area"] is None) or bool(np.array_equal(a["area"], b["area"]))
    ha, hb = a["w"], b["w"]
    out["pixar_a2_identical"] = bool((np.isnan(a["pixar_a2"]) and np.isnan(b["pixar_a2"])) or a["pixar_a2"] == b["pixar_a2"])
    out["detector_instrument_identical"] = a["detector"] == b["detector"] and a["instrument"] == b["instrument"]
    # header: astropy dict of the same header the prototype built its WCS from
    from astropy.io import fits
    hsrc = path if (rate_path and rate_wcs == "cal") else src
    sh = fits.getheader(hsrc, "SCI")
    hd_a = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
    hd_b = b["header_hd"]
    keys = sorted(set(hd_a) | set(hd_b))
    diffs = []
    for k in keys:
        va, vb = hd_a.get(k, "<absent>"), hd_b.get(k, "<absent>")
        same = (va == vb) and (type(va) is type(vb) or (isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool)))
        if not same:
            diffs.append([k, repr(va), repr(vb), type(va).__name__, type(vb).__name__])
    out["header_keys"] = len(keys)
    out["header_differences"] = diffs
    # the WCS objects built from the headers must give the same pixel -> sky positions
    pc = torch.tensor([[0.0, 0.0], [1000.3, 1500.7], [2047.0, 2047.0]], dtype=torch.float64)
    out["wcs_positions_bit_equal"] = bool(torch.equal(ha.pixel_to_world_differentiable(pc, origin=0), hb.pixel_to_world_differentiable(pc, origin=0)))
    return out


def is_complete(path):
    from astropy.io import fits
    try:
        with fits.open(path, memmap=False) as h:
            for n in ("SCI", "ERR", "DQ"):
                h[n].data  # noqa: B018
        return True
    except Exception:  # noqa: BLE001  (truncated sample files: ValueError on reshape)
        return False


def compare_partial(path):
    """Truncated calibrated sample files (20 MB of 33 to 118 MB): header and the (complete) SCI plane only."""
    import torchfits
    from astropy.io import fits
    out = {"file": os.path.basename(path), "kind": "cal_header_and_sci_only"}
    with fits.open(path, memmap=False) as h:
        sci_a = h["SCI"].data.astype(np.float32)
        sh = h["SCI"].header
        h0 = h[0].header
    hd_a = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
    sci_b = torchfits.read(path, hdu=1).numpy().astype(np.float32)
    out["sci_identical_including_nan"] = bool(np.array_equal(sci_a, sci_b, equal_nan=True))
    hd_b = dict(torchfits.read_header(path, "SCI"))
    hd_b = {k: v for k, v in hd_b.items() if k and k not in ("COMMENT", "HISTORY", "")}
    diffs = []
    for k in sorted(set(hd_a) | set(hd_b)):
        va, vb = hd_a.get(k, "<absent>"), hd_b.get(k, "<absent>")
        if not ((va == vb) and (type(va) is type(vb) or (isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool)))):
            diffs.append([k, repr(va), repr(vb)])
    p0 = dict(torchfits.read_header(path, 0))
    out["header_keys"] = len(hd_a)
    out["header_differences"] = diffs
    out["detector_instrument_identical"] = (str(h0.get("DETECTOR", "")).strip().upper() == str(p0.get("DETECTOR", "")).strip().upper()
                                            and str(h0.get("INSTRUME", "")).strip().upper() == str(p0.get("INSTRUME", "")).strip().upper())
    out["pixar_a2_identical"] = bool(float(sh.get("PIXAR_A2", float("nan"))) == float(hd_b.get("PIXAR_A2", float("nan"))) or
                                     (np.isnan(float(sh.get("PIXAR_A2", float("nan")))) and np.isnan(float(hd_b.get("PIXAR_A2", float("nan"))))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-dir", required=True)
    ap.add_argument("--cal-dir", action="append", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows, partial = [], []
    for rp in sorted(glob.glob(os.path.join(a.rate_dir, "*_rate.fits"))):
        base = os.path.basename(rp).replace("_rate.fits", "")
        cands = [os.path.join(d, base + "_cal.fits") for d in a.cal_dir if os.path.exists(os.path.join(d, base + "_cal.fits"))]
        cal = max(cands, key=os.path.getsize)   # the complete file when there is one (11 of the 12 local calibrated files are truncated to 20 MB)
        full = is_complete(cal)
        if not full:
            partial.append(compare_partial(cal))
        for ign in (0, 2):
            if full:
                rows.append(dict(compare(cal, None, ign, "cal"), kind="cal"))
            rows.append(dict(compare(cal, rp, ign, "cal"), kind="rate+calWCS"))
            if ign == 0:
                rows.append(dict(compare(cal, rp, ign, "rate"), kind="rate+rateWCS"))
    keys = ["usable_mask_identical", "sci_bit_equal_on_usable", "err_bit_equal_on_usable", "dq_plane_identical", "area_identical", "pixar_a2_identical",
            "detector_instrument_identical", "wcs_positions_bit_equal"]
    summ = {k: all(r[k] for r in rows) for k in keys}
    summ["header_differences_total"] = sum(len(r["header_differences"]) for r in rows)
    summ["n_comparisons"] = len(rows)
    summ["cal_full_files"] = sorted({r["file"] for r in rows if r["kind"] == "cal"})
    summ["cal_header_and_sci_only_files"] = len(partial)
    summ["cal_partial_sci_identical"] = all(r["sci_identical_including_nan"] for r in partial)
    summ["cal_partial_header_differences"] = sum(len(r["header_differences"]) for r in partial)
    summ["cal_partial_detector_pixar_identical"] = all(r["detector_instrument_identical"] and r["pixar_a2_identical"] for r in partial)
    summ["max_sci_diff_usable"] = max(r["sci_max_abs_diff_usable"] for r in rows)
    summ["max_err_diff_usable"] = max(r["err_max_abs_diff_usable"] for r in rows)
    summ["unusable_pixels_sci_differ_total"] = sum(r["sci_differs_on_unusable"] for r in rows)
    print(json.dumps(summ, indent=1))
    for r in rows:
        if r["header_differences"]:
            print(r["file"], r["kind"], r["header_differences"][:6])
    if a.out:
        json.dump({"summary": summ, "rows": rows, "partial": partial}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
