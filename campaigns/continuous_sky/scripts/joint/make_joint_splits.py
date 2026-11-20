#!/usr/bin/env python3
"""Add HST WFC3/IR exposures (training only) to the SMACS list of exposures of protocol v1.
usage: make_joint_splits.py SPLITS_IN HST_DIR CANFAR_HST_DIR OUT [--filters F105W,F125W,F140W,F160W]
Coverage of a box = fraction of a 21 x 21 lattice of the box (size_arcsec, tangent plane at the box centre) that lands on the chip
(1014 x 1014) through the file's own header solution (astropy, all distortion terms). Entries: see INSTRUMENT_INTERFACE.md section 7.
dither_group = first six characters of the rootname (the visit) + filter. Held-out sets are unchanged (JWST only)."""
import sys
import json
import glob
import os
import warnings
import datetime
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
warnings.simplefilter("ignore")
sin, hdir, cdir, out = sys.argv[1:5]
filters = (sys.argv[6] if len(sys.argv) > 6 else "F105W,F125W,F140W,F160W").split(",")
sp = json.load(open(sin)); cover_min = sp.get("cover_min", 0.5)
new, skipped = [], []
for f in sorted(glob.glob(os.path.join(hdir, "*_flt.fits"))):
    with fits.open(f) as h:
        p, s = h[0].header, h[1].header
        if p.get("INSTRUME") != "WFC3" or p.get("DETECTOR") != "IR" or p.get("FILTER") not in filters:
            skipped.append((os.path.basename(f), "instrument or filter")); continue
        w = WCS(s, fobj=h); ny, nx = h[1].data.shape
        cov = {}
        for b in sp["boxes"]:
            g = (np.arange(21) - 10) / 20.0 * b["size_arcsec"] / 3600.0
            dx, dy = np.meshgrid(g, g)
            ra = b["ra"] + dx / np.cos(np.radians(b["dec"])); dec = b["dec"] + dy
            x, y = w.all_world2pix(ra.ravel(), dec.ravel(), 0, quiet=True)
            c = float(np.mean((x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)))
            if c >= cover_min: cov[str(b["id"])] = round(c, 4)
        root = os.path.basename(f).replace("_flt.fits", "")
        if not cov:
            skipped.append((root, "does not cover a box")); continue
        new.append({"file": os.path.basename(f), "rootname": root, "path": os.path.join(cdir, os.path.basename(f)), "program": str(p.get("PROPOSID")),
                    "detector": "IR", "filter": p["FILTER"], "mjd": float(p["EXPSTART"]), "exptime_s": float(p["EXPTIME"]), "pa_v3": float(p.get("PA_V3", 0.0)),
                    "instrument": "WFC3_IR", "boxes": cov, "dither_group": f"{root[:6]}|{p['FILTER']}", "visit_key": root[:6]})
have = {e["rootname"] for e in sp["exposures"]}
new = [e for e in new if e["rootname"] not in have]
sp["exposures"] += new
for k, v in sp["splits"].items():
    if "train" in v:
        v["train"] = v["train"] + [e["rootname"] for e in new]; v["n_train"] = len(v["train"])
        v["rule"] = v.get("rule", "") + " | HST WFC3/IR exposures added as training only (18 Nov 2026)"
sp["filters"] = list(sp["filters"]) + sorted({e["filter"] for e in new} - set(sp["filters"]))
sp["n_exposures"] = len(sp["exposures"])
sp["notes"] = (sp.get("notes") or []) + [f"HST WFC3/IR added {datetime.datetime.utcnow().isoformat()}Z by make_joint_splits.py: {len(new)} exposures, training only; "
                                           "held-out exposures, masks and the scorer are unchanged (JWST)."]
json.dump(sp, open(out, "w"), indent=1)
import collections
print("added", len(new), dict(collections.Counter(e["filter"] for e in new)), "skipped", len(skipped), dict(collections.Counter(r for _, r in skipped)))
print("groups per filter", {f: len({e["dither_group"] for e in new if e["filter"] == f}) for f in filters})
print("exposure time per filter [s]", {f: round(sum(e["exptime_s"] for e in new if e["filter"] == f)) for f in filters})
