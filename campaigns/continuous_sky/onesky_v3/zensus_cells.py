"""Spectroscopic-redshift content of every HEALPix order-5 cell of the zensus catalogue, to choose the first cell of the rest-frame test.

    python zensus_cells.py --hats /arc/projects/hats/zensus/20261019/zensus_20261019 --out /arc/projects/ots/adass2026/onesky_v3/zensus_cells

Read only. One Parquet file per cell (HATS: dataset/Norder=5/Dir=*/Npix=<nested index>.parquet); only the redshift and class columns are read.
Per cell: rows; rows with a finite spectroscopic redshift; of those, "good" = wspec >= 0.9 (the catalogue's calibrated probability that the
value is right); good extragalactic rows (z >= 0.01) in redshift bins; rows with a photometric redshift. The cell centre is given in
equatorial, ecliptic and Galactic coordinates: SPHEREx covers a position more often at high ecliptic latitude, and low Galactic latitude
means crowding and dust.

Written: cells.csv (one line per cell), top.json (the best cells by good extragalactic redshifts below z = 0.6, with the surveys that
supplied them: column `ref`), schema.txt.

Stated limits: zensus holds no magnitudes, so "bright enough for SPHEREx" is not known here; z < 0.6 stands in for it. A cell is 3.4 square
degrees; counts are not corrected for masks.
"""

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ZB = [0.01, 0.1, 0.3, 0.6, 1.0, 2.0, 99.0]


def one(path):
    import pyarrow.parquet as pq

    m = re.search(r"Npix=(\d+)", path)
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    cols = [c for c in ("zspec", "wspec", "zphot", "prob_str") if c in names]
    t = pf.read(columns=cols)
    n = t.num_rows

    def col(c):
        return t.column(c).to_numpy(zero_copy_only=False).astype(np.float64) if c in cols else np.full(n, np.nan)

    z, w, zp, ps = col("zspec"), col("wspec"), col("zphot"), col("prob_str")
    has = np.isfinite(z)
    good = has & (w >= 0.9)
    gal = good & (z >= 0.01)
    r = {"npix": int(m.group(1)), "rows": n, "zspec": int(has.sum()), "good": int(good.sum()), "good_star": int((good & (z < 0.01)).sum()),
         "zphot": int(np.isfinite(zp).sum()), "star_class": int((ps > 0.5).sum())}
    for lo, hi in zip(ZB[:-1], ZB[1:]):
        r[f"z{lo:g}_{hi:g}"] = int((gal & (z >= lo) & (z < hi)).sum())
    r["z_median"] = float(np.median(z[gal])) if gal.any() else float("nan")
    return r


def refs_of(path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.ParquetFile(path).read(columns=["zspec", "wspec", "ref"])
    z = t.column("zspec").to_numpy(zero_copy_only=False).astype(np.float64)
    w = t.column("wspec").to_numpy(zero_copy_only=False).astype(np.float64)
    keep = np.isfinite(z) & (w >= 0.9) & (z >= 0.01) & (z < 0.6)
    vc = pc.value_counts(t.column("ref").filter(keep)).to_pylist()
    return sorted(((str(v["values"]), int(v["counts"])) for v in vc), key=lambda kv: -kv[1])[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import healpy as hp
    import pyarrow.parquet as pq
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.hats, "dataset", "Norder=5", "**", "*.parquet"), recursive=True))
    files = [f for f in files if "Npix=" in f]
    print(f"{len(files)} cell files", flush=True)
    open(os.path.join(a.out, "schema.txt"), "w").write(str(pq.ParquetFile(files[0]).schema_arrow))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(one, files, chunksize=16)):
            rows.append(r)
            if k % 1000 == 0:
                print(f"{k} of {len(files)}", flush=True)
    by = {}
    for r in rows:  # a cell may be split over several files
        b = by.setdefault(r["npix"], dict.fromkeys(r, 0) | {"npix": r["npix"], "z_median": []})
        for key, v in r.items():
            if key == "z_median":
                b[key].append((v, r["good"]))
            elif key != "npix":
                b[key] += v
    cells = []
    for npix, b in sorted(by.items()):
        zm = [v for v, n in b["z_median"] if np.isfinite(v)]
        b["z_median"] = float(np.median(zm)) if zm else float("nan")
        lon, lat = hp.pix2ang(32, npix, nest=True, lonlat=True)
        c = SkyCoord(lon * u.deg, lat * u.deg, frame="icrs")
        e, g = c.barycentricmeanecliptic, c.galactic
        b.update(ra=float(lon), dec=float(lat), ecl_lon=float(e.lon.deg), ecl_lat=float(e.lat.deg), gal_l=float(g.l.deg), gal_b=float(g.b.deg))
        b["bright_proxy"] = b["z0.01_0.1"] + b["z0.1_0.3"] + b["z0.3_0.6"]
        cells.append(b)
    keys = list(cells[0])
    with open(os.path.join(a.out, "cells.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for b in cells:
            f.write(",".join(f"{b[k]:.5g}" if isinstance(b[k], float) else str(b[k]) for k in keys) + "\n")
    top = sorted(cells, key=lambda b: -b["bright_proxy"])[:40]
    path_of = {}
    for f in files:
        path_of.setdefault(int(re.search(r"Npix=(\d+)", f).group(1)), []).append(f)
    for b in top[:16]:
        acc = {}
        for f in path_of[b["npix"]]:
            for name, n in refs_of(f):
                acc[name] = acc.get(name, 0) + n
        b["refs"] = sorted(acc.items(), key=lambda kv: -kv[1])[:10]
    json.dump({"n_cells": len(cells), "total_rows": sum(b["rows"] for b in cells), "total_zspec": sum(b["zspec"] for b in cells),
               "total_good": sum(b["good"] for b in cells), "top": top}, open(os.path.join(a.out, "top.json"), "w"), indent=1)
    print("total rows", sum(b["rows"] for b in cells), "zspec", sum(b["zspec"] for b in cells), "good", sum(b["good"] for b in cells))
    for b in top[:25]:
        print(f"cell {b['npix']:5d} ra {b['ra']:7.2f} dec {b['dec']:+6.2f} ecl lat {b['ecl_lat']:+6.1f} gal b {b['gal_b']:+6.1f}  good z<0.6 {b['bright_proxy']:7d} "
              f"(0.01-0.1 {b['z0.01_0.1']}, 0.1-0.3 {b['z0.1_0.3']}, 0.3-0.6 {b['z0.3_0.6']}); 0.6-1 {b['z0.6_1']}, 1-2 {b['z1_2']}, >2 {b['z2_99']}; "
              f"rows {b['rows']}; refs {b.get('refs', '')}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
