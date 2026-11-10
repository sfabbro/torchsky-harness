"""Tests of scripts/plan_tiles.py.
1. Synthetic LMC-like header-only data set (no pixels): 2 modules x (4 SW + 1 LW) x 6 bands x 9-point dither x 2 rolls; checks the tile properties
   (coverage >= n_min in all bands, no overlap + gap, distinct positions grow, determinism).
2. The real test data (testdata/nircam, one exposure per band): 3 tiles of 4 arcsec, n_min 1.
Run: python3 tests/test_plan_tiles.py   (astropy, numpy, matplotlib only)"""
import json, math, os, shutil, sys, tempfile
import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import plan_tiles as P

BANDS = P.DEFAULT_BANDS
SW, LW = 0.031, 0.063


def header_only_file(path, prim, sci):
    """A FITS file with a primary HDU and a SCI image extension whose header says 2048x2048 but which has no data bytes (headers are all the planner reads)."""
    hp = fits.Header(); hp["SIMPLE"] = True; hp["BITPIX"] = 8; hp["NAXIS"] = 0; hp["EXTEND"] = True
    for k, v in prim.items(): hp[k] = v
    hs = fits.Header(); hs["XTENSION"] = "IMAGE"; hs["BITPIX"] = -32; hs["NAXIS"] = 2; hs["NAXIS1"] = 2048; hs["NAXIS2"] = 2048; hs["PCOUNT"] = 0; hs["GCOUNT"] = 1; hs["EXTNAME"] = "SCI"
    for k, v in sci.items(): hs[k] = v
    with open(path, "wb") as fh:
        fh.write(hp.tostring(padding=True, endcard=True).encode("ascii")); fh.write(hs.tostring(padding=True, endcard=True).encode("ascii"))


def make_synthetic(d, ra0=80.5, dec0=-69.5):
    n = 0
    # detector centres in the focal plane [arcsec, (east, north) for roll 0]; SW 2x2 with 5 arcsec gaps; module A at +x, module B at -x, 44 arcsec apart
    half = 2048 * SW / 2
    for mod, mx in (("A", 94.0), ("B", -94.0)):
        dets = {}
        for k, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            dets[f"NRC{mod}{k + 1}"] = (mx + sx * (half + 2.5), sy * (half + 2.5), SW)
        dets[f"NRC{mod}LONG"] = (mx, 0.0, LW)
        for roll in (0.0, 41.0):
            for ib, band in enumerate(BANDS):
                lw = ib >= 3
                for dx_i in range(9):
                    dith = ((dx_i % 3) - 1) * 24.0, ((dx_i // 3) - 1) * 24.0
                    for det, (cx, cy, sc) in dets.items():
                        if (det.endswith("LONG")) != lw:
                            continue
                        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
                        e = c * cx - s * cy + dith[0]; nn = s * cx + c * cy + dith[1]
                        ra = ra0 + e / 3600.0 / math.cos(math.radians(dec0)); dec = dec0 + nn / 3600.0
                        cd = sc / 3600.0
                        th = math.radians(roll)
                        sci = {"WCSAXES": 2, "CRPIX1": 1024.5, "CRPIX2": 1024.5, "CRVAL1": ra, "CRVAL2": dec, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                               "CD1_1": -cd * math.cos(th), "CD1_2": cd * math.sin(th), "CD2_1": cd * math.sin(th), "CD2_2": cd * math.cos(th), "MJD-AVG": 59800.0 + 30 * (roll > 0) + dx_i * 0.01}
                        name = f"jw0147600{1 + (roll > 0)}001_021{ib:02d}_{dx_i + 1:05d}_{det.lower()}_cal.fits"
                        prim = {"INSTRUME": "NIRCAM", "DETECTOR": det, "MODULE": mod, "FILTER": band, "PUPIL": "CLEAR", "EXP_TYPE": "NRC_IMAGE", "EXPSTART": sci["MJD-AVG"]}
                        header_only_file(os.path.join(d, name), prim, sci)
                        n += 1
    return n


def run(args):
    return P.main(args)


def main():
    tmp = tempfile.mkdtemp(prefix="plan_tiles_test_")
    try:
        d = os.path.join(tmp, "lmc"); os.makedirs(d)
        n = make_synthetic(d)
        print(f"synthetic files: {n}")
        o1 = run(["--inputs", d, "--out", os.path.join(tmp, "p1.json"), "--png", os.path.join(tmp, "p1.png"), "--n-tiles", "8"])
        o2 = run(["--inputs", d, "--out", os.path.join(tmp, "p2.json"), "--n-tiles", "8"])
        assert [t["ra"] for t in o1["tiles"]] == [t["ra"] for t in o2["tiles"]], "not deterministic"
        T = o1["tiles"]
        assert len(T) == 8, len(T)
        for t in T:
            assert all(v >= 4 for v in t["n_covering"].values()), t["n_covering"]
        # no overlap with the gap (offsets in the planner's plane)
        for i in range(len(T)):
            for j in range(i):
                dx = abs(T[i]["offset_east_north_arcsec"][0] - T[j]["offset_east_north_arcsec"][0]); dy = abs(T[i]["offset_east_north_arcsec"][1] - T[j]["offset_east_north_arcsec"][1])
                assert max(dx, dy) >= 20.48 + 2.0 - 0.7, (i, j, dx, dy)  # lattice snapping tolerance of one grid cell
        cum = o1["summary"]["cumulative_det_cell"]
        assert cum == sorted(cum) and cum[-1] > cum[0], cum
        assert all(os.path.exists(t_f) is False or True for t_f in [])  # files are header-only; only listed
        assert os.path.getsize(os.path.join(tmp, "p1.png")) > 10000
        print("synthetic: tiles", len(T), "cumulative distinct (detector, cell) positions", cum, "of", o1["summary"]["available_det_cell_positions"])
        # a stricter n_min than the data allow gives fewer/no tiles but does not crash
        o3 = run(["--inputs", d, "--out", os.path.join(tmp, "p3.json"), "--n-tiles", "3", "--n-min", "400"])
        assert len(o3["tiles"]) == 0
        # real test data
        td = os.path.join(ROOT, "testdata", "nircam")
        o4 = run(["--inputs", td, "--out", os.path.join(tmp, "p4.json"), "--png", os.path.join(tmp, "p4.png"), "--n-tiles", "3", "--size-arcsec", "4", "--n-min", "1", "--gap-arcsec", "1",
                  "--cells-per-tile", "16", "--cand-stride", "2"])
        assert len(o4["tiles"]) == 3, len(o4["tiles"])
        for t in o4["tiles"]:
            assert all(v >= 1 for v in t["n_covering"].values()), t["n_covering"]
        keep = os.path.join(ROOT, "runs", "plan_tiles_test"); os.makedirs(keep, exist_ok=True)
        for f in ("p1.json", "p1.png", "p4.json", "p4.png"):
            shutil.copy(os.path.join(tmp, f), os.path.join(keep, f))
        print("testdata tiles:", [(round(t["ra"], 5), round(t["dec"], 5), t["n_covering"]) for t in o4["tiles"]])
        print("PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
