"""Builds a SMOKE-TEST multi-dither data set for tilefit from the six real SMACS cal files of testdata/nircam (module B, one SW and one LW detector image per band).
For every file, 5 'dithers' are made by cropping a 1024 x 1024 window (different integer-pixel origins within +-15 px; the WCS is shifted consistently: CRPIX - origin,
SIP relative to CRPIX is unchanged) around the SMACS pointing. Pixel data of the 5 'dithers' are therefore the SAME real pixels (same noise), consistently
registered: enough to run every code path (selection, held-out dithers, shared exposures, tiles), NOT to measure anything: held-out scores of such data are
optimistic. Output: runs/tilefit_testdata/*.fits (about 12 MB each)."""
import os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "runs", "tilefit_testdata")
SRC = os.path.join(ROOT, "testdata", "nircam")
OFFS = [(0, 0), (12, -9), (-14, 10), (8, 15), (-10, -13)]
RA, DEC = 110.8268, -73.4544


def main(force=False):
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(SRC)):
        if not f.endswith("_cal.fits"):
            continue
        dst0 = os.path.join(OUT, f.replace("_00002_", "_00001_").replace("_00003_", "_00001_").replace("_00004_", "_00001_"))
        if not force and os.path.exists(dst0):
            n += 5
            continue
        with fits.open(os.path.join(SRC, f), memmap=False) as h:
            h0, hs = h[0].header.copy(), h["SCI"].header.copy()
            sci, err, dq = h["SCI"].data.astype(np.float32), h["ERR"].data.astype(np.float32), h["DQ"].data.astype(np.uint32)
        w = WCS(hs)
        px, py = w.all_world2pix(RA, DEC, 0)
        ny, nx = sci.shape
        for k, (ox, oy) in enumerate(OFFS):
            x0 = int(np.clip(round(float(px)) - 512 + ox, 0, nx - 1024)); y0 = int(np.clip(round(float(py)) - 512 + oy, 0, ny - 1024))
            hs2 = hs.copy(); hs2["CRPIX1"] = hs["CRPIX1"] - x0; hs2["CRPIX2"] = hs["CRPIX2"] - y0
            name = f.replace("_00002_", f"_{k + 1:05d}_").replace("_00003_", f"_{k + 1:05d}_").replace("_00004_", f"_{k + 1:05d}_")
            hl = fits.HDUList([fits.PrimaryHDU(header=h0), fits.ImageHDU(sci[y0:y0 + 1024, x0:x0 + 1024], header=hs2, name="SCI"),
                               fits.ImageHDU(err[y0:y0 + 1024, x0:x0 + 1024], name="ERR"), fits.ImageHDU(dq[y0:y0 + 1024, x0:x0 + 1024], name="DQ")])
            hl.writeto(os.path.join(OUT, name), overwrite=True)
            n += 1
    print(f"{n} files in {OUT}")
    return OUT


if __name__ == "__main__":
    main("--force" in sys.argv)
