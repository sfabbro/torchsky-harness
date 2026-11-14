#!/usr/bin/env python3
"""Fetch the official pipeline mosaic of one filter from CANFAR storage (read-only), keep a 40" cut-out around the scoring box
(SCI in MJy/sr, WCS header, PIXAR_A2, PHOTMJSR-related keywords), and delete the full file.  Usage: python3 fetch_mosaics.py F200W"""
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "dl")
BOX = (110.8268, -73.4544)

f = sys.argv[1]
full = os.path.join(DL, f"ref_{f}.fits")
out = os.path.join(DL, f"cut_{f}.fits")
if not os.path.exists(out):
    if not os.path.exists(full):
        subprocess.run(["/home/claude/arc/arcget.sh", f"projects/ots/adass2026/protocol/v1/ref_SMACS_{f}_i2d.fits", full], check=True,
                       stdout=subprocess.DEVNULL)
    with fits.open(full, memmap=True) as hl:
        h0, h1 = hl[0].header, hl["SCI"].header
        w = WCS(h1)
        c = Cutout2D(hl["SCI"].data, SkyCoord(BOX[0] * u.deg, BOX[1] * u.deg), (40 * u.arcsec, 40 * u.arcsec), wcs=w)
        hdr = c.wcs.to_header()
        for k in ("BUNIT", "PIXAR_A2", "PIXAR_SR", "PHOTMJSR", "PHOTUJA2"):
            if k in h1:
                hdr[k] = h1[k]
        for k in ("FILTER", "PUPIL", "TELESCOP", "CAL_VER", "CRDS_CTX", "SKYSUB"):
            if k in h0:
                hdr["P_" + k[:6]] = h0[k]
        err = None
        if "ERR" in hl:
            err = Cutout2D(hl["ERR"].data, SkyCoord(BOX[0] * u.deg, BOX[1] * u.deg), (40 * u.arcsec, 40 * u.arcsec), wcs=w).data
        hdus = [fits.PrimaryHDU(np.asarray(c.data, np.float32), header=hdr)]
        if err is not None:
            hdus.append(fits.ImageHDU(np.asarray(err, np.float32), name="ERR"))
        fits.HDUList(hdus).writeto(out, overwrite=True)
        print(f, "full shape", hl["SCI"].data.shape, "cut", c.data.shape, {k: h1.get(k) for k in ("BUNIT", "PIXAR_A2", "PHOTMJSR")},
              "sky-matched:", h0.get("SKYMATCH"), h1.get("BKGLEVEL"), h1.get("BKGSUB"))
    os.remove(full)
else:
    print("exists", out)
