#!/usr/bin/env python3
"""Independent check of the learned kernel and flux scale of the fit fieldop_smacs_fix_200_s0 (JWST NIRCam, SMACS J0723).

Runs every step in order, one process each (one thread, each under 4 minutes):
  fetch_mosaics.py <F>   official pipeline mosaics (read-only fetch from CANFAR storage, cut-outs kept in dl/)
  q1_kernel.py           question 1: kernel shape, widths, encircled fractions, normalisation, decomposition
  q3_flux.py             question 3: flux scale against the mosaics (aperture 0.3", 1", 2", box totals)
  q2_q4_render.py q2     question 2: stars through the whole model (sky alone, prediction, mosaic)
  q2_node_response.py    question 2: the sharpest sky object (one tent node) through the kernel
  q2_q4_render.py q4     question 4: mean level carried by kernel parts beyond +-32 detector pixels
  bg_terms.py            question 4: fitted background terms (constant, Legendre, 8x8 pattern) over the footprints
  crosscheck_render.py   my renderer against the package's render_fft
  make_figs.py           figures
Shared loading code: kc_common.py. Results: results/*.json; figures: figs/*.png; report: KERNEL_CHECK_REPORT.md.
Usage: python3 kernel_check.py [step ...]   (default: all)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [["fetch_mosaics.py", f] for f in ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")] + [
    ["q1_kernel.py"], ["q3_flux.py"], ["q2_q4_render.py", "q2"], ["q2_node_response.py"], ["q2_q4_render.py", "q4"],
    ["bg_terms.py"], ["crosscheck_render.py"], ["make_figs.py"]]

if __name__ == "__main__":
    want = sys.argv[1:]
    for st in STEPS:
        if want and st[0] not in want:
            continue
        print(">>", " ".join(st), flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, st[0])] + st[1:], cwd=HERE, check=True, timeout=240,
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
