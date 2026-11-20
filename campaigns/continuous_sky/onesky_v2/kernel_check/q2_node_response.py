#!/usr/bin/env python3
"""Question 2, sharpest possible sky source: the model's prediction for a sky that is ONE tent node (the narrowest object the 0.04" sky
grid can hold), through the kernel of one exposure per filter (same exposures as q2). FWHM and encircled fraction of the total.
The kernel beyond 2.7" is not rendered; its share is 1 - (kernel integral inside 2.7"), so fractions are of the full unit integral."""
import json
import os
import sys
import numpy as np
import torch
from scipy.signal import fftconvolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kc_common import FILTERS, HERE, PIX, PUB_FWHM, SKY_PITCH, Fit
from q2_q4_render import kernel_table, half_max_fwhm, enc, H, KR, EPSF
torch.set_num_threads(1)
fit = Fit()
res = {}
for f in range(6):
    chan = 0 if f < 3 else 1
    e = int([i for i in fit.s["gref"].numpy() if fit.e_filt[i] == f][0])
    R = fit.R[e]
    h = fit.code(f, det=int(fit.e_det[e]), xi=(0.0, 0.0), e=e)
    logZ = fit.logZ_levels(h, chan, R)
    K, g = kernel_table(fit, h, logZ, R, KR, H)
    X, Y = np.meshgrid(g, g)
    tent = np.clip(1 - np.abs(X) / SKY_PITCH, 0, None) * np.clip(1 - np.abs(Y) / SKY_PITCH, 0, None)
    c = fftconvolve(tent, K[::-1, ::-1], mode="same") * H * H
    tot = SKY_PITCH ** 2
    ee = {k: v / tot for k, v in enc(c, H, 0.0, 0.0, [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]).items()}
    res[FILTERS[f]] = dict(exposure=e, fwhm_arcsec=half_max_fwhm(c, H), fwhm_detector_px=half_max_fwhm(c, H) / PIX[chan],
                           encircled_fraction_of_total=ee, published_fwhm=PUB_FWHM[FILTERS[f]], pixel_convolved_fwhm_approx=EPSF[FILTERS[f]])
    print(FILTERS[f], round(res[FILTERS[f]]["fwhm_arcsec"], 4), {k: round(v, 3) for k, v in ee.items()})
json.dump(res, open(os.path.join(HERE, "results", "q2_single_node_response.json"), "w"), indent=1)
