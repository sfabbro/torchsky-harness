#!/usr/bin/env python3
"""Figures for questions 2-4 from the results files."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, "results", n)))
F = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]

# Q2: radial profiles about the source for source 1, sky / prediction / mosaic, all in MJy/sr
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axs.ravel(), F):
    d = np.load(os.path.join(HERE, "results", f"q2_profiles_{f}.npz"))
    g = d["g"]; X, Y = np.meshgrid(g, g)
    for key, lab in (("sky", "fitted sky alone"), ("pred", "sky through kernel (prediction)"), ("mosaic", "official mosaic")):
        im = d[key]; i0, j0 = np.unravel_index(np.argmax(im), im.shape)
        r = np.hypot(X - g[j0], Y - g[i0]).ravel(); v = im.ravel()
        bins = np.geomspace(0.005, 2.0, 40); idx = np.digitize(r, bins)
        prof = [v[idx == k].mean() if (idx == k).any() else np.nan for k in range(1, len(bins))]
        ax.loglog(np.sqrt(bins[1:] * bins[:-1]), np.clip(prof, 1e-4, None), label=lab)
    ax.set_title(f + ", star at (-9.0, -5.0)\""); ax.set_xlabel("radius [arcsec]"); ax.set_ylabel("MJy/sr")
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "q2_source_profiles.png"), dpi=100)

# Q3: flux ratio against aperture
q3 = R("q3_flux.json")
fig, ax = plt.subplots(figsize=(7, 4))
for i, f in enumerate(F):
    v = q3[f]
    ax.plot([0.3, 1.0, 2.0], [v["galaxy_ratio_median"][k] for k in ("0.3", "1.0", "2.0")], "o-", color=plt.cm.viridis(i / 5), label=f)
    ax.plot([0.3], [v["compact_ratio_median"]], "s", color=plt.cm.viridis(i / 5), ms=4)
    ax.plot([9.0], [v["totals"]["ratio_level_subtracted"]], "D", color=plt.cm.viridis(i / 5))
q1 = R("q1_kernel.json")
for i, f in enumerate(F):
    ax.plot([2.0], [1 / q1["filters"][f]["positions"]["footprint_centre"]["encircled_about_origin_levels_norm"]["2.0"]], "x", color=plt.cm.viridis(i / 5), ms=9)
ax.axhline(1, c="grey", ls=":")
ax.set_xscale("log"); ax.set_xlabel("aperture radius [arcsec] (diamonds at 9: whole 20.48\" box)"); ax.set_ylabel("fitted sky / official mosaic")
ax.set_title("flux ratio (circles: 8 bright galaxies, squares: 20 compact; x: 1 / kernel fraction within 2\")", fontsize=9)
ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "q3_flux_ratio.png"), dpi=100)

# Q4: wing map for F200W
w = np.load(os.path.join(HERE, "results", "q4_wing_F200W.npz"))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
ext = [-12.26, 12.26, -12.26, 12.26]
for a, key, t in zip(ax, ("band", "full", "wing"), ("fitted sky, F200W band", "prediction, full kernel", "kernel parts beyond +-32 px")):
    im = a.imshow(w[key], origin="lower", extent=ext, vmin=0, vmax=0.4 if key != "wing" else 0.25, cmap="magma")
    a.set_title(t + " [MJy/sr]", fontsize=9); plt.colorbar(im, ax=a, fraction=0.046)
    a.add_patch(plt.Rectangle((-11.24, -11.24), 22.48, 22.48, fill=False, ec="c", lw=0.8))
for a in ax: a.set_xlabel("X (west) [arcsec]")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "q4_wing_map_F200W.png"), dpi=100)
print("ok")
