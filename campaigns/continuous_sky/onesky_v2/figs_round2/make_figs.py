#!/usr/bin/env python3
"""Figures for the grid-sky versus item-sky comparison (SMACS J0723, JWST NIRCam). Run: python3 make_figs.py
Reuses flux_check/flux_ratio.py (sky_coeffs) and kernel_check (kc_common, q1_kernel). Heavy steps are cached in ./cache (prep.py, kern.py)."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.transforms import blended_transform_factory
from astropy.io import fits
from scipy.ndimage import map_coordinates
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prep
import kern
CACHE = prep.CACHE
FILTS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
PUB = [0.033, 0.050, 0.066, 0.092, 0.116, 0.145]
COL = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#8C1D1D"]       # blue to red, colour-blind-safe
C_GRID, C_ITEM, C_MOS = "#D55E00", "#0072B2", "#222222"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "figure.dpi": 100, "savefig.dpi": 200})
CEN = (-0.155, -1.302)    # brightest extended object: peak of the 0.4 arcsec box-smoothed F200W mosaic inside the central 14 arcsec (same peak in all six mosaics to 0.06 arcsec)

# ---------------------------------------------------------------- data
for key in ("grid", "items"):
    if not os.path.exists(f"{CACHE}/band_{key}.npz"):
        im, p = prep.band_images(prep.STATES[key]); np.savez(f"{CACHE}/band_{key}.npz", im=im, pitch=p)
    if not os.path.exists(f"{CACHE}/kern_{key}.npz"):
        kern.run(key, prep.STATES[key])
class Im:
    """Image of square cells, value at X = (col - cx) p, Y = (row - cy) p; X to the west, Y to the north, row 0 at the south."""
    def __init__(s, d, p, cx, cy): s.d, s.p, s.cx, s.cy = d, p, cx, cy
    def sample(s, X, Y): return map_coordinates(s.d, [Y / s.p + s.cy, X / s.p + s.cx], order=1, mode="nearest")
    def crop(s, hx, cx=0.0, cy=0.0):
        j0, j1 = int(np.floor((cx - hx) / s.p + s.cx)), int(np.ceil((cx + hx) / s.p + s.cx)) + 1
        i0, i1 = int(np.floor((cy - hx) / s.p + s.cy)), int(np.ceil((cy + hx) / s.p + s.cy)) + 1
        ext = [(j0 - s.cx - .5) * s.p, (j1 - 1 - s.cx + .5) * s.p, (i0 - s.cy - .5) * s.p, (i1 - 1 - s.cy + .5) * s.p]
        return s.d[i0:i1, j0:j1], ext
mos, grid, items = {}, {}, {}
gi, ii = np.load(f"{CACHE}/band_grid.npz"), np.load(f"{CACHE}/band_items.npz")
for f, n in enumerate(FILTS):
    fn = f"/home/claude/work/kernel_check/dl/cut_{n}.fits"; h = fits.getheader(fn)
    assert h["PC1_1"] < 0 and h.get("PC2_2", 1) > 0
    mos[n] = Im(fits.getdata(fn).astype(float), abs(h["CDELT1"]) * 3600, h["CRPIX1"] - 1, h["CRPIX2"] - 1)
    N = gi["im"].shape[-1]
    grid[n] = Im(gi["im"][f].astype(float), float(gi["pitch"]), (N - 1) / 2, (N - 1) / 2)
    items[n] = Im(ii["im"][f].astype(float), float(ii["pitch"]), (N - 1) / 2, (N - 1) / 2)

# ---------------------------------------------------------------- orientation check: correlate sky with mosaic under the four flips
print("Orientation check (correlation of sky band image with mosaic resampled to the sky grid, central 16 arcsec; flips applied to the sky):")
g1 = np.arange(-8, 8, 0.04); XX, YY = np.meshgrid(g1, g1)
for key, S in (("grid", grid), ("items", items)):
    for n in ("F150W", "F444W"):
        m = mos[n].sample(XX, YY); s = S[n].sample(XX, YY)
        r = {nm: np.corrcoef(np.arcsinh(m / .05).ravel(), np.arcsinh(t / .05).ravel())[0, 1] for nm, t in
             (("as is", s), ("flip X", s[:, ::-1]), ("flip Y", s[::-1]), ("flip both", s[::-1, ::-1]), ("transpose", s.T))}
        print(f"  {key} {n}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))

# ---------------------------------------------------------------- figure 1
def sky_three_ways(n):
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 9.6), constrained_layout=True)
    titles = ["Official mosaic", "Grid sky (pixel-grid model)", "Item sky (Gaussian-items model)"]
    for row, (hx, cx, cy, plo, phi) in enumerate([(10.0, 0, 0, 1, 99.9), (3.0, CEN[0], CEN[1], 1, 99.5)]):
        mm, _ = mos[n].crop(hx, cx, cy)
        lo, hi = np.percentile(mm, [plo, phi]); lw = max(np.percentile(mm, 90) - lo, 1e-3)
        norm = AsinhNorm(linear_width=lw, vmin=lo, vmax=hi)
        for c, (S, t) in enumerate(zip((mos[n], grid[n], items[n]), titles)):
            d, ext = S.crop(hx, cx, cy)
            im = ax[row, c].imshow(d, origin="lower", extent=ext, norm=norm, cmap="viridis", interpolation="nearest")
            ax[row, c].set_xlim(cx - hx, cx + hx); ax[row, c].set_ylim(cy - hx, cy + hx)
            ax[row, c].set_title(t + (" (20 arcsec box)" if row == 0 else " (zoom, 6 arcsec)"))
            ax[row, c].set_xlabel("Offset towards west (arcsec)"); ax[row, c].set_ylabel("Offset towards north (arcsec)" if c == 0 else "")
        cb = fig.colorbar(im, ax=ax[row, :], shrink=0.85, pad=0.01); cb.set_label(f"{n} surface brightness (MJy/sr), asinh stretch")
        ticks = [t for t in (0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if lo <= t <= hi]; cb.set_ticks(ticks); cb.set_ticklabels([f"{t:g}" for t in ticks])
        ax[row, 0].text(0.02, 0.97, f"limits {lo:.3g} to {hi:.3g}", transform=ax[row, 0].transAxes, va="top", color="w", fontsize=8)
    ax[1, 0].plot([cx - 2.5, cx - 1.5] if False else [CEN[0] - 2.5, CEN[0] - 1.5], [CEN[1] - 2.6] * 2, color="w", lw=2); ax[1, 0].text(CEN[0] - 2.0, CEN[1] - 2.45, "1 arcsec", color="w", ha="center", fontsize=8)
    fig.suptitle(f"{n}: the same sky area and the same colour stretch, three ways (stretch set by the mosaic)", fontsize=12)
    fig.savefig(f"{HERE}/sky_three_ways_{n}.png"); plt.close(fig)
for n in ("F150W", "F444W"): sky_three_ways(n)

# ---------------------------------------------------------------- figure 2: enclosed flux ratios
rad = np.geomspace(0.1, 2.0, 40); fine = 0.01; gf = np.arange(-3.4, 3.4 + 1e-9, fine); XF, YF = np.meshgrid(gf + CEN[0], gf + CEN[1]); DF = np.hypot(XF - CEN[0], YF - CEN[1])
def enclosed(S):
    z = S.sample(XF, YF); bg = np.median(z[(DF > 2) & (DF < 3)])
    return np.array([(z[DF < r] - bg).sum() * fine ** 2 for r in rad]), bg
fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True, constrained_layout=True)
prof_out = {}
for a, n in zip(ax.flat, FILTS):
    em, bm = enclosed(mos[n]); eg, bgg = enclosed(grid[n]); ei, bi = enclosed(items[n])
    a.axhline(1, color=C_MOS, lw=1)
    a.plot(rad, eg / em, color=C_GRID, lw=2, label="grid sky"); a.plot(rad, ei / em, color=C_ITEM, lw=2, label="item sky")
    a.set_xscale("log"); a.set_title(n); prof_out[n] = (eg / em, ei / em, bm, bgg, bi, em[-1])
    for r_, col in ((0.3, "0.6"), (1.0, "0.6")): a.axvline(r_, color=col, lw=0.6, ls=":")
    a.set_xticks([0.1, 0.2, 0.5, 1, 2]); a.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"]); a.minorticks_off()
    a.grid(alpha=0.25)
for a in ax[1]: a.set_xlabel("Aperture radius (arcsec)")
for a in ax[:, 0]: a.set_ylabel("Enclosed flux / mosaic")
ax[0, 0].legend(loc="upper right", frameon=False)
ax[0, 0].set_ylim(0, None)
fig.suptitle("Enclosed flux of the brightest galaxy, relative to the official mosaic (1 = agrees)", fontsize=12)
fig.savefig(f"{HERE}/sky_ratio_profiles.png"); plt.close(fig)
for n in FILTS:
    a_, b_, bm, bg, bi, e2 = prof_out[n]; k = lambda r: int(np.argmin(abs(rad - r)))
    print(f"{n}: ratio grid/item at 0.3\" {a_[k(.3)]:.2f}/{b_[k(.3)]:.2f}, 1\" {a_[k(1)]:.2f}/{b_[k(1)]:.2f}, 2\" {a_[-1]:.2f}/{b_[-1]:.2f}; annulus bg mosaic {bm:.3f} grid {bg:.3f} item {bi:.3f} MJy/sr; mosaic flux in 2\" {e2:.3g} MJy/sr arcsec^2")

# ---------------------------------------------------------------- figures 3a, 3b: kernels
KG, KI = np.load(f"{CACHE}/kern_grid.npz"), np.load(f"{CACHE}/kern_items.npz")
def ee_of(K, f):
    r, p = K["r"], K[f"p{f}"]; ring = 2 * np.pi * r * p
    return r, np.concatenate([[0], np.cumsum(0.5 * (ring[1:] + ring[:-1]) * np.diff(r))])
def ticks(a):
    tr = blended_transform_factory(a.transData, a.transAxes)
    for w, c in zip(PUB, COL): a.plot([w], [0.0], marker="^", ms=7, color=c, mec="k", mew=0.5, transform=tr, clip_on=False, zorder=5)
for kind in ("profiles", "encircled"):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True)
    for a, K, t in zip(ax, (KG, KI), ("Grid sky fit", "Item sky fit")):
        for f, n in enumerate(FILTS):
            if kind == "profiles":
                r, p = K["r"], K[f"p{f}"]; m = r >= 0.004; a.loglog(r[m], p[m], color=COL[f], lw=1.6, label=n)
            else:
                r, e = ee_of(K, f); m = (r >= 0.02) & (r <= 10); a.semilogx(r[m], e[m], color=COL[f], lw=1.6, label=n)
        a.set_title(t); a.set_xlabel("Radius from kernel centre (arcsec)"); a.grid(alpha=0.25, which="both" if kind == "profiles" else "major"); ticks(a)
    if kind == "profiles":
        ax[0].set_ylabel("Azimuthally averaged kernel (per arcsec$^2$)"); ax[0].set_xlim(0.004, 10); ax[0].set_ylim(1e-4, 1e3)
        ax[1].legend(loc="upper right", frameon=False, ncol=2, title="filter", fontsize=9)
        fig.suptitle("Learned instrument kernel, radial profile (triangles on the axis: published width at half maximum per filter)", fontsize=11)
    else:
        ax[0].set_ylabel("Fraction of kernel inside radius"); ax[0].set_xlim(0.02, 10); ax[0].set_ylim(0, 1.02)
        for a in ax: a.axvline(2.0, color="0.5", ls=":", lw=1); a.text(2.1, 0.03, "2 arcsec", color="0.35", fontsize=8)
        ax[1].legend(loc="lower right", frameon=False, ncol=2, title="filter", fontsize=9, bbox_to_anchor=(0.97, 0.12))
        fig.suptitle("Learned instrument kernel, encircled fraction (triangles on the axis: published width at half maximum per filter)", fontsize=11)
    fig.savefig(f"{HERE}/kernel_{kind}.png"); plt.close(fig)
for nm, K in (("grid", KG), ("items", KI)):
    print(nm, "kernel fraction inside 0.1/0.5/2 arcsec:", [[round(float(np.interp(x, *ee_of(K, f))), 2) for x in (0.1, 0.5, 2.0)] for f in range(6)])

# ---------------------------------------------------------------- figure 4: item population
s = torch.load("/home/claude/work/states/items004b_state.pt", map_location="cpu", weights_only=False)["skies"][0]
floor = 0.4 * 0.04; cap = 0.32
lw = s["lw"][0].double().clamp(max=np.log(cap)); w = torch.sqrt(floor ** 2 + torch.exp(2 * lw)).numpy()
wl, ws = w.max(1), w.min(1); pos = s["pos"][0].double().numpy(); coef = torch.exp(s["lc"][0].double()).sum(1).numpy()
top = np.argsort(coef)[::-1][:20000]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
b = np.geomspace(0.012, 0.4, 60)
ax[0].hist(wl, b, color=C_ITEM, alpha=0.75, label="larger width"); ax[0].hist(ws, b, color=C_GRID, alpha=0.75, label="smaller width")
ax[0].set_xscale("log"); ax[0].axvline(floor, color="k", ls="--", lw=1); ax[0].axvline(cap, color="k", ls="-.", lw=1)
ax[0].text(floor * 1.04, 0.97, "floor 0.016", transform=ax[0].get_xaxis_transform(), va="top", fontsize=8, rotation=90)
ax[0].text(cap * 0.93, 0.97, "cap 0.32", transform=ax[0].get_xaxis_transform(), va="top", ha="right", fontsize=8, rotation=90)
ax[0].set_xticks([0.016, 0.03, 0.05, 0.1, 0.2, 0.32]); ax[0].set_xticklabels(["0.016", "0.03", "0.05", "0.1", "0.2", "0.32"], rotation=0, fontsize=8); ax[0].minorticks_off()
ax[0].set_xlabel("Item width, Gaussian sigma (arcsec)"); ax[0].set_ylabel("Number of items (of 60,000)"); ax[0].legend(loc="upper center", frameon=False)
ax[0].set_title("Item widths")
ax[1].hist(ws / wl, np.linspace(0, 1, 51), color="#009E73"); ax[1].set_xlabel("Axis ratio (smaller width / larger width)"); ax[1].set_ylabel("Number of items"); ax[1].set_title("Item axis ratios")
sc = ax[2].scatter(pos[top, 0], pos[top, 1], c=wl[top], s=0.6, norm=matplotlib.colors.LogNorm(floor, cap), cmap="viridis", rasterized=True)
ax[2].set_aspect("equal"); ax[2].set_xlabel("Position towards west (arcsec)"); ax[2].set_ylabel("Position towards north (arcsec)")
ax[2].set_title("Centres of the 20,000 items with the largest total coefficient"); ax[2].set_facecolor("#f2f2f2")
cb = fig.colorbar(sc, ax=ax[2], shrink=0.85); cb.set_label("Larger width (arcsec)")
cb.set_ticks([0.02, 0.05, 0.1, 0.2, 0.32]); cb.set_ticklabels(["0.02", "0.05", "0.1", "0.2", "0.32"]); cb.ax.minorticks_off()
fig.savefig(f"{HERE}/item_population.png"); plt.close(fig)
q = lambda x: np.percentile(x, [5, 50, 95]).round(4)
print("items: larger width 5/50/95%", q(wl), "smaller", q(ws), "axis ratio", np.percentile(ws / wl, [5, 50, 95]).round(3))
print("items at cap (larger width >= 0.99*sqrt(floor^2+cap^2)):", int((wl >= 0.99 * np.hypot(floor, cap)).sum()), "; both widths at cap:", int((ws >= 0.99 * np.hypot(floor, cap)).sum()),
      "; smaller width within 1% of floor:", int((ws <= 1.01 * floor).sum()), "; larger within 1% of floor:", int((wl <= 1.01 * floor).sum()))
print("positions range X", pos[:, 0].min(), pos[:, 0].max(), "Y", pos[:, 1].min(), pos[:, 1].max(), "; top-20000 share of total coefficient", coef[top].sum() / coef.sum())
print("top items: median larger width", np.median(wl[top]), "; others", np.median(np.delete(wl, top)))
