"""Star-zoom diagnostic (standard diagnostic list in the research plan): does the fitted sky still contain the PSF
(diffraction spikes, a box at the PSF window edge)? For the brightest compact peaks of the drizzle F200W cube, show
3.0" cut-outs of each run's raw sky (<name>_sky.fits) in one SW and one LW band, plus the azimuthal profile of the
sky normalised to its peak. A deconvolved sky should be compact with no spikes; a sky that absorbs the PSF keeps them.

usage: star_zooms.py OUT.png DRIZZLE.fits LABEL=SKY.fits [LABEL=SKY.fits ...]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
SHOW = [2, 5]  # F200W, F444W
HALF = 37  # 75 px = 3.0" at 0.04"/px


def peaks(img, n=4, snr=100, conc_min=0.2):
    """Brightest compact local maxima: peak > snr x NMAD and peak / 5x5 sum > conc_min (stars, not galaxy cores)."""
    from scipy.ndimage import maximum_filter
    d = np.nan_to_num(img)
    nmad = 1.4826 * np.median(np.abs(d - np.median(d)))
    m = (d == maximum_filter(d, 9)) & (d > snr * nmad)
    ys, xs = np.nonzero(m)
    out = []
    for i in np.argsort(-d[ys, xs]):
        y, x = ys[i], xs[i]
        if d[y, x] / d[max(0, y - 2):y + 3, max(0, x - 2):x + 3].sum() > conc_min:
            out.append((int(y), int(x)))
        if len(out) == n:
            break
    return out


def radial(cut):
    yy, xx = np.indices(cut.shape)
    r = np.hypot(yy - HALF, xx - HALF).astype(int)
    ok = np.isfinite(cut).ravel()
    prof = np.bincount(r.ravel()[ok], cut.ravel()[ok], minlength=HALF + 1) / np.maximum(np.bincount(r.ravel()[ok], minlength=HALF + 1), 1)
    return prof[:HALF] / max(prof[0], 1e-30)


def main():
    pad = lambda c: np.pad(c.astype(float), ((0, 0), (HALF, HALF), (HALF, HALF)), constant_values=np.nan)
    out, driz0 = sys.argv[1], fits.getdata(sys.argv[2])
    driz = pad(driz0)
    runs = [(a.split("=", 1)[0], pad(fits.getdata(a.split("=", 1)[1]))) for a in sys.argv[3:]]
    stars = [(y + HALF, x + HALF) for y, x in peaks(driz0[2])]  # padded coordinates
    cols = [("drizzle", driz)] + runs
    fig, ax = plt.subplots(len(stars) * len(SHOW), len(cols) + 1, figsize=(2.2 * (len(cols) + 1), 2.2 * len(stars) * len(SHOW)))
    for si, (y, x) in enumerate(stars):
        for bi, b in enumerate(SHOW):
            row = si * len(SHOW) + bi
            for ci, (lab, cube) in enumerate(cols):
                cut = cube[b, y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
                edge = np.concatenate([cut[:4].ravel(), cut[-4:].ravel(), cut[:, :4].ravel(), cut[:, -4:].ravel()])
                cut = cut - np.nanmedian(edge)  # local background from the cut-out border
                s = max(np.nanmax(cut[HALF - 3:HALF + 4, HALF - 3:HALF + 4]), 1e-30)  # each panel at its own peak
                ax[row, ci].imshow(np.arcsinh(cut / (3e-3 * s)), origin="lower", cmap="magma",
                                   vmin=-1, vmax=np.arcsinh(1 / 3e-3))
                ax[row, ci].set_xticks([]); ax[row, ci].set_yticks([])
                if row == 0:
                    ax[row, ci].set_title(lab, fontsize=8)
                ax[row, -1].semilogy(np.arange(HALF) * 0.04, np.clip(radial(cut), 1e-5, None), lw=1, label=lab)
            ax[row, 0].set_ylabel(f"({x - HALF},{y - HALF}) {BANDS[b]}", fontsize=7)
            ax[row, -1].set_ylim(1e-5, 1.5)
            ax[row, -1].tick_params(labelsize=6)
    ax[0, -1].legend(fontsize=5)
    ax[-1, -1].set_xlabel("radius (arcsec)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("stars (y, x):", [(y - HALF, x - HALF) for y, x in stars])


if __name__ == "__main__":
    main()
