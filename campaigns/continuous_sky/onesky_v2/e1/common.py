"""Common ground for experiment E1 (any-scale sky representations, no instrument).

Everything every method needs so that comparisons are fair:
  * make_spec(seed)            -> an analytic synthetic sky (stars, galaxies, diffuse light)
  * render(spec, n, zoom)      -> its exact image on an n x n pixel grid, at any magnification
  * make_data(spec, n, zoom)   -> dict(truth, noisy, sigma)
  * score(model, data)         -> error by octave of angular scale, PSNR, chi2 against truth
  * save_result(...)           -> one JSON per (method, budget, variant) in results/

Units: one scene unit = one pixel at zoom 1. The scene is the square [0, L)^2 with L = 512.
At magnification `zoom` an n x n image shows the central n/zoom scene units, so every
feature is `zoom` times larger in pixels. Nothing is interpolated: the scene is re-evaluated.
Pixel values are the mean over OVERSAMPLE^2 sub-samples (stars: exact pixel integrals).
"""
from __future__ import annotations
import json
import math
import os
import time
import numpy as np
from scipy.special import erf, gammaincinv

L = 512.0
OVERSAMPLE = 4
FLOOR_SIGMA = 0.45          # width of an unresolved source, scene units (declared resolution floor)
HERE = os.path.dirname(os.path.abspath(__file__))


def make_spec(seed: int = 0, n_star: int = 350, n_gal: int = 140, n_wave: int = 260) -> dict:
    r = np.random.default_rng(seed)
    stars = dict(x=r.uniform(0, L, n_star), y=r.uniform(0, L, n_star),
                 flux=10 ** r.uniform(1.0, 4.0, n_star))
    gal = dict(x=r.uniform(0, L, n_gal), y=r.uniform(0, L, n_gal),
               re=10 ** r.uniform(0.0, np.log10(60.0), n_gal),       # 1 to 60 units
               n=r.uniform(0.7, 4.0, n_gal), q=r.uniform(0.3, 1.0, n_gal),
               pa=r.uniform(0, np.pi, n_gal), flux=10 ** r.uniform(2.3, 4.8, n_gal),
               bulge=r.uniform(0, 0.6, n_gal) * (r.uniform(size=n_gal) < 0.5),
               arm=r.uniform(0.15, 0.6, n_gal) * (r.uniform(size=n_gal) < 0.4),
               m=r.integers(2, 4, n_gal), pitch=r.uniform(0.25, 0.6, n_gal),
               phase=r.uniform(0, 2 * np.pi, n_gal))
    # clumps: small Gaussians inside some galaxies (irregular structure)
    cl = []
    for i in range(n_gal):
        if gal['re'][i] > 6 and r.uniform() < 0.5:
            for _ in range(r.integers(2, 7)):
                rr = gal['re'][i] * r.uniform(0.3, 1.8); ph = r.uniform(0, 2 * np.pi)
                cl.append((gal['x'][i] + rr * np.cos(ph), gal['y'][i] + rr * gal['q'][i] * np.sin(ph),
                           gal['re'][i] * r.uniform(0.04, 0.15), gal['flux'][i] * r.uniform(0.003, 0.03)))
    cl = np.array(cl).reshape(-1, 4)
    # diffuse light: exp of a sum of plane waves with a power-law spectrum (scales 8 to 512 units)
    lam = 10 ** r.uniform(np.log10(8.0), np.log10(512.0), n_wave)
    th = r.uniform(0, 2 * np.pi, n_wave)
    waves = dict(kx=2 * np.pi / lam * np.cos(th), ky=2 * np.pi / lam * np.sin(th),
                 amp=(lam / 512.0) ** 1.0 * r.normal(0, 1, n_wave) * 0.22, ph=r.uniform(0, 2 * np.pi, n_wave))
    return dict(seed=seed, stars=stars, gal=gal, clumps=cl, waves=waves, diffuse_level=6.0, sky_level=2.0)


def _bn(n):
    return gammaincinv(2 * n, 0.5)


def _coords(n, zoom, over):
    """Scene coordinates of sub-sample centres, shape (n*over,)."""
    half = n / (2.0 * zoom)
    return L / 2 - half + (np.arange(n * over) + 0.5) / (over * zoom)


def render(spec: dict, n: int = 512, zoom: float = 1.0, parts: bool = False):
    over = OVERSAMPLE
    xs = _coords(n, zoom, over)
    img_f = np.zeros((n * over, n * over))
    g = spec['gal']
    step = 1.0 / (over * zoom)
    x0 = xs[0]
    for i in range(len(g['x'])):
        re, nn, q, pa = g['re'][i], g['n'][i], g['q'][i], g['pa'][i]
        R = 9.0 * re
        i0 = max(0, int((g['x'][i] - R - x0) / step)); i1 = min(n * over, int((g['x'][i] + R - x0) / step) + 1)
        j0 = max(0, int((g['y'][i] - R - x0) / step)); j1 = min(n * over, int((g['y'][i] + R - x0) / step) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1] - g['x'][i], xs[j0:j1] - g['y'][i])
        c, s = np.cos(pa), np.sin(pa)
        u = c * X + s * Y; v = (-s * X + c * Y) / q
        rr = np.sqrt(u * u + v * v) + 1e-6
        soft = np.sqrt(rr * rr + (0.35 * FLOOR_SIGMA) ** 2)      # tiny softening so cusps are finite
        b = _bn(nn)
        prof = np.exp(-b * ((soft / re) ** (1.0 / nn) - 1.0))
        norm = 2 * np.pi * nn * re * re * q * np.exp(b) * math.gamma(2 * nn) / b ** (2 * nn)
        disc = prof / norm
        if g['arm'][i] > 0:
            phi = np.arctan2(v, u)
            disc = disc * (1 + g['arm'][i] * np.cos(g['m'][i] * (phi - np.log(rr / re + 0.05) / np.tan(g['pitch'][i])) + g['phase'][i])
                           * (1 - np.exp(-(rr / (0.5 * re)) ** 2)))
        val = (1 - g['bulge'][i]) * disc
        if g['bulge'][i] > 0:
            rb = 0.25 * re; b4 = _bn(4.0)
            pb = np.exp(-b4 * ((np.sqrt(u * u + v * v * q * q + (0.35 * FLOOR_SIGMA) ** 2) / rb) ** 0.25 - 1.0))
            nb = 2 * np.pi * 4 * rb * rb * np.exp(b4) * math.gamma(8) / b4 ** 8
            val = val + g['bulge'][i] * pb / nb
        img_f[j0:j1, i0:i1] += g['flux'][i] * val
    for (cx, cy, sg, fl) in spec['clumps']:
        R = 5 * sg
        i0 = max(0, int((cx - R - x0) / step)); i1 = min(n * over, int((cx + R - x0) / step) + 1)
        j0 = max(0, int((cy - R - x0) / step)); j1 = min(n * over, int((cy + R - x0) / step) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1] - cx, xs[j0:j1] - cy)
        img_f[j0:j1, i0:i1] += fl * np.exp(-0.5 * (X * X + Y * Y) / sg ** 2) / (2 * np.pi * sg * sg)
    gal_img = img_f.reshape(n, over, n, over).mean(axis=(1, 3))          # per scene unit^2
    # stars: exact pixel integrals of a Gaussian of width FLOOR_SIGMA (scene units)
    edges = L / 2 - n / (2.0 * zoom) + np.arange(n + 1) / zoom
    star_img = np.zeros((n, n)); s2 = FLOOR_SIGMA * math.sqrt(2)
    st = spec['stars']
    for k in range(len(st['x'])):
        ex = 0.5 * (erf((edges[1:] - st['x'][k]) / s2) - erf((edges[:-1] - st['x'][k]) / s2))
        ey = 0.5 * (erf((edges[1:] - st['y'][k]) / s2) - erf((edges[:-1] - st['y'][k]) / s2))
        if ex.max() < 1e-12 or ey.max() < 1e-12:
            continue
        star_img += st['flux'][k] * np.outer(ey, ex) * zoom * zoom         # per scene unit^2
    # diffuse light at pixel centres (smooth: shortest wave is 8 units)
    xc = L / 2 - n / (2.0 * zoom) + (np.arange(n) + 0.5) / zoom
    Xc, Yc = np.meshgrid(xc, xc); w = spec['waves']; f = np.zeros((n, n))
    for m in range(len(w['kx'])):
        f += w['amp'][m] * np.cos(w['kx'][m] * Xc + w['ky'][m] * Yc + w['ph'][m])
    diff_img = spec['diffuse_level'] * np.exp(f) + spec['sky_level']
    # surface brightness per scene unit^2 -> counts per pixel
    area = 1.0 / (zoom * zoom)
    out = dict(stars=star_img * area, galaxies=gal_img * area, diffuse=diff_img * area)
    total = out['stars'] + out['galaxies'] + out['diffuse']
    return (total, out) if parts else total


def make_data(spec: dict, n: int = 512, zoom: float = 1.0, noise_seed: int = 1) -> dict:
    """truth, a noisy realisation, and the per-pixel sigma (Poisson-like: sqrt(truth) plus a floor)."""
    truth, parts = render(spec, n, zoom, parts=True)
    sigma = np.sqrt(truth + 1.0)
    noisy = truth + np.random.default_rng(noise_seed).normal(0, 1, truth.shape) * sigma
    return dict(truth=truth, noisy=noisy, sigma=sigma, parts=parts, n=n, zoom=zoom)


BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1024)]   # wavelength in pixels


def band_rms(img: np.ndarray) -> dict:
    """RMS of an image restricted to each octave of wavelength (pixels)."""
    n = img.shape[0]; F = np.fft.fft2(img) / (n * n)
    fx = np.fft.fftfreq(n); K = np.sqrt(fx[None, :] ** 2 + fx[:, None] ** 2)
    out = {}
    for lo, hi in BANDS:
        m = (K > 1.0 / hi) & (K <= 1.0 / lo)
        out[f'{lo}-{hi}'] = float(np.sqrt((np.abs(F[m]) ** 2).sum()))
    return out


def score(model: np.ndarray, data: dict) -> dict:
    """All errors are against the noise-free truth."""
    t = data['truth']; r = model - t
    peak = float(np.percentile(t, 99.9))
    mse = float((r ** 2).mean())
    tb = band_rms(t - t.mean()); rb = band_rms(r)
    return dict(
        psnr=float(10 * np.log10(peak ** 2 / mse)),
        rms=float(np.sqrt(mse)),
        chi2_truth=float(((r / data['sigma']) ** 2).mean()),      # 0 is perfect; 1 means errors as large as the noise
        chi2_data=float((((model - data['noisy']) / data['sigma']) ** 2).mean()),
        frac_err_by_band={k: rb[k] / max(tb[k], 1e-30) for k in tb},
        resid_rms_by_band=rb, truth_rms_by_band=tb,
        neg_fraction=float((model < 0).mean()),
    )


def save_result(method: str, budget: int, variant: str, model: np.ndarray, data: dict, *,
                n_params_image: int, n_params_shared: int = 0, fit_seconds: float = 0.0,
                target: str = 'noisy', extra: dict | None = None, outdir: str | None = None) -> dict:
    """variant: free text such as 'seed0_zoom1.0'. target: which image was fitted ('noisy' or 'truth')."""
    outdir = outdir or os.path.join(HERE, 'results'); os.makedirs(outdir, exist_ok=True)
    res = dict(method=method, budget=int(budget), variant=variant, target=target,
               n_params_image=int(n_params_image), n_params_shared=int(n_params_shared),
               fit_seconds=float(fit_seconds), n=int(data['n']), zoom=float(data['zoom']), **score(model, data))
    if extra:
        res['extra'] = extra
    stem = f'{method}__{budget}__{variant}__{target}'
    with open(os.path.join(outdir, stem + '.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.save(os.path.join(outdir, stem + '.npy'), model.astype(np.float32))
    return res


if __name__ == '__main__':
    t0 = time.time(); spec = make_spec(0); d = make_data(spec)
    print('render s', round(time.time() - t0, 1), 'min/median/max', d['truth'].min(), np.median(d['truth']), d['truth'].max())
    for k, v in d['parts'].items():
        print(k, 'sum', v.sum(), 'max', v.max())
    print('score of noisy vs truth', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in score(d['noisy'], d).items() if k in ('psnr', 'chi2_truth')})
    print('frac err by band, noisy', {k: round(v, 3) for k, v in score(d['noisy'], d)['frac_err_by_band'].items()})
    from scipy.ndimage import gaussian_filter
    print('score of smoothed noisy', round(score(gaussian_filter(d['noisy'], 1.0), d)['chi2_truth'], 3))
    for z in (1.4, 2.0, 2.8):
        dz = make_data(spec, zoom=z); print('zoom', z, 'sum', dz['truth'].sum(), 'max', dz['truth'].max())
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    for a, (im, ttl) in zip(ax, [(d['truth'], 'truth, zoom 1'), (d['noisy'], 'noisy'), (make_data(spec, zoom=2.0)['truth'] , 'truth, zoom 2'), (d['parts']['diffuse'], 'diffuse part')]):
        a.imshow(np.arcsinh(im / 3.0), origin='lower', cmap='gray'); a.set_title(ttl); a.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figs', 'synthetic_sky.png'), dpi=110)
