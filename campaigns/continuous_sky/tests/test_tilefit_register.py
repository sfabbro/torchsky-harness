"""CPU test of tilefit.py A12 (--coarse-shift-jwst, per-exposure coarse registration), synthetic, ~20 s.
Two tiles (128 x 128 px at 0.04 arcsec, 92 px apart in x, overlapping), four NIRCam bands (F090W, F150W, F200W, F277W) each in its OWN visit (the engine takes its reference from another visit), 4 dithers x 2
detector images per band; the two detector images of a dither cover different parts of the two tiles (so every exposure is in both tiles with different pixel counts). The
header positions of some dithers of F090W and F277W are wrong by about 40 mas (one pixel) with different values per dither; all the data are samples of one analytic sky (Gaussian sources, width
growing with the band), noise sigma 0.3. Checks: (1) the combined shift of every exposure is minus the injected error within 5 mas; (2) the spread across tiles is < 5 mas;
(3) the shifts are applied to tx, ty of both tiles (the pixel positions move by the shift, in pixels); (4) the combination is the pixel-count weighted mean (direct unit test of
combine_tile_shifts); (5) coarse_shifts.json has the documented content.
Run: python3 tests/test_tilefit_register.py"""
import json, math, os, sys, tempfile, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, HERE)  # tests/ in the work tree, or next to tilefit.py on arc
import tilefit as T  # noqa: E402

SCALE = 0.04
TOL = 5.0   # mas
BANDS = ["F090W", "F150W", "F200W", "F277W"]
# injected header error (mas, x, y) per (band, dither index): positions in the header are WRONG by this; the registration must return minus this
# F150W and F200W are each other's nearest-wavelength reference and are kept clean; F090W and F277W (leaves: nobody's reference) carry the errors, so the reference coadds are right
INJECT = {("F090W", 1): (40.0, 20.0), ("F090W", 2): (-35.0, 45.0), ("F277W", 0): (40.0, -30.0), ("F277W", 3): (-45.0, 10.0)}


def sky_fn(X, Y, src, sig):
    v = torch.zeros_like(X)
    for (x0, y0, amp) in src:
        v = v + amp * torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sig ** 2))
    return v


def build(seed=3):
    g = torch.Generator().manual_seed(seed)
    ns = 70
    src = [(float(torch.rand(1, generator=g)) * 215 + 2, float(torch.rand(1, generator=g)) * 124 + 2, 5 + 45 * float(torch.rand(1, generator=g)) ** 2) for _ in range(ns)]
    tiles = [types.SimpleNamespace(id=k, role="train", thead={"NAXIS1": 128, "NAXIS2": 128, "CD2_2": SCALE / 3600.0}, x0=x0, exps=[]) for k, x0 in enumerate((0, 92))]
    truth = {}
    for bi, b in enumerate(BANDS):
        sig_b = math.sqrt(1.7 ** 2 + (T.V.REG.fwhm_arcsec[b] / SCALE / 2.3548) ** 2) if T.V is not None and b in T.V.REG.fwhm_arcsec else 1.0 + 0.2 * bi
        for dth in range(4):
            off = ((0.31 * dth) % 1 - 0.5, (0.17 + 0.43 * dth) % 1 - 0.5)   # sub-pixel dither of the sampling grid
            for det, (gx0, gx1) in (("NRCB1", (0, 130)), ("NRCB2", (90, 220))):
                name = f"jw01234{bi + 1:03d}001_02101_{dth + 1:05d}_{det.lower()}_cal.fits"
                err = INJECT.get((b, dth), (0.0, 0.0))
                truth[name] = err
                for t in tiles:
                    xs, ys = torch.meshgrid(torch.arange(gx0, gx1, dtype=torch.float64), torch.arange(0, 128, dtype=torch.float64), indexing="xy")
                    Xg, Yg = xs.flatten() + off[0], ys.flatten() + off[1]       # true global position of the pixel centres
                    tx, ty = Xg - t.x0, Yg                                    # tile pixel coordinates (true)
                    m = (tx > -6) & (tx < 134)
                    Xg, Yg, tx, ty = Xg[m], Yg[m], tx[m], ty[m]
                    d = sky_fn(Xg, Yg, src, sig_b) + 0.3 * torch.randn(Xg.shape, generator=g, dtype=torch.float64)
                    t.exps.append({"name": name, "band": b, "detname": det, "tile": t.id, "d": d.float(), "ivar": torch.full(d.shape, 1 / 0.09), "bkg0": 0.0, "holdout": False, "val": False,
                                   "tx": (tx + err[0] / 1000 / SCALE).float(), "ty": (ty + err[1] / 1000 / SCALE).float(), "npix": int(d.numel())})
    return tiles, truth


def main():
    T.load_engine("smacs_four_v38")
    tiles, _ = build()     # first build only to get the bands for the registry
    T.V.set_registry(T.V.registry_from_exposures(tiles[0].exps))
    tiles, truth = build()
    ref_tx = {(t.id, e["name"]): e["tx"].clone() for t in tiles for e in t.exps}
    a = types.SimpleNamespace(coarse_shift_jwst=0.2, coarse_shift=0.0)
    out = tempfile.mkdtemp()
    res = T.coarse_register_tiles(tiles, a, out, apply=True)
    ex = res["exposures"]
    worst, worst_sp = 0.0, 0.0
    for name, (ex_, ey_) in truth.items():
        r = ex[name]
        assert r["registered"] and len(r["tiles"]) == 2, (name, r["tiles"])
        err = math.hypot(r["shift_mas"][0] + ex_, r["shift_mas"][1] + ey_)
        worst, worst_sp = max(worst, err), max(worst_sp, r["spread_mas"])
        assert err < TOL, f"{name}: combined shift {r['shift_mas']} vs injected {(ex_, ey_)}: error {err:.1f} mas"
        assert r["spread_mas"] < TOL, f"{name}: spread {r['spread_mas']:.1f} mas"
    print(f"recovery: worst |recovered - (-injected)| over {len(truth)} exposures = {worst:.2f} mas (tolerance {TOL}); worst spread across tiles {worst_sp:.2f} mas")
    inj_mag = [math.hypot(*v) for v in truth.values() if any(v)]
    print(f"  injected |error| {min(inj_mag):.0f}-{max(inj_mag):.0f} mas on {sum(1 for v in truth.values() if any(v))} of {len(truth)} exposures")
    # (3) applied to both tiles
    for t in tiles:
        for e in t.exps:
            r = ex[e["name"]]
            dpx = (e["tx"] - ref_tx[(t.id, e["name"])]).double()
            assert abs(float(dpx.mean()) - r["shift_mas"][0] / 1000 / SCALE) < 1e-3 and float(dpx.std()) < 1e-3, "shift not applied as a constant in pixels"
    # residual header error after the correction, per exposure, is the recovery error (same number seen from the data)
    # NB the registration is RELATIVE to the cross-band references (engine design): errors in a band that is itself a reference bias the others, so the test puts them in leaf bands
    # (4) combination = pixel-count weighted mean
    comb, sp = T.combine_tile_shifts([{"tile": 0, "shift_mas": [10.0, 0.0], "npix": 3000}, {"tile": 1, "shift_mas": [-30.0, 40.0], "npix": 1000}])
    assert np.allclose(comb, [0.0, 10.0]) and abs(sp - 40 * math.sqrt(2)) < 1e-9, (comb, sp)
    # (5) json
    js = json.load(open(os.path.join(out, "coarse_shifts.json")))
    e0 = next(iter(js["exposures"].values()))
    for k in ("band", "detector", "programme", "visit", "tiles", "shift_mas", "spread_mas", "chi2_ratio", "by_tile"):
        assert k in e0, k
    assert set(js["summary"]["by_band"]) == set(BANDS) and js["summary"]["by_programme"]["01234"]["n"] == len(truth) and js["summary"]["n_spread_gt_10mas"] == 0
    print("coarse_shifts.json ok; summary |shift| mas", {k: round(v, 1) for k, v in js["summary"]["abs_shift_mas_all"].items()})
    # R = 0 leaves the engine untouched (the main() hook is not entered); direct call with R = 0 changes nothing either
    tiles2, _ = build()
    before = [e["tx"].clone() for t in tiles2 for e in t.exps]
    for t in tiles2:
        T.V.coarse_register_all(t.exps, t.thead, torch.device("cpu"), types.SimpleNamespace(coarse_shift_jwst=0.0, coarse_shift=0.0))
    assert all(torch.equal(b, e["tx"]) for b, e in zip(before, [e for t in tiles2 for e in t.exps])), "R = 0 changed positions"
    print("ALL PASS")


if __name__ == "__main__":
    main()
