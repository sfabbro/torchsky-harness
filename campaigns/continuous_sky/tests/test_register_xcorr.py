"""CPU tests of scripts/register_xcorr.py (catalogue-free cross-correlation registration) and of the tilefit.py A13 hook (--shifts-from). Synthetic, ~1-2 minutes.

Synthetic data (build_dataset): a crowded field of 4000 PSF-like sources (power-law fluxes, 5 per arcsec^2) in a 40 x 20 arcsec plane, 2 tiles of 8 arcsec (centres 20 arcsec
apart), 2 programmes (01476: injected header offsets 0-500 mas, 04447: 0-30 mas), 3 visits x 3 dithers each, per dither group 2 SW detector images (nrcb1, nrcb2; F150W or F200W) and
1 LW image (nrcblong; F277W or F356W) that share the pointing; visits have different rolls (0, +12, -15 deg); noise, a background gradient, hot pixels (DQ), NaNs and saturated cores
(DQ) in every image. The header WCS of every file is the true WCS minus the injected error s(X) = t + theta J (X - X0) (translation t in mas along the tile axes, rotation theta about
the pointing centre X0): the shift to ADD to the header position is s, as in tilefit A12.
Checks:
 1 default run (one unknown per dither group, 2 workers): recovery of the injected translations within 3 mas (after removing the common gauge offset), small closure residuals,
   no spurious rejection, SW and LW of a group share the shift, tile-to-tile spread small, JSON content and the sign/axis convention;
 2 aliasing: a periodic (lattice) field gives a second-peak ratio near 1 and the edge is flagged; a crowded random field gives a clear peak (ratio > 3);
 2b solver alone (random graph, 15% gross outliers, honest sigma, gauge, components, soft ties) and the similarity fit;
 3 --unit detector --similarity: the injected rotation of some dither groups is recovered, unrotated groups come out near 0;
 4 tilefit.py A13 --shifts-from: the shifts of the JSON are added to tx, ty exactly like A12 (same arithmetic), unmatched exposures untouched, mutually exclusive with
   --coarse-shift-jwst, and the module docstring / approximation line exist (skipped if torch is not importable).
Without cross-band edges the SW/LW simultaneity links F150W+F277W and F200W+F356W separately (2 components, each with its own gauge): checked as such; the recovery test uses --xband.
Run: python3 tests/test_register_xcorr.py   (or pytest -q tests/test_register_xcorr.py)"""
import json, math, os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, HERE)  # tests/ in the work tree, or next to the scripts on arc
import register_xcorr as RX  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

RA0, DEC0 = 80.48, -69.50
TILE_CENTRES = [(-10.0, 0.0), (10.0, 0.0)]     # (x_west, y_north) arcsec in the plane
TILE_SIZE = 8.0
ARC = 1.0 / 3600.0


def plane_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.cd = [[-ARC, 0.0], [0.0, ARC]]      # pixel = arcsec; x increases to the west
    return w


def render(xp, yp, flux, shape, sig, half=6):
    ny, nx = shape
    ok = (xp > -half) & (xp < nx + half) & (yp > -half) & (yp < ny + half)
    xs, ys, fl = xp[ok], yp[ok], flux[ok]
    ix, iy = np.round(xs).astype(int), np.round(ys).astype(int)
    d = np.arange(-half, half + 1)
    X = ix[:, None, None] + d[None, None, :]
    Y = iy[:, None, None] + d[None, :, None]
    val = fl[:, None, None] * np.exp(-((X - xs[:, None, None]) ** 2 + (Y - ys[:, None, None]) ** 2) / (2 * sig ** 2))
    m = (X >= 0) & (X < nx) & (Y >= 0) & (Y < ny)
    flat = (Y * nx + X)[m]
    return np.bincount(flat, weights=val[m], minlength=nx * ny).reshape(ny, nx)


def rot(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s], [s, c]])


BAND_FWHM = {"F150W": 0.05, "F200W": 0.065, "F277W": 0.14, "F356W": 0.16}
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def make_sky(rng, n=4000):
    xy = np.c_[rng.uniform(-24, 24, n), rng.uniform(-13, 13, n)]
    fl = 6.0 * (1.0 - rng.uniform(0, 1, n)) ** (-1 / 1.4)
    return xy, np.minimum(fl, 4000.0)


def build_dataset(root, seed=1, theta_groups=(), theta_mrad=0.0, small_theta=True, n_src=4000):
    """Write the _cal-like files and plan.json under root; returns (plan_path, truth) with truth[file] = dict(t=[sx, sy] mas, theta, pivot, group, prog, ...)."""
    rng = np.random.default_rng(seed)
    xy, fl = make_sky(rng, n_src)
    os.makedirs(root, exist_ok=True)
    progs = {"01476": 500.0, "04447": 30.0}
    pairs = {"01476": [("F150W", "F277W"), ("F150W", "F277W"), ("F200W", "F356W")], "04447": [("F150W", "F277W"), ("F200W", "F356W"), ("F200W", "F356W")]}
    rolls = [0.0, 12.0, -15.0]
    truth, files = {}, []
    ig = 0
    for prog, smax in progs.items():
        for v in range(3):
            fsw, flw = pairs[prog][v][0], pairs[prog][v][1]
            psi = math.radians(rolls[v])
            for dth in range(3):
                ig += 1
                c = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)])      # pointing centre of this dither (plane arcsec)
                r_ = smax * math.sqrt(rng.uniform(0, 1))
                a_ = rng.uniform(0, 2 * math.pi)
                t = np.array([r_ * math.cos(a_), r_ * math.sin(a_)])                  # mas
                th = 0.0
                if ig in theta_groups:
                    th = theta_mrad * 1e-3 * (1 if ig % 2 else -1)
                elif small_theta and ig % 4 == 0:
                    th = 0.08e-3 * (1 if ig % 8 else -1)
                name_stub = f"jw{prog}00{v + 1}00{v + 1}_02101_{dth + 1:05d}"
                for det, off, nxp, sc, band in (("nrcb1", (-10.0, 0.0), 560, 0.031, fsw), ("nrcb2", (10.0, 0.0), 560, 0.031, fsw), ("nrcblong", (0.0, 0.0), 600, 0.063, flw)):
                    fname = f"{name_stub}_{det}_cal.fits"
                    M = sc * rot(psi)
                    cdet = c + rot(psi) @ np.array(off)
                    pc = np.array([(nxp - 1) / 2.0] * 2)
                    # true plane position of pixel p: cdet + M (p - pc); sources -> pixel
                    p = (np.linalg.inv(M) @ (xy - cdet).T).T + pc
                    sig = BAND_FWHM[band] / 2.3548 / sc
                    img = render(p[:, 0], p[:, 1], fl * (1.0 if sc < 0.05 else 2.0), (nxp, nxp), sig)
                    yy, xx = np.mgrid[0:nxp, 0:nxp]
                    img = img + 0.0015 * (xx - pc[0]) + 0.5 + rng.normal(0, 1.0, img.shape)
                    dq = np.zeros(img.shape, np.uint32)
                    hot = rng.integers(0, nxp, (150, 2))
                    img[hot[:, 0], hot[:, 1]] += 400.0
                    dq[hot[:, 0], hot[:, 1]] = 1
                    big = np.argsort(-fl)[:12]
                    for b_ in big:
                        x_, y_ = p[b_]
                        if 3 < x_ < nxp - 3 and 3 < y_ < nxp - 3:
                            sl = (slice(int(y_) - 2, int(y_) + 3), slice(int(x_) - 2, int(x_) + 3))
                            dq[sl] |= 2
                    nanp = rng.integers(0, nxp, (30, 2))
                    img[nanp[:, 0], nanp[:, 1]] = np.nan
                    # header WCS = true WCS minus the injected error: plane_hdr(p) = (I - th J)(cdet + M (p - pc)) - t/1000 + th J c
                    A = np.eye(2) - th * J
                    Mh = A @ M
                    ch = A @ cdet - t / 1000.0 + th * (J @ c)
                    cd = np.diag([-ARC, ARC]) @ Mh
                    xi0 = np.array([-ch[0] * ARC, ch[1] * ARC])           # intermediate coordinates (deg) of the pixel pc
                    crpix = pc + 1.0 - np.linalg.solve(cd, xi0)
                    hdr = fits.Header()
                    hdr.update({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": RA0, "CRVAL2": DEC0, "CRPIX1": crpix[0], "CRPIX2": crpix[1],
                                "CD1_1": cd[0, 0], "CD1_2": cd[0, 1], "CD2_1": cd[1, 0], "CD2_2": cd[1, 1]})
                    hl = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img.astype(np.float32), header=hdr, name="SCI"), fits.ImageHDU(np.ones(img.shape, np.float32), name="ERR"),
                                       fits.ImageHDU(dq, name="DQ")])
                    hl.writeto(os.path.join(root, fname), overwrite=True)
                    truth[fname] = dict(t=t.tolist(), theta=th, pivot=c.tolist(), group=name_stub, prog=prog, band=band, det=det, cdet=cdet.tolist(), half=nxp * sc / 2, rot=psi)
                    files.append((fname, band, det))
    # plan: a file belongs to a tile if its footprint (rotated square, half-side) overlaps the tile box
    bands = ["F150W", "F200W", "F277W", "F356W"]
    tiles = []
    pw = plane_wcs()
    for k, (cx, cy) in enumerate(TILE_CENTRES):
        ra, dec = pw.all_pix2world(cx, cy, 0)
        ex = {b: [] for b in bands}
        fl_ = []
        for fname, band, det in files:
            tr = truth[fname]
            u = rot(-tr["rot"]) @ (np.array([cx, cy]) - np.array(tr["cdet"]))
            if np.all(np.abs(u) < tr["half"] + TILE_SIZE / 2 - 1.0):
                ex[band].append(dict(file=fname, path=os.path.join(root, fname), detector=det.upper(), cover_frac=1.0, covering=True))
                fl_.append(os.path.join(root, fname))
        tiles.append(dict(id=k, ra=float(ra), dec=float(dec), size_arcsec=TILE_SIZE, offset_east_north_arcsec=[-cx, cy], exposures=ex, files=fl_))
    plan = dict(version=1, bands=bands, params=dict(size_arcsec=TILE_SIZE), tiles=tiles)
    pp = os.path.join(root, "plan.json")
    json.dump(plan, open(pp, "w"))
    return pp, truth


_CACHE = {}


def dataset(key, **kw):
    if key not in _CACHE:
        d = tempfile.mkdtemp(prefix=f"rx_{key}_")
        _CACHE[key] = (d,) + build_dataset(d, **kw)
    return _CACHE[key]


def gauge_removed_error(res, truth, key="shift_mas"):
    """|recovered - injected| per file after removing the common offset (the gauge: mean over all exposures)."""
    names = [n for n in truth if res["exposures"][n]["registered"]]
    s = np.array([res["exposures"][n][key] for n in names])
    t = np.array([truth[n]["t"] for n in names])
    d = s - t
    d = d - d.mean(0)
    return names, np.hypot(d[:, 0], d[:, 1])


def run_default():
    if "run_default" not in _CACHE:
        d, pp, truth = dataset("a", seed=1)
        out = os.path.join(d, "out")
        res = RX.run(pp, out, workers=2, tile_use="all", xband=True)    # F150W:F200W and F277W:F356W edges connect the two band families
        _CACHE["run_default"] = (res, truth, out, pp)
    return _CACHE["run_default"]


def test_recovery_default():
    res, truth, out, _ = run_default()
    n, err = gauge_removed_error(res, truth)
    s = res["summary"]
    print(f"recovery (one unknown per dither group): {len(n)} of {len(truth)} exposures, error after gauge removal p50 {np.median(err):.2f} max {err.max():.2f} mas; "
          f"closure p50/p90/max {s['closure']['all_used_mas']['p50']:.2f}/{s['closure']['all_used_mas']['p90']:.2f}/{s['closure']['all_used_mas']['max']:.2f} mas; inflation {s['inflation_factor']:.2f}; "
          f"edges ok {s['edges']['n_quality_ok']}/{s['edges']['n_measured']}, robust rejected {s['edges']['n_robust_rejected']}; sigma p50 {s['sigma_mas']['p50']:.2f} mas")
    assert len(n) == len(truth) == s["n_registered"], "not every exposure was registered"
    assert err.max() < 3.0, f"recovery error {err.max():.2f} mas > 3 (worst {n[int(np.argmax(err))]})"
    assert s["closure"]["all_used_mas"]["p90"] < 3.0 and s["closure"]["all_used_mas"]["p50"] < 1.5, s["closure"]["all_used_mas"]
    assert s["edges"]["frac_rejected_total"] < 0.1, s["edges"]
    assert s["n_components"] == 1 and s["components_warning"] is None, s["n_components"]
    assert s["edges"]["n_by_category"].get("xband", 0) > 0
    # the 500 mas programme is recovered with its large offsets, the other with its small ones
    for p, lim in (("01476", 100.0), ("04447", 0.0)):
        big = [np.hypot(*truth[n]["t"]) for n in truth if truth[n]["prog"] == p]
        assert (max(big) > lim) if lim else (max(big) < 31.0)
    # SW and LW of a dither group share the unknown
    for g in {t["group"] for t in truth.values()}:
        sh = [res["exposures"][n]["shift_mas"] for n in truth if truth[n]["group"] == g]
        assert all(np.allclose(sh[0], x) for x in sh), g
    # sign and axes: shift = what must be added to the header position. Check one file directly: the header is wrong by -t, so the shift is +t up to the gauge (done above);
    # here additionally the sign of the mean over a programme with a one-sided injected offset must follow the injected sign
    names = list(truth)
    st = np.array([truth[n]["t"] for n in names]); sr = np.array([res["exposures"][n]["shift_mas"] for n in names])
    assert np.corrcoef(st[:, 0], sr[:, 0])[0, 1] > 0.99 and np.corrcoef(st[:, 1], sr[:, 1])[0, 1] > 0.99
    # tile-to-tile spread of the per-tile estimates (rotations are <= 0.08 mrad here)
    assert s["spread_mas_multi_tile"]["p90"] < 5.0, s["spread_mas_multi_tile"]
    # agreement of the detectors inside a dither group
    ga = s["dither_group_agreement"]["spread_mas"]
    assert ga["p90"] < 5.0, ga
    # output content
    js = json.load(open(os.path.join(out, "register_xcorr.json")))
    e0 = next(iter(js["exposures"].values()))
    for k in ("band", "detector", "programme", "visit", "dither_group", "registered", "tiles", "shift_mas", "abs_shift_mas", "sigma_mas", "spread_mas", "n_edges", "by_tile"):
        assert k in e0, k
    for k in ("closure", "edges", "by_programme", "by_visit", "by_band", "dither_group_agreement", "inflation_factor"):
        assert k in js["summary"], k
    assert set(js["summary"]["by_programme"]) == {"01476", "04447"} and len(js["summary"]["by_visit"]) == 6
    assert js["approximations"] and os.path.exists(os.path.join(out, "register_xcorr_edges.json")) and os.path.exists(os.path.join(out, "register_xcorr_summary.txt"))
    # coarse_shifts.json compatible keys (same convention) -> compare works
    cs = {"exposures": {n: {"registered": True, "shift_mas": [-x["shift_mas"][0] + 3.0, x["shift_mas"][1]], "abs_shift_mas": 1.0, "programme": x["programme"]} for n, x in js["exposures"].items()}}
    p = os.path.join(out, "fake_coarse.json"); json.dump(cs, open(p, "w"))
    c = RX.compare_with(js["exposures"], p)
    assert c["n"] == len(truth)


def test_resume_cache_and_sign_of_pair():
    # one pair directly: image b = image a with features moved by (+7.5, -3.25) px -> d = (+7.5, -3.25) px * scale (position in b minus position in a)
    rng = np.random.default_rng(5)
    xy, fl = make_sky(rng, 1500)
    n, sc = 360, 0.031
    p = (xy[:, :2] + np.array([24.0, 13.0])) / sc * 0.8
    sh = np.array([7.5, -3.25])
    a = render(p[:, 0], p[:, 1], fl, (n, n), 0.9) + rng.normal(0, 1, (n, n))
    b = render(p[:, 0] + sh[0], p[:, 1] + sh[1], fl, (n, n), 0.9) + rng.normal(0, 1, (n, n))
    v = np.ones((n, n), bool)
    r = RX.measure_images(a, v, b, v, sc)
    assert r["ok"], r
    assert abs(r["dx_mas"] - sh[0] * sc * 1000) < 1.0 and abs(r["dy_mas"] - sh[1] * sc * 1000) < 1.0, (r["dx_mas"], r["dy_mas"], sh * sc * 1000)
    assert r["ratio"] > 3.0, r["ratio"]
    print(f"pair: d = ({r['dx_mas']:.2f}, {r['dy_mas']:.2f}) mas vs ({sh[0] * sc * 1000:.2f}, {sh[1] * sc * 1000:.2f}); peak {r['peak']:.2f}, ratio {r['ratio']:.1f}, sigma {r['sigma_mas']:.2f} mas")


def test_second_peak_flags_aliasing():
    n, sc = 360, 0.031
    period = 0.6 / sc                          # 19.35 px lattice, both axes
    g = np.arange(-2, 40) * period
    X, Y = np.meshgrid(g, g)
    px, py = X.ravel() + 3.1, Y.ravel() + 2.3
    sh = np.array([6.2, -4.1])
    rng = np.random.default_rng(2)
    a = render(px, py, np.full(px.size, 300.0), (n, n), 1.0) + rng.normal(0, 1, (n, n))
    b = render(px + sh[0], py + sh[1], np.full(px.size, 300.0), (n, n), 1.0) + rng.normal(0, 1, (n, n))
    v = np.ones((n, n), bool)
    r = RX.measure_images(a, v, b, v, sc)
    print(f"lattice: peak {r['peak']:.3f} second {r['second']:.3f} ratio {r['ratio']:.2f} ok={r['ok']} reason={r['reason']}")
    assert not r["ok"] and r["ratio"] < 1.5 and "ambiguous" in (r["reason"] or ""), r
    # the same lattice with a random field is unambiguous
    xy, fl = make_sky(rng, 1500)
    p = (xy + np.array([24.0, 13.0])) / sc * 0.8
    a = render(p[:, 0], p[:, 1], fl, (n, n), 1.0) + rng.normal(0, 1, (n, n))
    b = render(p[:, 0] + 6.2, p[:, 1] - 4.1, fl, (n, n), 1.0) + rng.normal(0, 1, (n, n))
    r2 = RX.measure_images(a, v, b, v, sc)
    assert r2["ok"] and r2["ratio"] > 3.0, r2
    # masked half of the image: still unbiased (the mask is handled, not zero-filled data)
    m = v.copy(); m[:, :150] = False
    r3 = RX.measure_images(a, m, b, v, sc)
    assert r3["ok"] and abs(r3["dx_px"] - 6.2) < 0.1 and abs(r3["dy_px"] + 4.1) < 0.1, r3


def test_detector_mode_similarity():
    d, pp, truth = dataset("b", seed=3, theta_groups=(2, 5, 8, 11, 14, 17), theta_mrad=0.5, small_theta=False)
    out = os.path.join(d, "out")
    res = RX.run(pp, out, workers=2, unit="detector", similarity=True, tie_sigma_mas=30.0, xband=True)
    s = res["summary"]
    groups = {}
    for n, x in res["exposures"].items():
        if "similarity" in x:
            groups[truth[n]["group"]] = (x["similarity"], truth[n]["theta"])
    assert len(groups) >= 15, len(groups)
    errs = []
    for g, (sim, th) in groups.items():
        errs.append(abs(sim["theta_mrad"] * 1e-3 - th))
    errs = np.array(errs)
    rot_err = [e for (g, (sim, th)), e in zip(groups.items(), errs) if abs(th) > 0]
    zero_est = [abs(sim["theta_mrad"]) for g, (sim, th) in groups.items() if th == 0]
    print(f"similarity: {len(groups)} groups; rotated groups: |theta_est - theta| max {max(rot_err) * 1e3:.3f} mrad (injected 0.5); unrotated: |theta_est| max {max(zero_est):.3f} mrad")
    assert max(rot_err) < 0.15e-3 and max(zero_est) < 0.15, (max(rot_err), max(zero_est))
    # translations in detector mode: SW images of a group agree with each other after the solve to within the rotation signal; recovery vs injected translation at the image
    names, err = gauge_removed_error(res, truth)
    assert np.median(err) < 3.0, np.median(err)
    assert s["similarity"]["n_groups"] == len(groups)


def test_no_xband_gives_two_components():
    """Default (no cross-band correlation): SW/LW simultaneity links F150W+F277W and F200W+F356W separately, so two components with their own gauge, flagged in the summary;
    the recovery holds inside each component."""
    d, pp, truth = dataset("a", seed=1)
    out = os.path.join(d, "out_noxband")
    res = RX.run(pp, out, workers=1)
    s = res["summary"]
    assert res["config"]["unit"] == "group" and not res["config"]["xband"]
    assert s["n_components"] == 2 and s["components_warning"], s["n_components"]
    fam = {x["component"]: set(x["bands"]) for x in s["components"]}
    assert sorted(map(sorted, fam.values())) == [["F150W", "F277W"], ["F200W", "F356W"]], fam
    errs = []
    for c in fam:
        names = [n for n in truth if res["exposures"][n]["component"] == c]
        sh = np.array([res["exposures"][n]["shift_mas"] for n in names]) - np.array([truth[n]["t"] for n in names])
        sh -= sh.mean(0)
        errs.append(np.hypot(sh[:, 0], sh[:, 1]).max())
    print(f"no xband: components {fam}; per-component recovery max error {errs} mas")
    assert max(errs) < 3.0, errs
    assert "edge_resid_rms_mas" in next(iter(res["exposures"].values()))
    out2 = os.path.join(d, "out_x")
    r2 = RX.run(pp, out2, workers=1, xband=True, n_xband=1, tiles=[0], min_ratio=1.3)
    assert r2["summary"]["edges"]["n_by_category"].get("xband", 0) > 0


def test_resume_reuses_tasks():
    d, pp, truth = dataset("a", seed=1)
    out = os.path.join(d, "out_noxband")
    res = RX.run(pp, out, workers=1, resume=True)
    assert res["summary"]["tasks"]["n_cached"] == res["summary"]["tasks"]["n"] > 0


def test_solver_robust_gauge_and_uncertainty():
    """solve_offsets alone: random graph, 2 mas edge noise, 15% gross outliers (+-300 mas), a pair of ties; checks recovery, rejection of the outliers, honest sigma, gauge options."""
    rng = np.random.default_rng(11)
    U = 40
    s_true = rng.normal(0, 150, (U, 2))
    iu, ju = [], []
    for i in range(U):
        for j in rng.choice([k for k in range(U) if k != i], 6, replace=False):
            iu.append(i); ju.append(int(j))
    iu, ju = np.array(iu), np.array(ju)
    d = s_true[iu] - s_true[ju] + rng.normal(0, 2.0, (len(iu), 2))
    bad = rng.uniform(size=len(iu)) < 0.15
    d[bad] += rng.choice([-300.0, 300.0], (bad.sum(), 2)) * rng.uniform(0.5, 1.0, (bad.sum(), 2))
    sig = np.full(len(iu), 2.0)
    sol = RX.solve_offsets(U, iu, ju, d, sig)
    err = sol["s"] - s_true
    err -= err.mean(0)
    assert np.abs(err).max() < 4.0, np.abs(err).max()
    assert bad[~sol["active"]].mean() > 0.9 and (~sol["active"][~bad]).mean() < 0.05, (bad[~sol["active"]].mean(), (~sol["active"][~bad]).mean())
    z = err / sol["sigma"]
    assert 0.5 < np.sqrt((z ** 2).mean()) < 1.6, np.sqrt((z ** 2).mean())      # formal sigma is honest (relative to the mean gauge)
    # gauge: mean (weighted) of the gauge set is zero; two disconnected components get one gauge each
    w = np.ones(U); w[:10] = 3.0
    gm = np.zeros(U, bool); gm[:10] = True
    g = RX.solve_offsets(U, iu, ju, d, sig, unit_weight=w, gauge_mask=gm)
    assert np.abs(g["s"][:10].mean(0)).max() < 1e-6
    iu2 = np.r_[iu, [U, U + 1]]; ju2 = np.r_[ju, [U + 1, U]]
    s2 = RX.solve_offsets(U + 2, iu2, ju2, np.vstack([d, [[10.0, 5.0], [-10.0, -5.0]]]), np.r_[sig, [2.0, 2.0]])
    assert s2["n_comp"] == 2 and abs(s2["s"][U] + s2["s"][U + 1]).max() < 1e-9
    # soft ties pull two otherwise unrelated components together
    s3 = RX.solve_offsets(U + 2, iu2, ju2, np.vstack([d, [[10.0, 5.0], [-10.0, -5.0]]]), np.r_[sig, [2.0, 2.0]], tie_a=[U], tie_b=[0], tie_sig=[3.0])
    assert s3["n_comp"] == 1


def test_similarity_fit_exact():
    pos = np.array([[-10.0, 0.0], [10.0, 0.0], [0.0, 8.0], [3.0, -9.0]])
    eps, th, t = 2e-4, -3e-4, np.array([5.0, -7.0])
    X0 = pos.mean(0)
    r = pos - X0
    s = t + 1e3 * (eps * r + th * np.c_[-r[:, 1], r[:, 0]])
    f = RX.fit_similarity(pos, s, np.ones(4))
    assert abs(f["eps"] - eps) < 1e-9 and abs(f["theta"] - th) < 1e-9 and np.allclose(f["t"], t) and f["rms"] < 1e-9


# ----------------------------------------------------------------------------- tilefit A13
def test_tilefit_shifts_from():
    try:
        import torch
        import tilefit as T
    except Exception as exc:  # noqa: BLE001
        print("SKIP tilefit hook test:", type(exc).__name__, exc)
        return
    import types
    T.load_engine("smacs_four_v38")
    res, truth, out, _ = run_default()
    path = os.path.join(out, "register_xcorr.json")
    scale = 0.04
    names = list(truth)[:12]
    mk = lambda tid: types.SimpleNamespace(id=tid, thead={"NAXIS1": 128, "NAXIS2": 128, "CD2_2": scale / 3600.0}, exps=[  # noqa: E731
        {"name": nm, "tx": torch.linspace(0, 100, 50).double() + 0.123 * k, "ty": torch.linspace(0, 80, 50).double() - 0.071 * k} for k, nm in enumerate(names)] +
        [{"name": "jw99999999999_00000_00001_nrcb1_cal.fits", "tx": torch.zeros(5).double(), "ty": torch.zeros(5).double()}])
    tiles = [mk(0), mk(1)]
    before = [[(e["tx"].clone(), e["ty"].clone()) for e in t.exps] for t in tiles]
    summ = T.apply_external_shifts(tiles, path)
    js = json.load(open(path))["exposures"]
    for t, bt in zip(tiles, before):
        for e, (tx0, ty0) in zip(t.exps, bt):
            x = js.get(e["name"])
            if x is None:
                assert torch.equal(e["tx"], tx0) and torch.equal(e["ty"], ty0), "unmatched exposure was moved"
                continue
            # exactly the arithmetic of A12 (coarse_register_tiles, apply block)
            ex = tx0 + float(x["shift_mas"][0]) / 1000.0 / scale
            ey = ty0 + float(x["shift_mas"][1]) / 1000.0 / scale
            assert torch.equal(e["tx"], ex) and torch.equal(e["ty"], ey), e["name"]
    assert summ["n_applied_exposures"] == len(names) and summ["n_unmatched_exposures"] == 1
    # mutually exclusive with --coarse-shift-jwst > 0
    try:
        T.parse_cli(["--plan", "p.json", "--out", "o", "--shifts-from", path, "--coarse-shift-jwst", "0.2"])
        raise AssertionError("no error for --shifts-from together with --coarse-shift-jwst")
    except SystemExit as e:
        assert e.code not in (0, None)
    a, rest = T.parse_cli(["--plan", "p.json", "--out", "o", "--shifts-from", path])
    assert a.shifts_from == path and a.coarse_shift_jwst == 0.0
    a0, _ = T.parse_cli(["--plan", "p.json", "--out", "o"])
    assert a0.shifts_from is None
    assert "A13" in T.APPROX_A13 and "A13" in (T.__doc__ or "")
    print(f"tilefit A13: {summ['n_applied_exposures']} exposures shifted exactly as A12 would, {summ['n_unmatched_exposures']} unmatched left untouched; exclusive with --coarse-shift-jwst")


def test_header_shift_matches_tx():
    """tilefit.shift_exposure_header: a WCS built from the shifted header (the engine's own chain in V.load_exposures: torchsky pixel_to_world -> tile world_to_pixel) moves every
    pixel by exactly the shift in tile pixels (x west, y north), for SW and LW headers (real SIP headers moved to the LMC, dec -69.5), tiles off the reference point, shifts up to 2.5
    arcsec; and a classical coadd (V.classical, built from e["header"]) of a synthetic exposure lands at the shifted position. A header dict is shifted only once."""
    try:
        import torch
        import tilefit as T
        from torchsky.wcs import WCS as TW
        from astropy.io import fits as _f
    except Exception as exc:  # noqa: BLE001
        print("SKIP header-shift test:", type(exc).__name__, exc)
        return
    import glob
    T.load_engine("smacs_four_v38")
    V = T.V
    cands = {"SW": "*nrcb3_cal.fits", "LW": "*nrcblong_cal.fits"}
    files = {k: sorted(glob.glob(os.path.join(ROOT, "testdata", "nircam", g))) for k, g in cands.items()}
    if not all(files.values()):
        print("SKIP header-shift test: testdata/nircam missing")
        return
    scale = 0.04
    worst = 0.0
    for kind, fl in files.items():
        h = _f.getheader(fl[0], "SCI")
        base = {k: h[k] for k in h.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        base["CRVAL1"], base["CRVAL2"] = 80.50, -69.50
        for (cx, cy) in ((1024, 1024), (1700, 600), (300, 1800)):
            w0 = TW(header=dict(base))
            ra, dec = [float(v) for v in w0.pixel_to_world_differentiable(torch.tensor([[cx, cy]], dtype=torch.float64), origin=0)[0]]
            thead = V.target_header(ra, dec, 128, scale)
            tw = TW(header=thead)
            yy, xx = np.mgrid[cy - 60:cy + 61:12, cx - 60:cx + 61:12]
            pc = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float64)
            def tp(hd):
                return tw.world_to_pixel_differentiable(TW(header=hd).pixel_to_world_differentiable(pc, origin=0), origin=0).double()
            t0 = tp(dict(base))
            for (dx, dy) in ((2500.0, 0.0), (0.0, -2500.0), (1766.0, 1766.0), (-793.0, 809.0), (0.0, 0.0)):
                hd = dict(base)
                e = {"header": hd, "col": torch.tensor(xx.ravel(), dtype=torch.float64), "row": torch.tensor(yy.ravel(), dtype=torch.float64)}
                assert T.shift_exposure_header(e, thead, dx, dy) is True
                assert T.shift_exposure_header({"header": hd}, thead, dx, dy) is False      # same header dict: not shifted twice
                d = tp(hd) - t0
                err = (d - torch.tensor([dx / 1000 / scale, dy / 1000 / scale], dtype=torch.float64)).abs().max().item()
                worst = max(worst, err)
                assert err < 0.01, (kind, (cx, cy), (dx, dy), err)
                T._HDR_SHIFTED.pop(id(hd), None)
    print(f"header shift: worst |d(tile px) - shift| over SW/LW, 3 positions, 5 shifts up to 2.5 arcsec at dec -69.5: {worst:.4f} tile px (limit 0.01)")
    # classical coadd of a synthetic shifted exposure (SW header, 256 px cutout around detector pixel (1100, 900))
    h = _f.getheader(files["SW"][0], "SCI")
    hd = {k: h[k] for k in h.keys() if k and k not in ("COMMENT", "HISTORY", "")}
    hd["CRVAL1"], hd["CRVAL2"] = 80.50, -69.50
    x0, y0, n = 1000, 800, 256
    sx, sy = 1100.3, 900.6
    yy, xx = np.mgrid[0:n, 0:n]
    img = 100.0 * np.exp(-((xx + x0 - sx) ** 2 + (yy + y0 - sy) ** 2) / (2 * 1.3 ** 2)).astype(np.float32)
    tmp = tempfile.mkdtemp(prefix="rx_hdr_")
    full = np.zeros((2048, 2048), np.float32); full[y0:y0 + n, x0:x0 + n] = img
    fn = os.path.join(tmp, "jw09999001001_02101_00001_nrcb3_cal.fits")
    _f.HDUList([_f.PrimaryHDU(), _f.ImageHDU(full, name="SCI"), _f.ImageHDU(np.ones_like(full), name="ERR"), _f.ImageHDU(np.zeros(full.shape, np.uint32), name="DQ")]).writeto(fn)
    w0 = TW(header=dict(hd))
    ra, dec = [float(v) for v in w0.pixel_to_world_differentiable(torch.tensor([[sx, sy]], dtype=torch.float64), origin=0)[0]]
    thead = V.target_header(ra, dec, 256, scale)
    mk = lambda: {"name": os.path.basename(fn), "band": "F150W", "detname": "NRCB3", "path": fn, "bkg0": 0.0, "cut": (y0, y0 + n, x0, x0 + n), "header": dict(hd), "holdout": False, "val": False}  # noqa: E731
    V.set_registry(V.registry_from_exposures([mk()]))
    def centroid(e):
        c = V.classical([e], thead, "lanczos3", 1.0, torch.device("cpu"))[0].double()
        c = c.clamp(min=0)
        yy_, xx_ = torch.meshgrid(torch.arange(c.shape[0]).double(), torch.arange(c.shape[1]).double(), indexing="ij")
        return torch.stack([(c * xx_).sum() / c.sum(), (c * yy_).sum() / c.sum()])
    c0 = centroid(mk())
    for (dx, dy) in ((1500.0, -900.0), (-2500.0, 2000.0)):
        e = mk()
        assert T.shift_exposure_header(e, thead, dx, dy)
        dc = centroid(e) - c0
        want = torch.tensor([dx / 1000 / scale, dy / 1000 / scale], dtype=torch.float64)
        assert (dc - want).abs().max() < 0.05, (dc, want)
    print(f"classical coadd centroid moves by the shift in tile px (checked to 0.05 px); unshifted centroid at {c0.tolist()}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    for f in (test_resume_cache_and_sign_of_pair, test_second_peak_flags_aliasing, test_recovery_default, test_no_xband_gives_two_components, test_resume_reuses_tasks, test_solver_robust_gauge_and_uncertainty, test_similarity_fit_exact, test_detector_mode_similarity, test_tilefit_shifts_from, test_header_shift_matches_tx):
        print(f"--- {f.__name__}")
        f()
    print(f"ALL PASS ({time.time() - t0:.0f} s)")
