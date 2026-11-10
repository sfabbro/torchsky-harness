"""Plain-python tests for scripts/hst_io.py (run: python3 tests/test_hst_io.py)."""
import logging
import os
import sys

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import hst_io  # noqa: E402

logging.basicConfig(level=logging.INFO, format="  [log] %(message)s")
S = os.path.join(HERE, "..", "hst", "samples")
IR = os.path.join(S, "iczgzbmgq_flt.fits")
ACS = os.path.join(S, "jczgz3k8q_flc.fits")
WFPC2 = os.path.join(S, "u9yt0g01m_c1m.fits")
JW = os.path.join(S, "jw02736001001_02103_00004_nrcblong_cal.fits")
RA0, DEC0, SIZE, SCALE, PAD = 110.8268, -73.4544, 512, 0.04, 24


def target_header(ra, dec, size, scale_arcsec):  # identical to smacs_four_v28.target_header
    s = scale_arcsec / 3600.0
    return {"NAXIS": 2, "NAXIS1": size, "NAXIS2": size, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg", "CUNIT2": "deg", "CRVAL1": ra, "CRVAL2": dec,
            "CRPIX1": size / 2 + 0.5, "CRPIX2": size / 2 + 0.5,
            "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s}


thead = target_header(RA0, DEC0, SIZE, SCALE)
tw = WCS(fits.Header(thead))
dev = torch.device("cpu")
exps = hst_io.load_hst_exposures([IR, ACS, WFPC2], thead, dev, pad=PAD)
KEYS = ["name", "band", "tx", "ty", "d", "ivar", "row", "col", "dx", "dy", "bkg0", "path", "header", "cut", "noise_scale",
        "detname", "instrument", "filter", "mjd", "exptime", "pixar_sr_nominal", "area_ratio", "units_note"]

print("\n=== (a) loading, counts, ranges")
byfile = {}
for e in exps:
    for k in KEYS:
        assert k in e, k
    n = e["tx"].numel()
    assert all(e[k].numel() == n for k in ("ty", "d", "ivar", "row", "col", "dx", "dy", "area_ratio"))
    assert e["tx"].dtype == torch.float32 and e["ty"].dtype == torch.float32
    assert torch.isfinite(e["d"]).all() and (e["ivar"] > 0).all()
    assert e["tx"].min() > -PAD and e["tx"].max() < SIZE + PAD and e["ty"].min() > -PAD and e["ty"].max() < SIZE + PAD
    assert e["dx"].min() >= -1 and e["dx"].max() <= 1 and e["dy"].min() >= -1 and e["dy"].max() <= 1
    inside = ((e["tx"] >= 0) & (e["tx"] <= SIZE) & (e["ty"] >= 0) & (e["ty"] <= SIZE)).float().mean().item()
    print(f"{e['name']:18s} {e['detname']:10s} {e['instrument']:8s} {e['filter']} band={e['band']:11s} n={n:7d} "
          f"tx[{e['tx'].min():.1f},{e['tx'].max():.1f}] ty[{e['ty'].min():.1f},{e['ty'].max():.1f}] "
          f"frac inside 512 grid={inside:.3f} cut={e['cut']} mjd={e['mjd']:.4f} exptime={e['exptime']:.1f}")
    byfile.setdefault(os.path.basename(e["path"]), []).append(e)
assert len(byfile["iczgzbmgq_flt.fits"]) == 1
assert "u9yt0g01m_c1m.fits" not in byfile  # WFPC2 skipped (reason logged above)
assert {e["name"] for e in byfile["iczgzbmgq_flt.fits"]} == {"iczgzbmgq_sci1"}
assert all(e["name"].startswith("jczgz3k8q_sci") for e in byfile["jczgz3k8q_flc.fits"])

# independent check: every chip that was NOT returned has < 2000 kept-eligible pixels inside the padded grid
with fits.open(ACS) as h:
    for ext in (1, 2):
        w = hst_io.hst_wcs(h, ext)
        yy, xx = np.mgrid[0:2048:4, 0:4096:4]
        ra, de = w.all_pix2world(xx.ravel() * 1.0, yy.ravel() * 1.0, 0)
        tx, ty = tw.all_world2pix(ra, de, 0)
        n_in = int(((tx > -PAD) & (tx < SIZE + PAD) & (ty > -PAD) & (ty < SIZE + PAD)).sum()) * 16
        returned = any(e["name"].endswith(f"_sci{ext}") for e in byfile["jczgz3k8q_flc.fits"])
        print(f"ACS sci{ext}: ~{n_in} pixels inside padded grid (sparse count x16); returned={returned}")
        assert returned == (n_in >= 2000)
print("(a) OK; chips returned:", {k: len(v) for k, v in byfile.items()}, " (WFPC2 file: skipped)")

print("\n=== (b) round trip target-grid coords -> detector pixels")
rng = np.random.default_rng(1)
for e in exps:
    with fits.open(e["path"]) as h:
        w = hst_io.hst_wcs(h, e["ext"])
    i = rng.choice(e["tx"].numel(), 3000, replace=False)
    tx, ty = e["tx"].numpy()[i].astype(float), e["ty"].numpy()[i].astype(float)
    ra, de = tw.all_pix2world(tx, ty, 0)
    xb, yb = w.all_world2pix(ra, de, 0, tolerance=1e-7, maxiter=100, quiet=True)
    c, r = e["col"].numpy()[i], e["row"].numpy()[i]
    scale_mas = np.sqrt(hst_io.pixel_area_arcsec2(w, c.astype(float), r.astype(float))) * 1e3
    err_mas = np.hypot(xb - c, yb - r) * scale_mas
    print(f"{e['name']:18s} round-trip error: max {err_mas.max():.4f} mas, median {np.median(err_mas):.5f} mas")
    assert err_mas.max() < 1.0
print("(b) OK")

print("\n=== (c) lookup-table (D2IM + NPOL/CPDIS) contribution, full WCS vs lookups removed (all_pix2world both)")
for fpath, exts in ((IR, [1]), (ACS, [1, 2])):
    with fits.open(fpath) as h:
        for ext in exts:
            w = hst_io.hst_wcs(h, ext)
            ny, nx = h[("SCI", ext)].data.shape
            full = hst_io.lookup_table_contribution(w, (ny, nx), step=16)
            ee = [e for e in exps if e["path"] == fpath and e["ext"] == ext]
            crop = ""
            if ee:
                px = np.stack([ee[0]["col"].numpy(), ee[0]["row"].numpy()], 1)[::50]
                cc = hst_io.lookup_table_contribution(w, None, pixels=px)
                crop = f" | kept pixels (target crop): max {cc['max_mas']:.2f} median {cc['median_mas']:.2f} mas"
            print(f"{os.path.basename(fpath)} sci{ext}: lookups present={full['has_lookup']}; full chip: max {full['max_mas']:.2f} "
                  f"median {full['median_mas']:.2f} rms {full['rms_mas']:.2f} mas{crop}")
            if fpath == ACS:
                assert full["has_lookup"] and full["max_mas"] > 1.0
            else:
                assert not full["has_lookup"]
print("(c) OK")

print("\n=== (d) units: median sky and conversion factors")
for e in exps:
    print(f"{e['name']:18s} median d = {e['bkg0']:.4f} MJy/sr | to_mjysr={e['to_mjysr']:.6g} (MJy/sr per raw unit incl. /EXPTIME if any) "
          f"PHOTFNU={e['photfnu_jy']:.6g} Jy s/e- PHOTFLAM={e['photflam']} PHOTPLAM={e['photplam']:.2f} A "
          f"BUNIT={e['header'].get('BUNIT')} EXPTIME={e['exptime']} Omega_nom={e['pixar_sr_nominal']:.6g} sr")
    print("    ", e["units_note"])
    assert 1e-3 < e["bkg0"] < 5.0, "sky level outside plausible range"
    # independent recomputation of the conversion from raw header values
    h0 = fits.getheader(e["path"], 0)
    raw = fits.getdata(e["path"], ("SCI", e["ext"]))[e["row"][:5], e["col"][:5]].astype(float)
    nu = e["photfnu_jy"]
    scale = e["header"]["IDCSCALE"]
    expect = raw * (1 / e["exptime"] if e["header"]["BUNIT"] == "ELECTRONS" else 1.0) * nu * 1e-6 / (scale ** 2 * (np.pi / 648000) ** 2)
    assert np.allclose(expect, e["d"][:5].numpy(), rtol=2e-6), (expect, e["d"][:5])
print("(d) OK")

print("\n=== (e) area_ratio (true/nominal pixel area, nominal = IDCSCALE^2)")
for e in exps:
    a = e["area_ratio"]
    print(f"{e['name']:18s} min {a.min():.4f} max {a.max():.4f} mean {a.mean():.4f} | at reference pixel {e['area_ratio_ref']:.4f}")
    assert 0.8 < a.min() and a.max() < 1.2

print("\n=== DQ policy: fraction of pixels kept (whole chip) and kept in target crop")
for fpath, exts in ((IR, [1]), (ACS, [1, 2])):
    for ext in exts:
        fr = hst_io.dq_keep_fractions(fpath, ext, ("default", "strict", 80))
        s = "; ".join(f"{k}(good_bits={v['good_bits']}): dq-only {v['dq_only']:.4f}, +finite/err>0 {v['dq_and_finite']:.4f}" for k, v in fr.items())
        print(f"{os.path.basename(fpath)} sci{ext}: {s}")
for pol in ("default", "strict"):
    ee = hst_io.load_hst_exposures([IR, ACS], thead, dev, pad=PAD, dq_policy=pol)
    print(f"  crop, policy={pol}: " + ", ".join(f"{e['name']}={e['tx'].numel()}" for e in ee))
assert hst_io.load_hst_exposures([IR, ACS], thead, dev, pad=PAD, dq_policy="strict")[0]["tx"].numel() <= exps[0]["tx"].numel()

print("\n=== hst_band_info")
for a in (("WFC3/IR", "F160W"), ("ACS/WFC", "F814W"), ("WFC3/UVIS", "F606W")):
    print(a, hst_io.hst_band_info(*a))

print("\n=== JWST NIRCam: PIXAR_SR / AREA / Jacobian")
if os.path.exists(JW):
    with fits.open(JW) as h:
        sh = h[("SCI", 1)].header
        print(f"BUNIT={sh['BUNIT']} PIXAR_SR={sh['PIXAR_SR']:.6e} PIXAR_A2={sh['PIXAR_A2']:.6e} PHOTMJSR={sh['PHOTMJSR']}")
        a = h["AREA"].data
        print(f"AREA ext: min {a.min():.4f} max {a.max():.4f} mean {a.mean():.4f}")
    yy, xx = np.mgrid[0:2048:16, 0:2048:16]
    px = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    ra_ = hst_io.jwst_area_ratio(JW, px, "area")
    rj = hst_io.jwst_area_ratio(JW, px, "jacobian")
    rh = hst_io.jwst_area_ratio(fits.getheader(JW, ("SCI", 1)), px)  # header-only -> jacobian
    assert np.allclose(rj, rh)
    d = rj - ra_
    print(f"area vs jacobian over 128x128 lattice: AREA [{ra_.min():.4f},{ra_.max():.4f}] mean {ra_.mean():.4f}; "
          f"Jacobian/PIXAR_A2 [{rj.min():.4f},{rj.max():.4f}] mean {rj.mean():.4f}; diff mean {d.mean():+.5f} rms {d.std():.5f} "
          f"max|diff| {np.abs(d).max():.5f}; corr {np.corrcoef(ra_, rj)[0, 1]:.4f}")
    assert np.abs(d).max() < 0.02
else:
    print("JWST sample missing, skipped")
print("\nALL TESTS PASSED")
