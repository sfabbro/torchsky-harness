"""Tests for scripts/smacs_four_v29.py (JWST + HST in one forward model). Run: python3 tests/test_v29.py
(a) registry for a NIRCam-only list == the v28 constants      (b) registry for NIRCam + WFC3/IR + ACS (printed)
(c) ChromaticPSF with mixed bands: finite, unit sum, core width scales with lambda/D (printed)
(d) HST splat coadd of the two sample files on a 512 px, 0.04" grid       (e) test selection with a mixed synthetic list
(f) end-to-end --quick CPU run (needs torchsky; skipped otherwise)
plus: HST file selection, load_bandpasses arguments, PSF layout/values vs v28, rgb groups, per-instrument bootstrap."""
import json
import os
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hst_io  # noqa: E402
import smacs_four_v28 as S28  # noqa: E402
import smacs_four_v29 as S  # noqa: E402

SAMPLES = os.path.join(ROOT, "hst", "samples")
IR = os.path.join(SAMPLES, "iczgzbmgq_flt.fits")
ACS = os.path.join(SAMPLES, "jczgz3k8q_flc.fits")
MANIFEST = os.path.join(ROOT, "hst", "hst_imaging_unique.csv")
NIRCAM_DIR = os.path.join(ROOT, "testdata", "nircam")
HST_EXTRA = os.path.join(ROOT, "testdata", "hst_extra")
CPU = torch.device("cpu")
RA0, DEC0 = 110.8268, -73.4544
_cache = {}


def hst_exposures(size):
    if size not in _cache:
        th = S.target_header(RA0, DEC0, size, 0.04)
        _cache[size] = (th, hst_io.load_hst_exposures([IR, ACS], th, CPU, pad=24))
    return _cache[size]


def stub_nircam():
    return [{"band": b, "name": f"jw_{b}"} for b in S.NIRCAM_BANDS]


# ---------------------------------------------------------------------------------------------------- (a)
def test_a_registry_nircam_equals_v28():
    reg = S.build_registry(S28.BANDS)
    assert reg.bands == S28.BANDS
    assert reg.lam_um == S28.LAM_UM and list(reg.lam_um) == list(S28.LAM_UM)
    assert reg.fwhm_arcsec == S28.FWHM_ARCSEC
    assert reg.chan_id == S28.CHANNEL, (reg.chan_id, S28.CHANNEL)
    assert reg.channels == ["NIRCAM_SW", "NIRCAM_LW"]
    assert reg.svo_id == {b: f"JWST/NIRCam.{b}" for b in S28.BANDS}
    assert all(reg.diameter_m[b] == 6.5 for b in reg.bands) and all(reg.kernel_px[b] == 31 for b in reg.bands)
    assert not reg.multi and reg.instruments == ["NIRCam"]
    assert S.REG.bands == S28.BANDS  # the module default is the same registry
    reg2 = S.registry_from_exposures(stub_nircam())
    assert reg2.bands == reg.bands and reg2.lam_um == reg.lam_um and reg2.chan_id == reg.chan_id
    # a subset keeps the canonical order and the two NIRCam channels (v28 one-hot width)
    sub = S.build_registry(["F444W", "F150W"])
    assert sub.bands == ["F150W", "F444W"] and sub.channels == ["NIRCAM_SW", "NIRCAM_LW"] and sub.chan_id == {"F150W": 0, "F444W": 1}
    print("(a) registry(NIRCam six) == v28 BANDS / LAM_UM / FWHM_ARCSEC / CHANNEL exactly; channels", reg.channels)


# ---------------------------------------------------------------------------------------------------- (b)
def print_registry(reg):
    print(f"  {'band':14s} {'instrument':10s} {'channel':10s} id {'lam_um':>8s} {'D_m':>4s} {'fwhm"':>7s} {'kern':>4s}  svo_id")
    for r in reg.summary():
        print(f"  {r['band']:14s} {r['instrument']:10s} {r['channel']:10s} {r['chan_id']:2d} {r['lam_um']:8.4f} {r['diameter_m']:4.1f} "
              f"{r['fwhm_arcsec']:7.4f} {r['kernel_px']:4d}  {r['svo_id']}")


def test_b_registry_mixed():
    _, hexps = hst_exposures(512)
    reg = S.registry_from_exposures(stub_nircam() + hexps)
    print("(b) registry NIRCam + WFC3/IR + ACS (from the loaded sample chips):")
    print_registry(reg)
    assert reg.bands == S.NIRCAM_BANDS + ["ACS_F814W", "WFC3IR_F160W"]  # HST by wavelength
    assert reg.channels == ["NIRCAM_SW", "NIRCAM_LW", "WFC3_IR", "ACS_WFC"]
    assert reg.channel["ACS_F814W"] == "ACS_WFC" and reg.channel["WFC3IR_F160W"] == "WFC3_IR"
    assert abs(reg.lam_um["WFC3IR_F160W"] - 1.5369) < 1e-3 and abs(reg.lam_um["ACS_F814W"] - 0.8046) < 1e-3  # PHOTPLAM / 1e4
    assert reg.diameter_m["ACS_F814W"] == 2.4 and reg.diameter_m["F150W"] == 6.5
    assert 0.14 < reg.fwhm_arcsec["WFC3IR_F160W"] < 0.17 and 0.06 < reg.fwhm_arcsec["ACS_F814W"] < 0.10
    assert reg.svo_id["WFC3IR_F160W"] == "HST/WFC3_IR.F160W" and reg.svo_id["ACS_F814W"] == "HST/ACS_WFC.F814W"
    assert reg.multi and reg.instruments == ["NIRCam", "ACS/WFC", "WFC3/IR"]
    assert reg.is_nircam["F090W"] and reg.is_hst["ACS_F814W"]
    S.set_registry(reg)
    try:
        assert S.uses_stripes("F090W") and S.uses_stripes("F444W") and not S.uses_stripes("ACS_F814W") and not S.uses_stripes("WFC3IR_F160W")
    finally:
        S.set_registry(S.build_registry(S.NIRCAM_BANDS))
    # NIRCam rows unchanged by adding HST
    for b in S28.BANDS:
        assert reg.lam_um[b] == S28.LAM_UM[b] and reg.fwhm_arcsec[b] == S28.FWHM_ARCSEC[b] and reg.chan_id[b] == S28.CHANNEL[b]
    # a WFC3/UVIS band from header-style info (no pixels needed) is sorted in by wavelength and gets its own channel
    reg3 = S.build_registry(S.NIRCAM_BANDS, [{"band": "UVIS_F606W", "instrument": "WFC3/UVIS", "filter": "F606W", "photplam": 5887.0, "svo_id": None},
                                             {"band": "ACS_F435W", "instrument": "ACS/WFC", "filter": "F435W", "photplam": 4329.0}])
    assert reg3.bands[6:] == ["ACS_F435W", "UVIS_F606W"] and reg3.channels[-1] == "WFC3_UVIS" and reg3.svo_id["UVIS_F606W"] == "HST/WFC3_UVIS2.F606W"
    # selection helpers: detection bands by name, rgb groups
    assert S.detection_bands(reg) == [0, 1, 2]
    assert S.detection_bands(S.build_registry(["F277W", "F444W"], [])) == [0, 1]  # no SW band: the shortest bands
    print("    detection bands:", [reg.bands[k] for k in S.detection_bands(reg)])


def test_b2_load_bandpasses_arguments():
    """load_bandpasses must ask SVO for the registry's filter id with the right facility / instrument (fake SVO, no network)."""
    calls = []

    def fake(**kw):
        calls.append(kw)
        wl = np.linspace(kw["_lo"] if "_lo" in kw else 5000.0, 6000.0, 50)
        return {"transmission_wavelength": wl, "transmission_throughput": np.ones_like(wl), "svo_filter_id": kw["filter_id"]}

    pkg, sub = types.ModuleType("torchsky"), types.ModuleType("torchsky.filter_profiles")
    sub.bandpass_from_svo = fake
    saved = {k: sys.modules.get(k) for k in ("torchsky", "torchsky.filter_profiles")}
    sys.modules["torchsky"], sys.modules["torchsky.filter_profiles"] = pkg, sub
    try:
        reg = S.build_registry(["F090W"], [{"band": "ACS_F814W", "instrument": "ACS/WFC", "filter": "F814W", "photplam": 8045.0},
                                            {"band": "WFC3IR_F160W", "instrument": "WFC3/IR", "filter": "F160W", "photplam": 15369.0}])
        lam, w, _ = S.load_bandpasses(reg.bands, 4, None, reg=reg)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    got = [(c["facility"], c["instrument"], c["filter_id"], c["band_name"], c["prefer"]) for c in calls]
    assert got == [("JWST", "NIRCam", "JWST/NIRCam.F090W", "F090W", "any"), ("HST", "ACS_WFC", "HST/ACS_WFC.F814W", "F814W", "any"),
                   ("HST", "WFC3_IR", "HST/WFC3_IR.F160W", "F160W", "any")], got
    assert lam.shape == (3, 4) and torch.allclose(w.sum(1), torch.ones(3))
    print("(b2) load_bandpasses SVO calls:", got)


# ---------------------------------------------------------------------------------------------------- (c)
def core_fwhm_arcsec(B0, scale):
    n = B0.shape[-1]
    r = torch.arange(n, dtype=torch.float64) - n // 2
    w = B0.double()
    tot = w.sum((-2, -1))
    cx = (w.sum(-2) * r).sum(-1) / tot
    cy = (w.sum(-1) * r).sum(-1) / tot
    vx = (w.sum(-2) * r ** 2).sum(-1) / tot - cx ** 2
    vy = (w.sum(-1) * r ** 2).sum(-1) / tot - cy ** 2
    return 2.3548 * torch.sqrt(0.5 * (vx + vy)) * scale


def test_c_chromatic_psf():
    reg = S.build_registry(S.NIRCAM_BANDS, [{"band": "ACS_F814W", "instrument": "ACS/WFC", "filter": "F814W", "photplam": 8045.5},
                                            {"band": "WFC3IR_F160W", "instrument": "WFC3/IR", "filter": "F160W", "photplam": 15369.2},
                                            {"band": "UVIS_F606W", "instrument": "WFC3/UVIS", "filter": "F606W", "photplam": 5887.0},
                                            {"band": "ACS_F435W", "instrument": "ACS/WFC", "filter": "F435W", "photplam": 4329.0}])
    bands = reg.bands
    scale = 0.02  # fine pixels so that the narrowest cores (0.03") are resolved
    torch.manual_seed(0)
    psf = S.ChromaticPSF(bands, 5, scale, nk=4, n=31, reg=reg)
    # lambda/D per band from the registry, not from 6.5 m
    lod = np.array([0.0317 * reg.lam_um[b] * 6.5 / reg.diameter_m[b] for b in bands])
    assert np.allclose(psf.pix_per_lod.numpy(), scale / lod, rtol=1e-5)
    assert psf.chan.shape == (len(bands), 5) and psf.nchan == 5
    B = psf.basis()
    assert torch.isfinite(B).all()
    assert torch.allclose(B[:, 0].sum((-2, -1)), torch.ones(len(bands)), atol=1e-5), "core not unit sum"
    assert B[:, 1:].sum((-2, -1)).abs().max() < 1e-5, "residual kernels not zero sum"
    # basis_lam at the registry's own wavelengths / channels == basis()
    lam = torch.tensor([reg.lam_um[b] for b in bands])
    assert torch.allclose(psf.basis_lam(lam, psf.chan_idx), B, atol=1e-6)
    # one network serves all instruments: warm start on the nominal Gaussians, then check the core widths
    sig0 = torch.tensor([reg.fwhm_arcsec[b] / scale / 2.3548 for b in bands])
    t0 = time.time()
    loss = S.warm_psf(psf, S.gauss_kernel(sig0, 31, CPU), 2500)
    B = psf.basis().detach()
    assert torch.isfinite(B).all() and torch.allclose(B[:, 0].sum((-2, -1)), torch.ones(len(bands)), atol=1e-5)
    fw = core_fwhm_arcsec(B[:, 0], scale)
    nom = torch.tensor([reg.fwhm_arcsec[b] for b in bands], dtype=torch.float64)
    ld = torch.tensor(lod, dtype=torch.float64)
    print(f"(c) ChromaticPSF, {len(bands)} bands, {psf.nchan} channels, scale {scale}\"/px, warm start 2500 steps ({time.time() - t0:.0f} s, loss {loss:.2e})")
    print(f"    {'band':14s} {'lam_um':>7s} {'D_m':>4s} {'lam/D"':>7s} {'nominal':>8s} {'core fwhm"':>10s} {'fwhm/(lam/D)':>12s}")
    for k, b in enumerate(bands):
        print(f"    {b:14s} {reg.lam_um[b]:7.3f} {reg.diameter_m[b]:4.1f} {lod[k]:7.4f} {nom[k]:8.4f} {fw[k]:10.4f} {fw[k] / lod[k]:12.3f}")
    rel = ((fw - nom) / nom).abs().max()
    corr = float(np.corrcoef(fw.numpy(), ld.numpy())[0, 1])
    print(f"    max |core/nominal - 1| = {rel:.3f}; corr(core fwhm, lambda/D) = {corr:.4f}")
    assert rel < 0.25 and corr > 0.95
    # the weights (spatial mixing) also run for every band and detector
    w = psf.weights(7, 3, torch.linspace(-1, 1, 5), torch.zeros(5))
    assert w.shape == (5, 4) and torch.isfinite(w).all()
    # NIRCam only: same state-dict layout and the same numbers as v28 for the same seed
    torch.manual_seed(1)
    p28 = S28.ChromaticPSF(S28.BANDS, 3, 0.04, 4, n=31)
    torch.manual_seed(1)
    p29 = S.ChromaticPSF(S.NIRCAM_BANDS, 3, 0.04, 4, n=31, reg=S.build_registry(S.NIRCAM_BANDS))
    s28, s29 = p28.state_dict(), p29.state_dict()
    assert list(s28) == list(s29) and all(s28[k].shape == s29[k].shape for k in s28)
    assert all(torch.equal(s28[k], s29[k]) for k in s28), "NIRCam-only ChromaticPSF differs from v28"
    assert torch.equal(p28.basis(), p29.basis())
    px, py = torch.rand(4) * 2 - 1, torch.rand(4) * 2 - 1
    assert torch.equal(p28.weights(2, 1, px, py), p29.weights(2, 1, px, py))
    lam6 = torch.tensor([S28.LAM_UM[b] for b in S28.BANDS])
    ch6 = torch.tensor([S28.CHANNEL[b] for b in S28.BANDS])
    assert torch.equal(p28.basis_lam(lam6, ch6), p29.basis_lam(lam6, ch6))
    print("    NIRCam-only ChromaticPSF: state-dict keys/shapes/values, basis() and basis_lam() bit-identical to v28")


# ---------------------------------------------------------------------------------------------------- (d)
def test_d_splat_coadd():
    th, hexps = hst_exposures(512)
    assert abs(th["CD2_2"] * 3600 - 0.04) < 1e-9 and th["NAXIS1"] == 512
    for band in ("WFC3IR_F160W", "ACS_F814W"):
        es = [e for e in hexps if e["band"] == band]
        img, wsum = S.splat_coadd(es, th, CPU)
        cov = wsum > 0
        assert img.shape == (512, 512) and wsum.shape == (512, 512)
        assert torch.isfinite(img[cov]).all(), "non-finite where covered"
        assert torch.isnan(img[~cov]).all() and (wsum[~cov] == 0).all(), "must be NaN / zero weight where nothing landed"
        # nothing is covered outside the footprint of the loaded pixels (+ the splat radius)
        tx = torch.cat([e["tx"] for e in es]); ty = torch.cat([e["ty"] for e in es])
        h = max(1.0, float(np.sqrt(es[0]["pixar_sr_nominal"] / S.ARCSEC2_TO_SR)) / 0.04)
        yy, xx = torch.nonzero(cov, as_tuple=True)
        assert xx.min() >= tx.min() - h - 1 and xx.max() <= tx.max() + h + 1 and yy.min() >= ty.min() - h - 1 and yy.max() <= ty.max() + h + 1
        # compare with a plain bilinear splat (radius 1): holes for the undersampled detector
        img1, w1 = S.splat_coadd(es, th, CPU, radius=1.0)
        print(f"(d) {band}: {len(es)} chip(s), {sum(e['d'].numel() for e in es)} pixels; input pixel/target pixel = {h:.2f}; "
              f"covered fraction: tent(h={max(h, 1):.2f}) {float(cov.float().mean()):.3f} vs plain bilinear {float((w1 > 0).float().mean()):.3f}; "
              f"NaN outside: {int(torch.isnan(img[~cov]).sum())}/{int((~cov).sum())}")
        # constant surface brightness in -> constant out; linear ramp in x -> the pixel coordinate out (centre convention)
        ec = [dict(e, d=torch.full_like(e["d"], 7.0), bkg0=0.0) for e in es]
        ic, wc = S.splat_coadd(ec, th, CPU)
        assert (ic[wc > 0] - 7.0).abs().max() < 1e-4, "constant not preserved"
        er = [dict(e, d=e["tx"].clone(), bkg0=0.0) for e in es]
        ir_, wr = S.splat_coadd(er, th, CPU)
        inner = (wr > 0) & (S.F.max_pool2d((wr > 0).float()[None, None], 2 * int(np.ceil(h)) + 1, 1, int(np.ceil(h)))[0, 0] > 0) & (wr > 0.5 * wr[wr > 0].median())
        xs = torch.arange(512, dtype=torch.float32)[None, :].expand(512, 512)
        err = (ir_ - xs)[inner]
        print(f"    ramp test: median |out - x| = {err.abs().median():.3f} px, 95% {err.abs().quantile(0.95):.3f} px over {int(inner.sum())} well-weighted pixels")
        assert err.abs().median() < 0.35 * max(h, 1.0), "coordinate convention / weights off"
        if h > 1.5:
            assert float((w1 > 0).float().mean()) < 0.7 * float(cov.float().mean())


# ---------------------------------------------------------------------------------------------------- (e)
def mk(name, band, n):
    return {"name": name, "band": band, "d": torch.zeros(n)}


def synthetic_list(seed=0):
    rng = np.random.default_rng(seed)
    exps = []
    i = 0
    grp = {"F090W": "02101", "F277W": "02101", "F150W": "02103", "F356W": "02103", "F200W": "02105", "F444W": "02105"}
    for b in S.NIRCAM_BANDS:
        dets = ["nrcb1", "nrcb3"] if b in ("F090W", "F150W", "F200W") else ["nrcblong"]
        for vis in ("001", "002"):
            for ex in range(1, 5):
                for d in dets:
                    exps.append(mk(f"{i:03d}_jw02736001{vis}_{grp[b]}_{ex:05d}_{d}_o001_crf_rshift_tweakreg.fits", b, int(rng.integers(5000, 90000))))
                    i += 1
    ir_roots = ["iczgzbmgq", "iczgzbmaq", "iczgzbmbq", "iczgz4kjq", "iczgz4klq", "iczgz2nxq", "iczgz2nzq"]
    acs_roots = ["jczgz3k8q", "jczgz3k6q", "jczgz5k8q", "jczgz5k6q"]
    for r in ir_roots:
        exps.append(mk(f"{r}_sci1", "WFC3IR_F160W", int(rng.integers(5000, 90000))))
    for r in acs_roots:
        for c in (1, 2):
            exps.append(mk(f"{r}_sci{c}", "ACS_F814W", int(rng.integers(5000, 90000))))
    return exps


def held(exps):
    return sorted(e["name"] for e in exps if e["holdout"])


def test_e_selection():
    reg = S.build_registry(S.NIRCAM_BANDS, [{"band": "ACS_F814W", "instrument": "ACS/WFC", "filter": "F814W", "photplam": 8045.5},
                                            {"band": "WFC3IR_F160W", "instrument": "WFC3/IR", "filter": "F160W", "photplam": 15369.2}])
    S.set_registry(reg)
    try:
        # names: JWST parser unchanged, HST chips of one exposure share a dither, one visit shares a visit key
        for nm in ("034_jw02736001001_02101_00009_nrcb3_o001_crf_rshift_tweakreg.fits", "jw02736001002_02103_00001_nrcblong_cal.fits", "garbage.fits"):
            assert S.parse_exposure_name(nm) == S28.parse_exposure_name(nm) and S.dither_key(nm) == S28.dither_key(nm)
        assert S.dither_key("jczgz3k8q_sci1") == S.dither_key("jczgz3k8q_sci2") != S.dither_key("jczgz3k6q_sci1")
        assert S.dither_key("iczgzbmgq_sci1") != S.dither_key("iczgzbmaq_sci1")
        assert S.visit_key("jczgz3k8q_sci2") == S.visit_key("jczgz3k6q_sci1") != S.visit_key("jczgz5k8q_sci1")
        assert S.visit_key("jczgz3k8q_sci1") != S.visit_key("iczgz3k8q_sci1")
        assert S.hst_root("iczgzbmgq_sci1") == "iczgzbmgq" and S.hst_root("jw02736001001_02101_00009_nrcb3_cal.fits") is None
        base = synthetic_list()
        nir = [e for e in base if e["band"] in S.NIRCAM_BANDS]
        for ntest in (1, 2, 3):
            # NIRCam rows of a mixed list are selected exactly as v28 selects the NIRCam-only list
            a28 = [dict(e) for e in nir]
            sel28 = S28.select_exposures(a28, n_test=ntest, n_val=1)
            mixed = [dict(e) for e in base]
            sel = S.select_exposures(mixed, n_test=ntest, n_val=1)
            assert sel28["holdout"] == [n for n in sel["holdout"] if n in {e["name"] for e in nir}], ntest
            assert sel28["validation"] == [n for n in sel["validation"] if n in {e["name"] for e in nir}], ntest
            hs = [e for e in mixed if e["holdout"] and e["band"] not in S.NIRCAM_BANDS]
            ir = [e for e in hs if e["band"] == "WFC3IR_F160W"]
            acs = [e for e in hs if e["band"] == "ACS_F814W"]
            if ntest == 1:  # v21 rule: one detector image per band (one chip for ACS)
                assert len(ir) == 1 and len(acs) == 1
            else:  # v22 rule: ntest dithers = ntest exposures = all their chips
                # 7 IR exposures; 4 ACS exposures (2 chips each) of which at least 2 stay in training: min(ntest, 4 - 2)
                assert len(ir) == ntest and len(acs) == 2 * min(ntest, 2), (ntest, len(ir), len(acs))
                roots = {}
                for e in acs:
                    roots.setdefault(S.hst_root(e["name"]), []).append(e["name"])
                assert all(len(v) == 2 for v in roots.values())
                # chosen over visits: ACS has 2 visits, so 2 dithers come from different visits
                if ntest == 2:
                    assert len({S.visit_key(e["name"]) for e in acs}) == 2
            print(f"(e) n_test={ntest}: NIRCam == v28 selection; held out HST: " + ", ".join(sorted(e["name"] for e in hs)))
        # deterministic
        m1, m2 = [dict(e) for e in base], [dict(e) for e in base]
        S.select_exposures(m1, n_test=2); S.select_exposures(m2, n_test=2)
        assert held(m1) == held(m2)
        # --holdout-from together with --n-test: file names held out, bands without any get the rule
        prev = [dict(e) for e in nir]
        S28.select_exposures(prev, n_test=2)
        names = held(prev)
        m = [dict(e) for e in base]
        sel = S.select_exposures(m, n_test=2, holdout_names=names, fill_missing_bands=True)
        assert [n for n in held(m) if n in {e["name"] for e in nir}] == names, "file names changed"
        assert len([e for e in m if e["holdout"] and e["band"] == "WFC3IR_F160W"]) == 2 and len([e for e in m if e["holdout"] and e["band"] == "ACS_F814W"]) == 4
        assert any("no held-out exposure in the holdout file" in w for w in sel["warnings"]) and len(sel["warnings"]) == 2
        m = [dict(e) for e in base]  # without the fill (v28 behaviour of --holdout-from): only the file
        S.select_exposures(m, n_test=2, holdout_names=names)
        assert held(m) == sorted(names)
        # a band that has some held-out exposure in the file is left alone; rootname entries hold out all chips
        m = [dict(e) for e in base]
        S.select_exposures(m, n_test=2, holdout_names=names + ["jczgz3k8q", "iczgz4kjq_sci1"], fill_missing_bands=True)
        assert [e["name"] for e in m if e["holdout"] and e["band"] == "ACS_F814W"] == ["jczgz3k8q_sci1", "jczgz3k8q_sci2"]
        assert [e["name"] for e in m if e["holdout"] and e["band"] == "WFC3IR_F160W"] == ["iczgz4kjq_sci1"]
        # read_holdout_names: metrics-style json, json list, text
        with tempfile.TemporaryDirectory() as td:
            for fn, txt in (("a.json", json.dumps({"holdout": ["x", "y"], "validation": []})), ("b.json", json.dumps(["x", "y"])), ("c.txt", "# c\nx\ny\n")):
                p = os.path.join(td, fn)
                open(p, "w").write(txt)
                assert S.read_holdout_names(p) == ["x", "y"]
        # v28 holdout-from semantic (names only, no fill) is what select_exposures(..., holdout_names) does by default
        m28 = [dict(e) for e in nir]
        S28.select_exposures(m28, holdout_names=names)
        m29 = [dict(e) for e in nir]
        S.select_exposures(m29, holdout_names=names)
        assert held(m28) == held(m29)
        print("    --holdout-from + --n-test: file names kept, HST bands without one get 2 dithers; rootname entries hold out all chips")
    finally:
        S.set_registry(S.build_registry(S.NIRCAM_BANDS))


def test_e2_rgb_and_bootstrap():
    reg = S.build_registry(S.NIRCAM_BANDS)
    S.set_registry(reg)
    torch.manual_seed(0); cube = torch.rand(6, 8, 8); nrm = torch.rand(6) + 0.5
    assert np.array_equal(S28.rgb_from_cube(cube, 0.3, norm=nrm), S.rgb_from_cube(cube, 0.3, norm=nrm)), "rgb differs from v28 for six NIRCam bands"
    regm = S.build_registry(S.NIRCAM_BANDS, [{"band": "ACS_F814W", "instrument": "ACS/WFC", "filter": "F814W", "photplam": 8045.5},
                                             {"band": "WFC3IR_F160W", "instrument": "WFC3/IR", "filter": "F160W", "photplam": 15369.2}])
    S.set_registry(regm)
    try:
        g = S.rgb_groups(8)
        order = sorted(range(8), key=lambda k: regm.lam_um[regm.bands[k]])
        assert sum(g, []) == order and [len(x) for x in g] == [3, 3, 2], g
        rgb = S.rgb_from_cube(torch.rand(8, 8, 8), 0.3)
        assert rgb.shape == (8, 8, 3) and np.isfinite(rgb).all()
        # per-instrument bootstrap: pooled estimate of a subset equals the direct pooled score
        recs = []
        rng = np.random.default_rng(1)
        for b, ins in (("F150W", None), ("F444W", None), ("ACS_F814W", None), ("WFC3IR_F160W", None)):
            for k in range(6):
                n = int(rng.integers(100, 200))
                recs.append({"name": f"jw02736001001_02101_{k:05d}_nrcb1_cal.fits" if b in S.NIRCAM_BANDS else f"i{k}zgzbmgq_sci1", "band": b,
                             "instrument": regm.instrument[b], "n": n, "n_src": 10, "chi2_sum": n * (1.0 + 0.1 * k), "chi2_src_sum": 10 * 2.0,
                             "robust_sum": n * 0.9, "robust_src_sum": 10 * 1.5, "pulls": list(rng.normal(size=10))})
        bs = S.bootstrap_scores(recs, B=200)
        assert set(bs["per_instrument"]) == {"NIRCam", "ACS/WFC", "WFC3/IR"}
        sub = [r for r in recs if r["instrument"] == "ACS/WFC"]
        direct = S._pooled_scores(sub, [torch.tensor(r["pulls"], dtype=torch.float32) for r in sub])
        assert abs(bs["per_instrument"]["ACS/WFC"]["chi2_all"]["est"] - direct["chi2_all"]) < 1e-12
        line = S.boot_line("test", bs)
        assert "ACS/WFC (n=6)" in line and "WFC3/IR (n=6)" in line
        # the single-instrument bootstrap is the v28 one (no extra keys, same numbers)
        nir_recs = [r for r in recs if r["band"] in S.NIRCAM_BANDS]
        S.set_registry(reg)
        b29 = S.bootstrap_scores(nir_recs, B=200)
        b28 = S28.bootstrap_scores(nir_recs, B=200)
        assert "per_instrument" not in b29 and b28 == b29
        print("(e2) rgb groups, per-instrument bootstrap: ok; BOOT line tail:", line.split("| n_dithers")[1][:200])
    finally:
        S.set_registry(S.build_registry(S.NIRCAM_BANDS))


# ---------------------------------------------------------------------------------------------------- HST file selection
def test_hst_selection():
    import pandas as pd
    d = pd.read_csv(MANIFEST)
    d["inst"] = (d["instrument"] + "/" + d["detector"]).str.upper()
    files, info = S.select_hst_files("/nonexistent", MANIFEST, programs="14096", instruments="WFC3/IR,ACS/WFC", filters="F160W,F814W",
                                     mjd_min=57800, mjd_max=57900)
    exp = d[(d.proposal_id == 14096) & d.inst.isin(["WFC3/IR", "ACS/WFC"]) & d["filter"].isin(["F160W", "F814W"]) &
            (d.t_start_mjd >= 57800) & (d.t_start_mjd <= 57900)]
    assert info["n_selected"] == len(exp) and info["n_missing_files"] == len(exp) and files == [], (info["n_selected"], len(exp))
    # WFPC2 / other products are never selected; flt + flc of one root: flc wins
    _, info = S.select_hst_files("/nonexistent", MANIFEST)
    assert info["n_unsupported_or_other_product"] == int((d.detector == "PC").sum()) and info["n_selected"] == int((d.detector != "PC").sum())
    # header path on the sample directory: the two supported samples (the WFPC2 c1m and the NIRCam cal never match)
    f_all, info = S.select_hst_files(SAMPLES)
    assert [os.path.basename(f) for f in f_all] == ["jczgz3k8q_flc.fits", "iczgzbmgq_flt.fits"][::-1] or len(f_all) == 2, f_all
    assert sorted(os.path.basename(f) for f in f_all) == ["iczgzbmgq_flt.fits", "jczgz3k8q_flc.fits"]
    assert [os.path.basename(f) for f in S.select_hst_files(SAMPLES, instruments="ACS/WFC")[0]] == ["jczgz3k8q_flc.fits"]
    assert [os.path.basename(f) for f in S.select_hst_files(SAMPLES, filters="f160w")[0]] == ["iczgzbmgq_flt.fits"]
    assert S.select_hst_files(SAMPLES, programs="1")[0] == [] and S.select_hst_files(SAMPLES, mjd_min=58000)[0] == []
    f_m, info = S.select_hst_files(SAMPLES, MANIFEST, filters="F160W,F814W")  # manifest rows whose files exist in the dir
    assert sorted(os.path.basename(f) for f in f_m) == ["iczgzbmgq_flt.fits", "jczgz3k8q_flc.fits"] and info["n_missing_files"] > 20
    assert S.parse_dq_policy("default") == "default" and S.parse_dq_policy("336") == 336 and S.parse_dq_policy("strict") == "strict"
    print("(hst selection) manifest filters == pandas; header path on samples ->", [os.path.basename(f) for f in f_all])


# ---------------------------------------------------------------------------------------------------- (f)
def have_torchsky():
    try:
        import torchsky  # noqa: F401
        from torchsky.wcs import WCS  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        print("    torchsky not importable:", repr(exc)[:120])
        return False


def test_f_end_to_end_quick():
    if not have_torchsky() or not os.path.isdir(NIRCAM_DIR):
        print("(f) SKIPPED: torchsky (or the NIRCam test data) not available")
        return
    out = os.path.join(ROOT, "runs", "v29_quick")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v29.py"), "--data-dir", NIRCAM_DIR, "--hst-dir", SAMPLES,
           "--hst-instruments", "WFC3/IR,ACS/WFC", "--hst-dq-policy", "default", "--out", out, "--size", "128", "--quick", "--no-fallback",
           "--no-holdout", "--score-all", "--boot-b", "50", "--budget-explicit", "1000", "--budget-implicit", "1000",
           "--explicit-psf", "chrom", "--psf-model", "chrom", "--stripes", "1"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    open(os.path.join(out, "stdout.log"), "w").write(log)
    print(f"(f) end-to-end --quick run: exit {p.returncode} in {time.time() - t0:.0f} s; log {out}/stdout.log")
    for ln in log.splitlines():
        if any(k in ln for k in ("band registry", "HST:", "APPROXIMATION", "TEST ", "TEST-BAND", "BOOT ", "per band", "Traceback", "Error")):
            print("    " + ln[:420])
    assert p.returncode == 0, log[-3000:]
    M = json.load(open(os.path.join(out, "metrics.json")))
    assert [r["band"] for r in M["band_registry"]] == S.NIRCAM_BANDS + ["ACS_F814W", "WFC3IR_F160W"]
    hp = M["metrics"]["heldout_per_instrument"]
    for method, per in hp.items():
        assert set(per) == {"NIRCam", "ACS/WFC", "WFC3/IR"}, (method, list(per))
        assert all(np.isfinite(v["chi2"]) for v in per.values()), per
    from astropy.io import fits
    sw = fits.getdata(os.path.join(out, "cubes", "swarp.fits"))
    assert sw.shape == (8, 128, 128) and np.isfinite(sw).all()
    assert (sw[6] != 0).sum() > 500 and (sw[7] != 0).sum() > 500, "HST coadd planes empty"
    assert "heldout_chi2" in M["metrics"] and all(np.isfinite(v) for v in M["metrics"]["heldout_chi2"].values())
    print("    metrics: heldout_chi2", {k: round(v, 3) for k, v in M["metrics"]["heldout_chi2"].items()})
    for k, v in hp.items():
        print("    per instrument", k, {i: round(x["chi2"], 3) for i, x in v.items()})


def _run_v29(args, out_name):
    out = os.path.join(ROOT, "runs", out_name)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v29.py"), "--data-dir", NIRCAM_DIR, "--out", out, "--size", "128",
           "--quick", "--no-fallback", "--boot-b", "50", "--budget-explicit", "1000", "--budget-implicit", "1000"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(out, "stdout.log"), "w").write(p.stdout + p.stderr)
    return p, out


def test_f2_hst_holdout_spectral_and_refusals():
    """HST holdouts from a file (chip name + rootname), --n-test fill, manifest selection, --lam-lo, SVO bandpasses of HST filters
    (network), and the two refusals (pupil PSF with HST bands; pivot below --lam-lo)."""
    hst_all = os.path.join(ROOT, "testdata", "hst_all")
    svo = os.path.join(ROOT, "testdata", "svo_cache")
    if not (have_torchsky() and os.path.isdir(hst_all) and os.path.isdir(NIRCAM_DIR)):
        print("(f2) SKIPPED: torchsky or test data not available")
        return
    with tempfile.TemporaryDirectory() as td:
        hl = os.path.join(td, "holdout.txt")
        open(hl, "w").write("iczgzbmbq_sci1\njczgz3k6q\n")
        common = ["--hst-dir", hst_all, "--hst-manifest", MANIFEST, "--hst-programs", "14096", "--hst-filters", "F160W,F814W"]
        p, out = _run_v29(common + ["--holdout-from", hl, "--n-test", "1", "--explicit-psf", "chrom", "--psf-model", "chrom", "--spectral", "rbf",
                                    "--lam-lo", "0.35", "--svo-cache", svo, "--spec-q", "3"], "v29_quick_b")
        assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
        M = json.load(open(os.path.join(out, "metrics.json")))
        assert sorted(M["holdout"]) == ["iczgzbmbq_sci1", "jczgz3k6q_sci1", "jczgz3k6q_sci2"], M["holdout"]
        assert M["hst_selection"]["n_missing_files"] == 5 and M["hst_selection"]["loaded_chips_per_band"] == {"ACS_F814W": 3, "WFC3IR_F160W": 3}
        assert M["lam_domain_um"] == [0.35, 5.3] and set(M["bandpass_nodes_um"]) == {r["band"] for r in M["band_registry"]}
        hp = M["metrics"]["heldout_per_instrument"]["explicit_chrom_3d"]
        assert hp["ACS/WFC"]["n_exp"] == 2 and hp["WFC3/IR"]["n_exp"] == 1 and "NIRCam" not in hp  # only HST exposures are held out
        print("(f2) HST holdouts [chip + rootname] =", M["holdout"], "| per-instrument n_exp:", {k: v["n_exp"] for k, v in hp.items()})
        print("     SVO bandpass nodes (um) of the HST bands:", {b: M["bandpass_nodes_um"][b] for b in ("ACS_F814W", "WFC3IR_F160W")})
        p, _ = _run_v29(common + ["--explicit-psf", "pupil"], "v29_refuse_pupil")
        assert p.returncode != 0 and "NIRCam-only" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-1500:]
        print("     refusal 1:", [ln for ln in (p.stdout + p.stderr).splitlines() if "NIRCam-only" in ln][0][:200])
        p, _ = _run_v29(common + ["--spectral", "rbf", "--lam-lo", "0.9", "--svo-cache", svo], "v29_refuse_lam")
        assert p.returncode != 0 and "set --lam-lo/--lam-hi" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-1500:]
        print("     refusal 2:", [ln for ln in (p.stdout + p.stderr).splitlines() if "lam-lo" in ln][0][:200])
        p, _ = _run_v29(["--hst-filters", "F160W"], "v29_refuse_flags")
        assert p.returncode != 0 and "need --hst-dir" in (p.stdout + p.stderr)


if __name__ == "__main__":
    tests = [test_a_registry_nircam_equals_v28, test_b_registry_mixed, test_b2_load_bandpasses_arguments, test_c_chromatic_psf,
             test_d_splat_coadd, test_e_selection, test_e2_rgb_and_bootstrap, test_hst_selection, test_f_end_to_end_quick,
             test_f2_hst_holdout_spectral_and_refusals]
    only = sys.argv[1:]
    for t in tests:
        if only and not any(o in t.__name__ for o in only):
            continue
        t()
    print("\nALL TESTS PASSED")
