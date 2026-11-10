"""Tests for smacs_four_v38.py, scripts/make_holdout.py and scripts/null_stacks.py. Plain asserts, no pytest:
    python tests/test_v38.py [a b c d e f]
(a) header parsing: NIRISS (filter in PUPIL or in FILTER), NIRCam module A, medium bands, rejected combinations, MIRI switch; registry rows
(b) make_holdout.py on the local HST manifest (T1-T4, JSON readable by v38's read_holdout_names) and on JWST headers
(c) synthetic NIRISS-like _cal files (a PLUMBING test only: copies of NIRCam LW test files with edited headers and CD matrix; the images are NIRCam
    images, the SVO curve is the NIRCam one renamed) + a normal quick run (run A, NIRCam only) with --dump-heldout-pixels
(d) frozen-sky run B (--freeze-sky-from A --freeze-instruments-from A, NIRCam frozen, synthetic NIRISS new, --extra-jwst-dirs): sky identical and frozen,
    NIRCam operators unchanged, NIRISS operators learned; mismatching grid refused with a message
(e) null_stacks.py on the dumped pixels of run A
(f) null_stacks.py statistics on synthetic residuals (noise only: chi2/dof ~ 1; a row-dependent bias is detected; the model-error floor is recovered)
tests/check_v38_equiv.py is the bitwise comparison with v37."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_holdout as MH  # noqa: E402
import null_stacks as NS  # noqa: E402
import smacs_four_v38 as S  # noqa: E402

NIR = os.path.join(ROOT, "testdata", "nircam")
SYN = os.path.join(ROOT, "testdata", "niriss_synth")
SVO = os.path.join(ROOT, "testdata", "svo_cache")
SVO_N = os.path.join(ROOT, "testdata", "svo_cache_niriss_synth")
MANIFEST = os.path.join(ROOT, "hst", "hst_imaging_unique.csv")
PY = sys.executable
RUNS = os.path.join(ROOT, "runs", "test_v38")
CODE = ("import sys, runpy, torch; torch.manual_seed(0); torch.set_num_threads(2); sys.argv = sys.argv[1:]; "
        "sys.path.insert(0, __import__('os').path.dirname(sys.argv[0])); runpy.run_path(sys.argv[0], run_name='__main__')")
E2 = ["--size", "128", "--quick", "--no-fallback", "--budget-explicit", "100000", "--skip-implicit", "--explicit-configs", "explicit", "--spectral", "rbf",
      "--spec-q", "3", "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--lib-source", "gauss", "--lib-learn-table", "--lib-res", "off",
      "--psf-window-arcsec", "3.0", "--sample-mode", "bicubic", "--halo-px", "16", "--eval-every-min", "0", "--boot-b", "20", "--stripes", "1", "--seed", "1"]


def run_v38(tag, extra, expect_fail=False):
    out = os.path.join(RUNS, tag)
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    p = subprocess.run([PY, "-u", "-c", CODE, os.path.join(ROOT, "scripts", "smacs_four_v38.py"), "--out", out] + E2 + extra, capture_output=True, text=True)
    print(f"  run {tag}: exit {p.returncode} in {time.time() - t0:.0f} s")
    if expect_fail:
        assert p.returncode != 0, p.stdout[-1500:]
        return out, p.stdout + p.stderr
    if p.returncode:
        print(p.stdout[-2500:], p.stderr[-3000:])
        raise SystemExit(1)
    return out, p.stdout


# ------------------------------------------------------------------------------------------------------------------------------ (a)
def hdr(**kw):
    h = fits.Header()
    base = {"TELESCOP": "JWST", "INSTRUME": "NIRISS", "DETECTOR": "NIS", "FILTER": "CLEAR", "PUPIL": "F150W", "EXP_TYPE": "NIS_IMAGE"}
    base.update(kw)
    for k, v in base.items():
        h[k] = v
    return h


def test_a():
    print("(a) header parsing and registry")
    i = S.jwst_header_info(hdr())
    assert (i["instrument"], i["band"], i["filter"], i["channel"], i["svo_id"]) == ("NIRISS", "NIRISS_F150W", "F150W", "NIRISS", "JWST/NIRISS.F150W"), i
    i = S.jwst_header_info(hdr(FILTER="F277W", PUPIL="CLEARP"))  # filter in the filter wheel
    assert i["band"] == "NIRISS_F277W" and i["svo_id"] == "JWST/NIRISS.F277W", i
    assert S.jwst_header_info(hdr(PUPIL="NRM", EXP_TYPE="NIS_AMI"))["skip"]
    assert "unsupported" in S.jwst_header_info(hdr(FILTER="F277W", PUPIL="F150W", EXP_TYPE="NIS_IMAGE"))["skip"]
    assert "not imaging" in S.jwst_header_info(hdr(EXP_TYPE="NIS_WFSS", FILTER="F150W", PUPIL="GR150R"))["skip"]
    for filt, det, chan in [("F182M", "NRCA3", "NIRCAM_SW"), ("F210M", "NRCA1", "NIRCAM_SW"), ("F300M", "NRCALONG", "NIRCAM_LW"), ("F410M", "NRCALONG", "NIRCAM_LW"),
                            ("F115W", "NRCA4", "NIRCAM_SW"), ("F150W", "NRCB2", "NIRCAM_SW"), ("F444W", "NRCBLONG", "NIRCAM_LW")]:
        i = S.jwst_header_info(hdr(INSTRUME="NIRCAM", DETECTOR=det, FILTER=filt, PUPIL="CLEAR", EXP_TYPE="NRC_IMAGE"))
        assert i["band"] == filt and i["channel"] == chan and i["svo_id"] == f"JWST/NIRCam.{filt}" and i["detector"] == det, i
    assert "unsupported" in S.jwst_header_info(hdr(INSTRUME="NIRCAM", DETECTOR="NRCBLONG", FILTER="F444W", PUPIL="F405N", EXP_TYPE="NRC_IMAGE"))["skip"]
    assert "LW" in S.jwst_header_info(hdr(INSTRUME="NIRCAM", DETECTOR="NRCA3", FILTER="F410M", PUPIL="CLEAR", EXP_TYPE="NRC_IMAGE"))["skip"] \
        or "NIRCAM_LW" in S.jwst_header_info(hdr(INSTRUME="NIRCAM", DETECTOR="NRCA3", FILTER="F410M", PUPIL="CLEAR", EXP_TYPE="NRC_IMAGE"))["skip"]
    m = hdr(INSTRUME="MIRI", DETECTOR="MIRIMAGE", FILTER="F770W", PUPIL="", EXP_TYPE="MIR_IMAGE")
    assert "--include-miri" in S.jwst_header_info(m)["skip"]
    assert S.jwst_header_info(m, include_miri=True)["band"] == "MIRI_F770W"
    assert S.jwst_header_info(hdr(TELESCOP="HST"))["skip"]
    # registry from fake exposure dicts: main-dir NIRCam (no instrument key), extra NIRCam, NIRISS, HST
    fake = lambda band, **kw: dict({"band": band}, **kw)
    exps = [fake("F150W"), fake("F444W"), fake("F182M", instrument="NIRCam", facility="JWST", filter="F182M"), fake("F410M", instrument="NIRCam", facility="JWST", filter="F410M"),
            fake("NIRISS_F115W", instrument="NIRISS", facility="JWST", filter="F115W"), fake("NIRISS_F200W", instrument="NIRISS", facility="JWST", filter="F200W"),
            fake("WFC3IR_F160W", instrument="WFC3/IR", filter="F160W", photplam=15369.0, svo_id=None)]
    reg = S.registry_from_exposures(exps)
    assert reg.bands == ["F150W", "F182M", "F410M", "F444W", "NIRISS_F115W", "NIRISS_F200W", "WFC3IR_F160W"], reg.bands
    assert reg.channel["NIRISS_F115W"] == "NIRISS" and reg.channel["F182M"] == "NIRCAM_SW" and reg.channel["F410M"] == "NIRCAM_LW"
    assert reg.channels == ["NIRCAM_SW", "NIRCAM_LW", "NIRISS", "WFC3_IR"], reg.channels
    assert reg.svo_id["NIRISS_F200W"] == "JWST/NIRISS.F200W" and reg.instrument["NIRISS_F200W"] == "NIRISS" and reg.facility["NIRISS_F200W"] == "JWST"
    assert not reg.is_hst["NIRISS_F115W"] and not reg.is_nircam["NIRISS_F115W"] and reg.is_hst["WFC3IR_F160W"] and reg.is_nircam["F182M"]
    assert abs(reg.pixel_scale["NIRISS_F115W"] - 0.0656) < 1e-9 and reg.pixel_scale["F410M"] == 0.063 and reg.pixel_scale["F182M"] == 0.031
    assert 0.05 < reg.fwhm_arcsec["F182M"] < 0.07 and 0.13 < reg.fwhm_arcsec["F410M"] < 0.14, (reg.fwhm_arcsec["F182M"], reg.fwhm_arcsec["F410M"])
    S.set_registry(reg)
    assert S.uses_stripes("F150W") and S.uses_stripes("F410M") and not S.uses_stripes("NIRISS_F115W"), "stripes must stay NIRCam-only"
    try:
        S.require_nircam_only("pupil")
        raise AssertionError("no refusal")
    except SystemExit as exc:
        assert "pupil" in str(exc)
    # v37 registry unchanged when nothing extra is present
    reg0 = S.registry_from_exposures([fake(b) for b in S.NIRCAM_BANDS])
    assert reg0.bands == S.NIRCAM_BANDS and reg0.channels == ["NIRCAM_SW", "NIRCAM_LW"] and all(reg0.is_nircam.values()) and not any(reg0.is_hst.values())
    print("  header parsing and registry OK")


# ------------------------------------------------------------------------------------------------------------------------------ (b)
def check_json_v38(path):
    names = S.read_holdout_names(path)  # the v38 reader
    assert names and len(set(names)) == len(names)
    return names


def test_b():
    print("(b) make_holdout.py")
    tmp = tempfile.mkdtemp()
    recs, _ = MH.hst_records_from_manifest(MANIFEST)
    assert len(recs) > 90, len(recs)
    byname = {r["name"]: r for r in recs}
    # T1
    o = os.path.join(tmp, "t1.json")
    MH.main(["t1", "--manifest", MANIFEST, "--out", o])
    names = check_json_v38(o)
    obj = json.load(open(o))
    assert all(v["n_train"] >= 2 for v in obj["summary"]["per_band"].values() if v["n_test"])
    assert all(v["n_test"] in (0, 2) for v in obj["summary"]["per_band"].values())
    # T3, both directions
    o = os.path.join(tmp, "t3.json")
    MH.main(["t3", "--manifest", MANIFEST, "--test-programs", "16729", "--train-programs", "14096", "--both-directions", "--out", o])
    f, r = check_json_v38(o), check_json_v38(os.path.join(tmp, "t3_reverse.json"))
    assert all(byname[n]["program"] == 16729 for n in f) and all(byname[n]["program"] == 14096 for n in r)
    assert {byname[n]["band"] for n in f} == {"WFC3/IR:F105W"}, {byname[n]["band"] for n in f}  # only band common to both epochs in the manifest
    assert not set(f) & set(r)
    # T4: WFC3/IR as the new instrument
    o = os.path.join(tmp, "t4.json")
    MH.main(["t4", "--manifest", MANIFEST, "--new-instrument", "WFC3/IR", "--out", o])
    t = check_json_v38(o)
    assert all(byname[n]["instrument"] == "WFC3/IR" for n in t)
    pb = json.load(open(o))["summary"]["per_band"]
    for band, v in pb.items():
        if band.startswith("WFC3/IR"):
            assert v["n_test"] >= 1 and v["n_train"] >= 1, (band, v)
            assert abs(v["n_test"] - v["n_train"]) <= max(3, 0.3 * (v["n_test"] + v["n_train"])), (band, v)  # about half
        else:
            assert v["n_test"] == 0
    # T2 with a synthetic roll table (the manifest has no rolls: TEST FIXTURE, not data): one roll per visit from four angles
    rolls = os.path.join(tmp, "rolls.csv")
    angles = [12.0, 55.0, 98.0, 143.0]
    with open(rolls, "w") as fh:
        fh.write("name,roll_deg\n")
        for k, rr in enumerate(recs):
            vis = sorted({x["visit"] for x in recs}).index(rr["visit"])
            fh.write(f"{rr['name']},{angles[vis % 4] + 0.3 * (k % 3)}\n")
    o = os.path.join(tmp, "t2.json")
    MH.main(["t2", "--manifest", MANIFEST, "--roll-csv", rolls, "--instruments", "WFC3/IR,ACS/WFC", "--out", o])
    t = check_json_v38(o)
    obj = json.load(open(o))
    rolltab = {rr["name"]: rr for rr in recs}
    attach = {}
    for ln in open(rolls).read().splitlines()[1:]:
        n, v = ln.split(",")
        attach[n] = float(v)
    for inst, info in obj["roll_clusters"].items():
        held = [byname[n] for n in t if byname[n]["instrument"] == inst]
        train = [rr for rr in recs if rr["instrument"] == inst and rr["name"] not in set(t)]
        if held:
            hr = [attach[x["name"]] for x in held]
            tr = [attach[x["name"]] for x in train]
            assert min(MH.circ_dist(a, b) for a in hr for b in tr) > 5.0, inst  # distinct roll angles
    assert t and obj["roll_clusters"]
    # T2 needs rolls
    try:
        MH.main(["t2", "--manifest", MANIFEST, "--out", os.path.join(tmp, "x.json")])
        raise AssertionError("T2 without rolls must stop")
    except SystemExit as exc:
        assert "roll" in str(exc)
    # T2 from real HST headers (testdata/hst_all: PA_V3 359.0 for WFC3/IR, 253.7 for ACS): one roll cluster per instrument -> warning, nothing held out
    nm = os.path.join(tmp, "names.txt")
    open(nm, "w").write("iczgzbmaq\niczgzbmbq\niczgzbmgq\njczgz3k6q\njczgz3k8q\n")
    rr = MH.records_from_names(nm)
    MH.attach_rolls(rr, hst_dir=os.path.join(ROOT, "testdata", "hst_all"))
    got = {r["name"]: r["roll"] for r in rr}
    assert abs(got["iczgzbmaq"] - 359.0) < 0.01 and abs(got["jczgz3k6q"] - 253.71) < 0.01, got
    # JWST headers: testdata/nircam_all (names + headers)
    o = os.path.join(tmp, "t1j.json")
    MH.main(["t1", "--jwst-dirs", os.path.join(ROOT, "testdata", "nircam_all"), "--out", o])
    jr = MH.jwst_records_from_dirs([os.path.join(ROOT, "testdata", "nircam_all")], "*_cal.fits")
    assert {r["instrument"] for r in jr} == {"NIRCam"} and {r["filter"] for r in jr} == {"F090W", "F150W", "F200W", "F277W", "F356W", "F444W"}, {r["filter"] for r in jr}
    assert all(r["roll"] is not None and np.isfinite(r["mjd"]) for r in jr)
    print("  make_holdout OK")


# ------------------------------------------------------------------------------------------------------------------------------ (c)
def make_synth_niriss():
    """Plumbing fixture: NIRCam LW test files turned into NIRISS-like files (headers, CD scaled to 0.0656/0.063). NOT NIRISS data."""
    if len(glob.glob(os.path.join(SYN, "*_cal.fits"))) >= 2 and os.path.isdir(SVO_N):
        return
    os.makedirs(SYN, exist_ok=True)
    spec = [("jw02736001001_02103_00004_nrcblong_cal.fits", "jw02736001001_02103_00001_nis_cal.fits", "CLEAR", "F356W"),
            ("jw02736001001_02101_00002_nrcblong_cal.fits", "jw02736001001_02101_00001_nis_cal.fits", "F277W", "CLEARP")]
    for src, dst, filt, pup in spec:
        with fits.open(os.path.join(NIR, src), memmap=False) as h:
            h0 = h[0].header
            h0["INSTRUME"], h0["DETECTOR"], h0["FILTER"], h0["PUPIL"], h0["EXP_TYPE"] = "NIRISS", "NIS", filt, pup, "NIS_IMAGE"
            for k in ("MODULE", "CHANNEL"):
                if k in h0:
                    del h0[k]
            h0["COMMENT"] = "SYNTHETIC plumbing-test file: a NIRCam long-wavelength image with edited headers; not NIRISS data"
            sh = h["SCI"].header
            for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
                if k in sh:
                    sh[k] = sh[k] * 0.0656 / 0.063
            h.writeto(os.path.join(SYN, dst), overwrite=True)
    os.makedirs(SVO_N, exist_ok=True)
    for f in glob.glob(os.path.join(SVO, "*.json")):
        shutil.copy(f, SVO_N)
    for filt in ("F356W", "F277W"):
        d = json.load(open(os.path.join(SVO, f"filter_JWST_NIRCam.{filt}.json")))
        d["filter_id"] = f"JWST/NIRISS.{filt}"
        json.dump(d, open(os.path.join(SVO_N, f"filter_JWST_NIRISS.{filt}.json"), "w"))


STATE = {}


def test_c():
    print("(c) synthetic NIRISS plumbing + run A")
    make_synth_niriss()
    for f in glob.glob(os.path.join(SYN, "*_cal.fits")):
        i = S.jwst_file_info(f)
        assert i["instrument"] == "NIRISS" and i["band"].startswith("NIRISS_F") and i["detector"] == "NIS", i
    out, so = run_v38("A", ["--data-dir", NIR, "--svo-cache", SVO_N, "--dump-heldout-pixels", "--score-all"])
    STATE["A"] = out
    d = json.load(open(os.path.join(out, "metrics.json")))
    assert "TEST explicit_lib_3d" in so and d["band_registry"] and "extra_jwst" not in d
    fl = glob.glob(os.path.join(out, "heldout_pixels", "explicit_lib_3d", "*.npz"))
    assert len(fl) == 6, len(fl)
    z = np.load(fl[0])
    need = {"name", "band", "instrument", "det", "row", "col", "dx", "dy", "tx", "ty", "data", "model", "sigma", "mjd", "roll_deg", "model_sky", "split"}
    assert need <= set(z.files), need - set(z.files)
    n = len(z["data"])
    assert all(len(z[k]) == n for k in ("row", "col", "dx", "dy", "tx", "ty", "model", "sigma", "model_sky")) and np.isfinite(z["mjd"]) and z["mjd"] > 59000
    assert str(z["instrument"]) == "NIRCam" and np.isfinite(z["roll_deg"])
    # every exposure has its own coordinates (the old heldout_resid.npy kept one set per band)
    allc = [np.load(f)["tx"][:50].tobytes() for f in fl]
    assert len(set(allc)) == len(fl)
    chk = np.load(fl[0])
    assert np.allclose(chk["model_sky"] + chk["model_bkg"], chk["model"], atol=1e-4 * np.abs(chk["model"]).max())
    print("  run A and dumps OK")


def test_d():
    print("(d) frozen-sky run B")
    if "A" not in STATE:
        test_c()
    A = STATE["A"]
    out, so = run_v38("B", ["--data-dir", NIR, "--extra-jwst-dirs", SYN, "--svo-cache", SVO_N, "--freeze-sky-from", A, "--freeze-instruments-from", A,
                            "--dump-heldout-pixels", "--score-all"])
    d = json.load(open(os.path.join(out, "metrics.json")))
    assert d["freeze_sky"]["band_buffers_not_loaded"] == ["sky.lam_nodes", "sky.w_nodes"] and "NIRISS_F356W" in d["exposures_per_band"], d["exposures_per_band"]
    assert d["freeze_sky"]["band_buffer_shapes"]["sky.lam_nodes"][0] == 8 and d["freeze_sky"]["source_band_buffer_shapes"]["sky.lam_nodes"][0] == 6  # 6 -> 8 bands
    assert d["freeze_instruments"]["n_exposures_frozen"] == 6 and d["freeze_instruments"]["n_exposures_new"] == 2
    assert d["freeze_instruments"]["new_exposures_per_instrument"] == {"NIRCam": 0, "NIRISS": 2}, d["freeze_instruments"]["new_exposures_per_instrument"]
    assert d["extra_jwst"]["dirs"][SYN]["n_loaded"] == 2 and d["pixel_scale_check"]["n_checked"] == 8
    reg = {r["band"]: r for r in d["band_registry"]}
    assert reg["NIRISS_F356W"]["svo_id"] == "JWST/NIRISS.F356W" and reg["NIRISS_F356W"]["channel"] == "NIRISS"
    sa = torch.load(os.path.join(A, "explicit_lib_3d_state.pt"), map_location="cpu")
    sb = torch.load(os.path.join(out, "explicit_lib_3d_state.pt"), map_location="cpu")
    assert torch.equal(sa["sky.coeffs.s"], sb["sky.coeffs.s"]), "frozen sky changed"
    nbands_a, nbands_b = sa["sky.lam_nodes"].shape[0], sb["sky.lam_nodes"].shape[0]
    assert (nbands_a, nbands_b) == (6, 8)
    # NIRCam operators: exposure order in A: names from metrics noise_scale; B has the same NIRCam files first
    na, nb = list(json.load(open(os.path.join(A, "metrics.json")))["noise_scale"]), list(d["noise_scale"])
    ia = {n: i for i, n in enumerate(na)}
    ib = {n: i for i, n in enumerate(nb)}
    for key in ("bkg.pe", "bkg.logg", "geom.M", "geom.t", "stripes.r", "stripes.c"):
        for n in na:
            assert torch.equal(sa[key][ia[n]], sb[key][ib[n]]), (key, n)
    changed_new = sum(float((sb[k][ib[n]] - 0).abs().max()) > 0 for k in ("bkg.logg", "stripes.r", "geom.t") for n in nb if n.endswith("_nis_cal.fits"))
    assert changed_new > 0, "the NIRISS operators did not move"
    da, db = json.load(open(os.path.join(A, "metrics.json")))["detectors"], d["detectors"]
    for k in ("bkg.pd", "geom.D"):
        for dn in da:
            assert torch.equal(sa[k][da.index(dn)], sb[k][db.index(dn)]), (k, dn)
    frozen_tabs = [k for k in d["freeze_instruments"]["tables"].values() if k == "frozen"]
    assert len(frozen_tabs) == 6 and sum(v.startswith("new") for v in d["freeze_instruments"]["tables"].values()) == 2, d["freeze_instruments"]["tables"]
    # the NIRISS table moved from its Gaussian start; the frozen NIRCam tables did not
    tabkeys = [k for k in sb if k.startswith("psf.tab.")]
    assert len(tabkeys) == 8
    # NIRISS predictions were scored
    assert any("NIRISS" in k for k in d["metrics"]["heldout_per_band"]["explicit_lib_3d"]), d["metrics"]["heldout_per_band"]["explicit_lib_3d"].keys()
    assert glob.glob(os.path.join(out, "heldout_pixels", "explicit_lib_3d", "*_nis_cal.npz"))
    # mismatching grid is refused with a message, nothing loaded silently
    out2, so2 = run_v38("B_bad", ["--data-dir", NIR, "--svo-cache", SVO_N, "--freeze-sky-from", A, "--score-all", "--halo-px", "8"], expect_fail=True)
    assert "not on the same grid" in so2 and "halo_px" in so2, so2[-800:]
    # option checks
    out3, so3 = run_v38("B_bad2", ["--data-dir", NIR, "--svo-cache", SVO_N, "--freeze-instruments-from", A], expect_fail=True)
    assert "needs --freeze-sky-from" in so3
    out4, so4 = run_v38("B_miri", ["--data-dir", NIR, "--svo-cache", SVO_N, "--include-miri"], expect_fail=True)
    assert "--include-miri needs --extra-jwst-dirs" in so4
    print("  frozen-sky transfer OK")


# ------------------------------------------------------------------------------------------------------------------------------ (e)
def test_e():
    print("(e) null_stacks on the dumped pixels of run A")
    if "A" not in STATE:
        test_c()
    A = STATE["A"]
    s = NS.main([A, "--splits", "all", "--peak-snr", "8", "--peak-radius", "4", "--min-n", "20"])
    od = os.path.join(A, "null_stacks", "explicit_lib_3d")
    for f in ("null_a_position.png", "null_b_rowcol.png", "null_c_roll.png", "null_d_mjd.png", "null_e_fractional.png", "null_summary.json"):
        assert os.path.getsize(os.path.join(od, f)) > 1000, f
    js = json.load(open(os.path.join(od, "null_summary.json")))
    assert js["n_exposures"] == 6 and js["n_pixels"] > 50000
    for dn, v in js["a_position"].items():
        assert v["chi2_unit"] is not None and np.isfinite(v["chi2_over_dof"]) and v["dof"] > 0, (dn, v)
        assert v["perm_mean"] is not None and v["perm_mean"] > 0 and v["perm_p95"] >= v["perm_mean"], (dn, v)  # shuffle null (includes any global mean offset of z)
    assert js["c_roll"]["n_exposures"] == 6 and js["d_mjd"]["n_exposures"] == 6
    assert "all" in js["e_fractional"]
    print("  f_peaks:", {k: js["f_peaks"].get(k) for k in ("n_peaks", "note")})
    print("  null_stacks on real quick-run pixels OK")


# ------------------------------------------------------------------------------------------------------------------------------ (f)
def synth_exposure(rng, name, det, n=200000, bias_row=0.0, floor=0.0, mjd=59700.0, roll=10.0, speak=True):
    row = rng.integers(0, 2048, n).astype(np.int32)
    col = rng.integers(0, 2048, n).astype(np.int32)
    sig = np.full(n, 0.01, np.float32)
    sky = (rng.pareto(1.5, n) * 0.02 + 0.01).astype(np.float32)
    model = sky + 0.5
    noise = rng.normal(0, 1, n) * sig
    data = model + noise + floor * sky * rng.normal(0, 1, n) + bias_row * sig * np.sin(row / 2048.0 * 2 * np.pi)
    return dict(name=name, band="F150W", instrument="NIRCam", det=det, split="test", mjd=mjd, roll_deg=roll, row=row, col=col,
                dx=(col / 1024.0 - 1).astype(np.float32), dy=(row / 1024.0 - 1).astype(np.float32), data=data.astype(np.float32), model=model.astype(np.float32),
                model_sky=sky, sigma=sig)


def dump_synth(root, exps):
    d = os.path.join(root, "heldout_pixels", "m")
    os.makedirs(d, exist_ok=True)
    for e in exps:
        np.savez_compressed(os.path.join(d, e["name"] + ".npz"), **e)


def test_f():
    print("(f) null_stacks statistics on synthetic residuals")
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    r0 = os.path.join(tmp, "null")
    dump_synth(r0, [synth_exposure(rng, f"e{i}", "NRCB1", mjd=59700 + i, roll=10 + 30 * i) for i in range(4)])
    s0 = NS.main([r0, "--splits", "all", "--peak-snr", "1e9"])
    v = s0["a_position"]["NRCB1"]
    assert 0.7 < v["chi2_over_dof"] < 1.4 and 0.7 < v["chi2_scatter"] / v["dof"] < 1.4, v  # noise only: chi2/dof ~ 1
    assert 0.8 < s0["b_rowcol"]["NRCB1"]["row"]["chi2_over_dof"] < 1.3
    # a row-dependent bias of 0.3 sigma amplitude is detected in the row test, not in the column test
    r1 = os.path.join(tmp, "bias")
    dump_synth(r1, [synth_exposure(rng, f"e{i}", "NRCB1", bias_row=0.3) for i in range(4)])
    s1 = NS.main([r1, "--splits", "all", "--peak-snr", "1e9"])
    assert s1["b_rowcol"]["NRCB1"]["row"]["chi2_over_dof"] > 20 and s1["b_rowcol"]["NRCB1"]["col"]["chi2_over_dof"] < 1.5, s1["b_rowcol"]["NRCB1"]
    assert s1["b_rowcol"]["NRCB1"]["row"]["perm_p95"] < 0.5 * s1["b_rowcol"]["NRCB1"]["row"]["chi2_unit"]
    # a 3 per cent model-error floor is recovered from (d - m)/m versus S/N
    r2 = os.path.join(tmp, "floor")
    dump_synth(r2, [synth_exposure(rng, f"e{i}", "NRCB1", floor=0.03, n=400000) for i in range(4)])
    s2 = NS.main([r2, "--splits", "all", "--peak-snr", "1e9"])
    fl = s2["e_fractional"]["all"]
    assert abs(fl["floor_f"] - 0.03) < 0.006 and 0.7 < fl["noise_factor_a"] < 1.4, fl
    s0f = s0["e_fractional"]["all"]
    assert s0f["floor_f"] < 0.02, s0f["floor_f"]  # noise only: floor consistent with 0 within the fit scatter (few high-S/N pixels)
    print(f"  noise only chi2/dof {v['chi2_over_dof']:.2f}; row bias chi2/dof {s1['b_rowcol']['NRCB1']['row']['chi2_over_dof']:.0f}; floor recovered {fl['floor_f']:.4f} (true 0.03), a {fl['noise_factor_a']:.2f}")
    print("  null_stacks statistics OK")


if __name__ == "__main__":
    which = sys.argv[1:] or list("abcdef")
    for k in which:
        {"a": test_a, "b": test_b, "c": test_c, "d": test_d, "e": test_e, "f": test_f}[k]()
    print("\nALL TESTS PASSED")
