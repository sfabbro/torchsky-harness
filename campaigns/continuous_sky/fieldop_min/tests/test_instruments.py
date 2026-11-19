"""Instrument interface (fieldop/instruments.py, INSTRUMENT_INTERFACE.md).

1. data-quality policies: NIRCam's is the old expression bit for bit; HST 16-bit words with the top bit
2. HST conversion to MJy/sr: PHOTFNU path against PHOTFLAM path on the real headers; ELECTRONS divided by EXPTIME
3. instrument names and aliases; unknown instruments refused
4. spectral domain, filter-curve grid and K from the loaded curves: unchanged for the synthetic and NIRCam-like filters, widened for a blue filter
5. real files through data.load_field (one NIRCam, one WFC3/IR, one two-chip ACS/WFC file; stand-in filter curves: no curve cache here)
6. two instruments through the command line: identical to the package before the interface (recorded), and the flux scale needs an
   explicit reference instrument
"""
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import bandpass, data, instruments as I, sky  # noqa: E402
from fieldop.cli import default_config  # noqa: E402

JW = "/home/claude/work/dl_full/jw02736001001_02101_00009_nrcb1_cal.fits"
IR = "/home/claude/work/dl_hst/iczgz2nsq_flt.fits"
ACS = "/home/claude/work/dl_hst/jczgz3k8q_flc.fits"
have_real = all(os.path.exists(p) for p in (JW, IR, ACS))
try:
    import torchsky.wcs  # noqa: F401  (the header solutions of the NIRCam files and the target plane; PYTHONPATH=/home/claude/torchsky/src)
    have_tsky = True
except ImportError:
    have_tsky = False


@pytest.fixture(autouse=True)
def reset_domain():
    yield
    sky.set_spectral_domain(sky.LAM_LO, sky.LAM_HI)
    I.DQ_IGNORE_OVERRIDE.clear()
    data.FILTER_CURVE = None


# ----------------------------------------------------------------------------- 1
def test_dq_policies():
    rng = np.random.default_rng(0)
    q32 = (rng.integers(0, 2, (200, 200)) * rng.choice([1, 2, 4, 2048, 1 << 31], (200, 200))).astype(np.uint32)
    for ig in (0, 2, 4, 6):
        data.DQ_IGNORE_BITS = ig
        new = I.INSTRUMENTS["NIRCAM"].dq_policy()(q32)
        old = (q32.astype(np.int64) & ~int(ig)) == 0
        assert np.array_equal(new, old)
    data.DQ_IGNORE_BITS = 0
    q16 = np.array([0, 16, 512, 528, 256, 4096, -32768, -32768 + 16], dtype=np.int16)      # -32768 = bit 32768 read as signed
    ok = I.INSTRUMENTS["WFC3_IR"].dq_policy()(q16)
    assert ok.tolist() == [True, True, True, True, False, False, False, False]
    ok = I.INSTRUMENTS["ACS_WFC"].dq_policy()(np.array([0, 16, 64, 80, 256, 2048, 4096], dtype=np.int16))
    assert ok.tolist() == [True, True, True, True, False, False, False]
    I.DQ_IGNORE_OVERRIDE.update(I.parse_dq_ignore("WFC3:0"))
    assert I.INSTRUMENTS["WFC3_IR"].dq_policy()(np.array([16], dtype=np.int16)).tolist() == [False]


# ----------------------------------------------------------------------------- 2
@pytest.mark.skipif(not have_real, reason="real sample files absent")
def test_hst_conversion_formula():
    from astropy.io import fits
    h0, sh = fits.getheader(IR, 0), fits.getheader(IR, ("SCI", 1))
    g = lambda k: I._hget(sh, h0, k)  # noqa: E731
    a, how_a = I.to_sky_factor(g("BUNIT"), g("EXPTIME"), photfnu=g("PHOTFNU"), nominal_scale=sh["IDCSCALE"])
    b, how_b = I.to_sky_factor(g("BUNIT"), g("EXPTIME"), photflam=g("PHOTFLAM"), photplam=g("PHOTPLAM"), nominal_scale=sh["IDCSCALE"])
    assert how_a["photfnu_source"] == "PHOTFNU" and how_b["photfnu_source"] != "PHOTFNU"
    assert abs(a / b - 1) < 1e-3
    # by hand: PHOTFNU [Jy s / e-] x 1e-6 / (IDCSCALE^2 arcsec^2 in sr)
    assert abs(a - g("PHOTFNU") * 1e-6 / (sh["IDCSCALE"] ** 2 * (math.pi / 648000) ** 2)) < 1e-12 * a
    e, _ = I.to_sky_factor("ELECTRONS", 500.0, photfnu=1e-7, nominal_scale=0.05)
    r, _ = I.to_sky_factor("ELECTRONS/S", 500.0, photfnu=1e-7, nominal_scale=0.05)
    assert abs(e * 500.0 / r - 1) < 1e-12
    with pytest.raises(ValueError):
        I.to_sky_factor("COUNTS", 1.0, photfnu=1e-7, nominal_scale=0.05)


# ----------------------------------------------------------------------------- 3
def test_resolve_names_and_levels():
    assert I.resolve({"instrument": "NIRCAM", "detector": "NRCB1"}).key == "NIRCAM"
    assert I.resolve({"instrument": "WFC3", "detector": "IR"}).key == "WFC3_IR"
    assert I.resolve({"instrument": "ACS", "detector": "WFC"}).key == "ACS_WFC"
    assert I.resolve({"filter": "F150W"}).key == "NIRCAM"                       # entries without the field are NIRCam (as before)
    assert I.resolve("WFC3_IR").bandpass_id("F160W") == "HST/WFC3_IR.F160W" and data.svo_id("NIRCAM", "F090W") == "JWST/NIRCam.F090W"
    assert I.curve_cache_file("HST/WFC3_IR.F160W") == "filter_HST_WFC3_IR.F160W.json"
    assert data.INST_DIAM["NIRCAM"] == 6.5 and data.INST_DIAM["WFC3"] == 2.4 and data.INST_DIAM["ACS_WFC"] == 2.4
    with pytest.raises(SystemExit):
        I.resolve({"instrument": "IRAC"})
    with pytest.raises(SystemExit):
        I.resolve({"instrument": "WFC3", "detector": "UVIS"})                  # not described yet: refused, not read as the infrared channel


# ----------------------------------------------------------------------------- 4
def _fset(bounds, Q=3):
    return bandpass.FilterSet([bandpass.box_filter(n, "X", lo, hi, Q) for n, lo, hi in bounds])


def test_spectral_domain_unchanged_for_reference_filters():
    for bounds in ([("A1", 1.0, 1.3), ("A2", 1.6, 2.1), ("B1", 2.0, 2.8)],                     # the synthetic fields
                   [("F090W", 0.80, 1.00), ("F150W", 1.33, 1.67), ("F444W", 3.88, 4.98)]):      # NIRCam-like wide filters, Q = 3 and 8
        for Q in (3, 8):
            fs = _fset(bounds, Q)
            before = [f.curve_vec.copy() for f in fs.filters]
            cfg = SimpleNamespace(spec_k=12, spec_k_rule="span", spectral_domain="union")
            rec = I.spectral_setup(fs, cfg, log=lambda s: None)
            assert rec["domain_um"] == [0.70, 5.30] and rec["curve_grid_um"] == [0.55, 5.6, 48] and not rec["curve_grid_changed"]
            assert cfg.spec_k == 12 and sky.SPECTRAL_DOMAIN == [0.70, 5.30]
            assert all(np.array_equal(a, f.curve_vec) for a, f in zip(before, fs.filters))
    lam = torch.linspace(0.8, 5.0, 7)
    assert torch.equal(sky.lam_unit(lam), (torch.log(lam) - math.log(0.70)) / (math.log(5.30) - math.log(0.70)))


def test_spectral_domain_widened_for_blue_filter():
    fs = _fset([("F435W", 0.37, 0.47), ("F160W", 1.40, 1.70), ("F444W", 3.88, 4.98)], Q=8)
    assert fs.filters[0].curve_vec.max() == 0.0                       # the old grid starts at 0.55 um: an all-zero curve input (audit finding 4)
    cfg = SimpleNamespace(spec_k=12, spec_k_rule="span", spectral_domain="union")
    rec = I.spectral_setup(fs, cfg, log=lambda s: None)
    lo, hi = rec["domain_um"]
    assert lo < fs.filters[0].nodes.min() and hi == 5.30
    assert fs.filters[0].curve_vec.max() == 1.0 and rec["curve_grid_changed"] and rec["curve_grid_um"][2] > 48
    k_expect = 1 + round(11 * math.log(hi / lo) / math.log(5.30 / 0.70))
    assert cfg.spec_k == k_expect and k_expect > 12
    u = sky.lam_unit(torch.tensor(fs.filters[0].nodes))
    assert float(u.min()) > 0 and float(u.max()) < 1                  # the blue filter's nodes are inside the basis, not on the end bump
    phi = sky.RBFBasis(cfg.spec_k)(torch.tensor(np.concatenate([f.nodes for f in fs.filters]), dtype=torch.float32))
    assert torch.allclose(phi.sum(1), torch.ones(phi.shape[0]))
    cfg2 = SimpleNamespace(spec_k=12, spec_k_rule="fixed", spectral_domain="fixed")
    rec2 = I.spectral_setup(_fset([("F435W", 0.37, 0.47)]), cfg2, log=lambda s: None)
    assert rec2["domain_um"] == [0.70, 5.30] and cfg2.spec_k == 12


# ----------------------------------------------------------------------------- 5
NIRCAM_PIVOT = {"F090W": (0.9022, 0.0560)}       # stand-in only: pivot and rms width [um] (no curve cache on this machine)


def _stand_in(inst, filt, Q, cache):
    if inst.key == "NIRCAM":
        piv, bw = NIRCAM_PIVOT[filt]
    else:
        from astropy.io import fits
        h = fits.getheader(IR if inst.key == "WFC3_IR" else ACS, ("SCI", 1))
        h0 = fits.getheader(IR if inst.key == "WFC3_IR" else ACS, 0)
        piv, bw = I._hget(h, h0, "PHOTPLAM") / 1e4, I._hget(h, h0, "PHOTBW") / 1e4
    return I.stand_in_filter(filt, inst.key, piv, bw, Q)


def _splits(tmp, entries, box_size=20.48):
    sp = json.load(open("/home/claude/work/dl_hst/splits_SMACS.json"))
    box = dict(sp["boxes"][0], size_arcsec=box_size)
    names = [e["rootname"] for e in entries]
    out = {"boxes": [box], "exposures": entries, "splits": {"dither": {"train": names[1:], "heldout": names[:1]}}}
    p = os.path.join(tmp, "splits.json")
    json.dump(out, open(p, "w"))
    return p


def _entries():
    return [{"file": os.path.basename(JW), "rootname": "jw02736001001_02101_00009_nrcb1", "path": JW, "detector": "NRCB1", "filter": "F090W",
             "mjd": 59737.02, "instrument": "NIRCAM", "boxes": {"0": 0.07}, "dither_group": "jw|F090W"},
            {"file": os.path.basename(IR), "rootname": "iczgz2nsq", "path": IR, "detector": "IR", "filter": "F105W", "mjd": 57817.709,
             "instrument": "WFC3", "boxes": {"0": 1.0}, "dither_group": "iczgz2|F105W"},
            {"file": os.path.basename(ACS), "rootname": "jczgz3k8q", "path": ACS, "detector": "WFC", "filter": "F814W", "mjd": 57858.755,
             "instrument": "ACS", "boxes": {"0": 1.0}, "dither_group": "jczgz3|F814W"}]


@pytest.mark.skipif(not (have_real and have_tsky), reason="real sample files or torchsky absent (run with PYTHONPATH=/home/claude/torchsky/src)")
def test_real_files_through_load_field():
    from fieldop.model import resolve_instrument_options
    data.FILTER_CURVE = _stand_in
    with tempfile.TemporaryDirectory() as tmp:
        cfg = default_config(splits=_splits(tmp, _entries(), box_size=60.0), split="dither", n_lambda=4, data_margin=1.0)
        cuts, fset, meta, info = data.load_field(cfg, "cpu", log=lambda s: None)
        assert info["insts"] == ["ACS_WFC", "NIRCAM", "WFC3_IR"]
        assert fset.names == ["F814W", "F090W", "F105W"]                          # several instruments: ordered by pivot wavelength
        names = {e.name: e for e in cuts}
        assert {"jw02736001001_02101_00009_nrcb1", "iczgz2nsq", "jczgz3k8q_sci1", "jczgz3k8q_sci2"} <= set(names)
        assert len({e.eid for e in cuts}) == len(cuts)                              # one exposure per chip
        assert {info["dets"][names[n].det] for n in ("jczgz3k8q_sci1", "jczgz3k8q_sci2")} == {"ACS_WFC1", "ACS_WFC2"}
        assert names["jw02736001001_02101_00009_nrcb1"].split == "held" and names["iczgz2nsq"].split == "train"
        assert names["jw02736001001_02101_00009_nrcb1"].stripes and not names["iczgz2nsq"].stripes
        dia = dict(zip([e.name for e in sorted(cuts, key=lambda c: c.eid)], meta["diameter_m_e"]))
        assert dia["iczgz2nsq"] == 2.4 and dia["jw02736001001_02101_00009_nrcb1"] == 6.5
        # units: the WFC3/IR cut is the file's e-/s times PHOTFNU 1e-6 / Omega_nom (MJy/sr); the NIRCam cut is the file's MJy/sr unchanged
        from astropy.io import fits
        e = names["iczgz2nsq"]
        y0, x0 = e.cut0
        sci = fits.getdata(IR, ("SCI", 1)).astype(np.float32)[y0:y0 + e.shape[0], x0:x0 + e.shape[1]]
        f = e.extra["to_sky"]
        assert 0.25 < f < 0.35 and e.extra["sci_unit"] == "ELECTRONS/S"
        v = e.valid.numpy()
        assert np.allclose(e.d.numpy()[v], (sci * np.float32(f))[v], rtol=1e-6)
        j = names["jw02736001001_02101_00009_nrcb1"]
        y0, x0 = j.cut0
        assert np.array_equal(j.d.numpy()[j.valid.numpy()], fits.getdata(JW, "SCI")[y0:y0 + j.shape[0], x0:x0 + j.shape[1]][j.valid.numpy()])
        # the pixel scale of each cut from its header solution
        pix = {n: abs(np.linalg.det(np.asarray(names[n].extra["J"]))) ** 0.5 for n in ("iczgz2nsq", "jczgz3k8q_sci1", "jw02736001001_02101_00009_nrcb1")}
        assert abs(pix["iczgz2nsq"] - 0.128) < 0.003 and abs(pix["jczgz3k8q_sci1"] - 0.05) < 0.002 and abs(pix["jw02736001001_02101_00009_nrcb1"] - 0.031) < 0.001
        # the flux scale of a several-instrument fit needs an explicit reference instrument (audit finding 5)
        with pytest.raises(SystemExit):
            resolve_instrument_options(default_config(gauge_ref_inst=None, kernel_norm_radius=2.0), len(info["insts"]), log=lambda s: None)
        c2 = default_config(gauge_ref_inst=info["insts"].index("NIRCAM"), kernel_norm_radius=2.0)
        resolve_instrument_options(c2, len(info["insts"]), log=lambda s: None)
        assert c2.gauge_ref_inst == info["insts"].index("NIRCAM")
        # the spectral domain of this field: F814W's nodes are above 0.70 um, so the reference domain holds (union rule)
        rec = I.spectral_setup(fset, SimpleNamespace(spec_k=12, spec_k_rule="span", spectral_domain="union"), log=lambda s: None)
        assert rec["domain_um"][1] == 5.30 and rec["domain_um"][0] <= 0.70


# ----------------------------------------------------------------------------- 6
def test_two_instruments_command_line_runs_and_reference_needed():
    sys.path.insert(0, HERE)
    import synth_two_instrument_cli as S
    with tempfile.TemporaryDirectory() as d:
        r = S.run(os.path.dirname(HERE), d)
        assert len(r["state_sha256"]) == 64 and r["heldout_summary"]["written"]["chi2_input_noise"] > 0
        assert set(r["instrument_mean_log_throughput"]) == {"A", "B"}
        opts = list(S.OPTIONS)
        i = opts.index("--gauge-ref-inst")
        del opts[i:i + 2]
        keep, S.OPTIONS = S.OPTIONS, opts
        try:
            with pytest.raises(SystemExit):
                S.run(os.path.dirname(HERE), d)
        finally:
            S.OPTIONS = keep
