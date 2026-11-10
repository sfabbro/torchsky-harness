"""CPU tests of protocol/score.py with synthetic data (no JWST data needed).

    /tmp/.../venvT/bin/python -m pytest -q /home/claude/work/tests/test_protocol_score.py     (or run this file directly)

Checks: a perfect model gives chi2/N = 1 in every S/N bin (also with a constant sky offset, which the scorer removes);
a 5% flux error gives chi2/N = 1 + <(f S/sigma)^2> per bin and fractional residual rms ~ f/(1+f); missing model_valid / missing
prediction files / unclaimed pixels are reported; bootstrap intervals bracket the value; the CLI end to end writes scores + comparison.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(os.path.dirname(HERE), "protocol")
sys.path.insert(0, PROTO)
import score  # noqa: E402

N = 128
FILTERS = ["F150W", "F200W"]
DETS = ["NRCB1", "NRCB2"]
GROUPS = 3          # dither groups per filter
SIGMA0 = 1.0


def make_world(seed=1, offset=0.0, field="TEST"):
    """Truth S with log-uniform S/N in 0.1..1000 (sigma = 1): returns splits dict, per-root truth/noisy data, masks."""
    rng = np.random.default_rng(seed)
    exposures, truth, data, masks = [], {}, {}, {}
    for f in FILTERS:
        for g in range(GROUPS):
            for d in DETS:
                root = f"jwTEST_{g:05d}_{d.lower()}_{f}"
                S = 10 ** rng.uniform(-1, 3, (N, N))
                sigma = np.full((N, N), SIGMA0)
                obs = S + sigma * rng.standard_normal((N, N)) + offset
                valid = np.ones((N, N), bool)
                valid[:4] = False
                masks[root] = dict(valid=valid, snr_ref=(S / sigma).astype(np.float32), sigma=sigma.astype(np.float32),
                                   noise_factor=np.ones((N, N), np.float32), box_id=np.zeros((N, N), np.int16))
                truth[root], data[root] = S, obs
                exposures.append(dict(rootname=root, file=root + "_cal.fits", path="", filter=f, detector=d, dither_group=f"grp{g}|{f}"))
    splits = dict(field=field, exposures=exposures, splits={"dither": dict(heldout=[e["rootname"] for e in exposures], train=[])})
    return splits, truth, data, masks


def run(splits, data, masks, preds, nboot=200):
    return score.score_run(splits, "m", "dither", lambda r: masks[r], lambda r: preds.get(r), lambda r: data[r], nboot)


def bins_of(res, variant, level="all"):
    return res["levels"]["field"]["all"][variant]


def test_perfect_model_is_one_in_every_bin():
    splits, S, data, masks = make_world(offset=0.0)
    preds = {r: dict(model=S[r].astype(np.float32)) for r in S}
    res = run(splits, data, masks, preds)
    for variant in ("with_offset", "without_offset"):
        agg = bins_of(res, variant)
        for b in agg["bins"]:
            assert b["n_pixels"] > 5000, b
            assert abs(b["chi2_per_pixel"] - 1.0) < 0.03, (variant, b)
            assert b["lo"] <= b["chi2_per_pixel"] <= b["hi"]
        h = agg["headline_chi2_snr_ref_ge5"]
        assert abs(h["value"] - 1.0) < 0.02 and h["lo"] < h["hi"]
    assert res["pixels"]["missing_fraction"] == 0.0
    assert not res["missing_prediction_files"]


def test_constant_offset_is_removed_by_scorer():
    splits, S, data, masks = make_world(offset=0.5)
    preds = {r: dict(model=S[r].astype(np.float32), model_valid=np.ones((N, N), bool)) for r in S}
    res = run(splits, data, masks, preds)
    w, wo = bins_of(res, "with_offset"), bins_of(res, "without_offset")
    for b in w["bins"]:
        assert abs(b["chi2_per_pixel"] - 1.0) < 0.03, b
    assert wo["bins"][0]["chi2_per_pixel"] > 1.15          # 1 + 0.5^2 without the offset
    off = [i["offset"] for i in res["per_exposure"].values()]
    assert abs(np.mean(off) - 0.5) < 0.03


def test_offset_fallback_when_no_pixel_has_snr_below_one():
    # bright field: every pixel has snr_ref >= 2 (like the SMACS cluster core); offset must still be found from the lowest-S/N pixels
    splits, S, data, masks = make_world(offset=0.5, seed=7)
    for r in S:
        masks[r]["snr_ref"] = np.maximum(masks[r]["snr_ref"], 2.0).astype(np.float32)
    res = run(splits, data, masks, {r: dict(model=S[r].astype(np.float32)) for r in S})
    assert all("offset_flag" in i for i in res["per_exposure"].values())
    assert abs(np.mean([i["offset"] for i in res["per_exposure"].values()]) - 0.5) < 0.05
    assert abs(bins_of(res, "with_offset")["headline_chi2_snr_ref_ge5"]["value"] - 1.0) < 0.04


def test_five_percent_flux_error_rises_as_fS_squared():
    f = 0.05
    splits, S, data, masks = make_world(offset=0.0, seed=3)
    preds = {r: dict(model=((1 + f) * S[r]).astype(np.float32)) for r in S}
    res = run(splits, data, masks, preds)
    snr_all = np.concatenate([masks[r]["snr_ref"][masks[r]["valid"]] for r in S]).astype(float)
    edges = score.EDGES
    agg = bins_of(res, "without_offset")
    for k, b in enumerate(agg["bins"]):
        sel = (snr_all >= edges[k]) & (snr_all < edges[k + 1])
        # d - m = -f S + noise  ->  E[chi2] = 1 + (f S/sigma)^2
        expected = 1.0 + np.mean((f * snr_all[sel]) ** 2)
        assert abs(b["chi2_per_pixel"] / expected - 1.0) < 0.04, (b, expected)
    # the high-S/N bins are dominated by the flux error: (f S)^2 >> 1
    assert agg["bins"][5]["chi2_per_pixel"] > 100 * agg["bins"][0]["chi2_per_pixel"]
    fr = agg["frac_resid_rms_snr_ref_gt50"]["value"]
    hi = snr_all[snr_all > 50]
    expected_fr = np.sqrt(np.mean((f / (1 + f)) ** 2 + (1.0 / ((1 + f) * hi)) ** 2))
    assert abs(fr / expected_fr - 1.0) < 0.05, (fr, expected_fr)


def test_negative_flux_error_gives_the_same_rise():
    f = 0.05
    splits, S, data, masks = make_world(offset=0.0, seed=4)
    up = run(splits, data, masks, {r: dict(model=((1 + f) * S[r]).astype(np.float32)) for r in S})
    dn = run(splits, data, masks, {r: dict(model=((1 - f) * S[r]).astype(np.float32)) for r in S})
    a, b = bins_of(up, "without_offset")["bins"][5], bins_of(dn, "without_offset")["bins"][5]
    assert abs(a["chi2_per_pixel"] / b["chi2_per_pixel"] - 1) < 0.1


def test_missing_model_valid_and_files_are_reported():
    splits, S, data, masks = make_world(seed=5)
    roots = list(S)
    preds = {}
    for i, r in enumerate(roots):
        p = dict(model=S[r].astype(np.float32))
        if i % 2 == 0:                                    # half of the exposures give model_valid; one block unclaimed
            mv = np.ones((N, N), bool)
            mv[:, : N // 2] = False
            p["model_valid"] = mv
        preds[r] = p
    dropped = roots[-1]
    del preds[dropped]                                    # one prediction file missing altogether
    nan_root = roots[1]
    preds[nan_root]["model"][10:20, 10:20] = np.nan       # non-finite model pixels count as unclaimed
    res = run(splits, data, masks, preds)
    assert res["missing_prediction_files"] == [dropped]
    assert res["n_scored_exposures"] == len(roots) - 1
    with_mv = [r for r, i in res["per_exposure"].items() if i["model_valid_provided"]]
    without = res["pixels"]["exposures_without_model_valid"]
    assert set(without) == set(res["per_exposure"]) - set(with_mv) and len(without) > 0
    r0 = roots[0]
    i0 = res["per_exposure"][r0]
    assert i0["n_missing"] == (N - 4) * (N // 2)          # valid rows 4.., left half unclaimed
    assert i0["n_scored"] + i0["n_missing"] == i0["n_mask_valid"]
    assert res["per_exposure"][nan_root]["n_missing"] == 100
    assert 0.1 < res["pixels"]["missing_fraction"] < 0.4
    # what is not claimed is not scored: chi2 stays ~1 even though half of some exposures is missing
    assert abs(bins_of(res, "with_offset")["headline_chi2_snr_ref_ge5"]["value"] - 1.0) < 0.03


def test_cli_end_to_end_and_comparison_table():
    from astropy.io import fits
    splits, S, data, masks = make_world(seed=6)
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "heldout", "TEST"))
        calroot = os.path.join(td, "cal")
        os.makedirs(calroot)
        for e in splits["exposures"]:
            r = e["rootname"]
            e["path"] = os.path.join(calroot, e["file"])
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data[r].astype(np.float32), name="SCI")]).writeto(e["path"])
            np.savez(os.path.join(td, "heldout", "TEST", r + ".npz"), **masks[r])
        json.dump(splits, open(os.path.join(td, "splits_TEST.json"), "w"))
        for meth, scale in (("perfect", 1.0), ("flux5", 1.05), ("zero", 0.0)):
            pd = os.path.join(td, "pred", "TEST", meth, "dither")
            os.makedirs(pd)
            for r in S:
                np.savez(os.path.join(pd, r + ".npz"), model=(scale * S[r]).astype(np.float32), meta=json.dumps({"fitted": "none"}))
            out = subprocess.run([sys.executable, os.path.join(PROTO, "score.py"), "score", "--field", "TEST", "--method", meth,
                                  "--split", "dither", "--root", td, "--nboot", "100"], capture_output=True, text=True)
            assert out.returncode == 0, out.stderr[-2000:]
            assert os.path.exists(os.path.join(td, "scores", "TEST", meth, "dither.json"))
        md = open(os.path.join(td, "scores", "TEST", "comparison_dither.md")).read()
        for meth in ("perfect", "flux5", "zero"):
            assert f"| {meth} |" in md
        tab = json.load(open(os.path.join(td, "scores", "TEST", "comparison_dither.json")))
        head = {t["method"]: t["headline"]["value"] for t in tab if t["variant"] == "with_offset"}
        assert head["perfect"] < head["flux5"] < head["zero"]
        sc = json.load(open(os.path.join(td, "scores", "TEST", "zero", "dither.json")))
        assert sc["method_meta"] and sc["levels"]["filter"]["F150W"]["with_offset"]["headline_chi2_snr_ref_ge5"]["n_pixels"] > 0
        assert set(sc["levels"]) >= {"field", "filter", "detector", "box", "filter_box"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
