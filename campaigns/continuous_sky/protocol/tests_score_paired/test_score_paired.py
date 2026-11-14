#!/usr/bin/env python3
"""Tests of score_paired.py.   python3 -I tests/test_score_paired.py   (run from anywhere; needs tests/data from fetch_test_data.sh for tests 1 and 3).
Writes tests/out/test_results.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import score_paired as SP          # noqa: E402
S = SP.S
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
RESULTS = {}
quiet = lambda *a, **k: None


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    RESULTS.setdefault("checks", []).append(dict(name=name, passed=bool(cond), detail=detail))
    if not cond:
        RESULTS["failed"] = RESULTS.get("failed", 0) + 1


# ------------------------------------------------------------------ real-data loaders
class Args:
    root, field, split, data_root = DATA, "SMACS", "dither", os.path.join(DATA, "cal")


def real_loaders():
    sp, mask_for, pred_for, data_for = SP.make_loaders(Args)
    held = [r for r in sp["splits"]["dither"]["heldout"] if os.path.exists(os.path.join(DATA, "heldout", "SMACS", r + ".npz"))]
    sp2 = json.loads(json.dumps(sp))
    sp2["splits"]["dither"]["heldout"] = held
    return sp2, mask_for, pred_for, data_for, held


# ------------------------------------------------------------------ test 1: identical to score.py on real files
def test1():
    print("== test 1: real files, same numbers as score.py")
    sp, mask_for, pred_for, data_for, held = real_loaders()
    methods = ["fieldop_fix_200", "stage3_blot"]
    ex = {e["rootname"]: e for e in sp["exposures"]}
    n_ok, n_tot = 0, 0
    for r in held:
        mk, d = mask_for(r), data_for(r)
        for m in methods:
            pr = pred_for(m, r)
            rows, info = S.exposure_rows(d, pr["model"], mk, pr["model_valid"])
            st = SP.pixel_state(d, pr["model"], mk, pr["model_valid"])
            mine = SP.paired_rows(st, st, mk)                      # a method against itself: its own sums on its own pixels
            for rs, rm in zip(rows, mine):
                n_tot += 1
                same = (rs["box"] == rm["box"] and np.array_equal(rs["n"], rm["n"][:, :6]) and np.array_equal(rs["chi2"], rm["a"][:, :6])
                        and np.array_equal(rm["a"], rm["b"]) and not rm["d"].any())
                n_ok += bool(same)
            if st["b"] != info["offset"]:
                n_ok -= 1
    check("per-exposure sums per S/N bin (both offset variants) bit-identical to score.exposure_rows (%d rows)" % n_tot, n_ok == n_tot, "%d/%d" % (n_ok, n_tot))
    # headline with its interval through the whole pipeline
    cols = {}
    for m in methods:
        res = S.score_run(sp, m, "dither", mask_for, lambda r, m=m: pred_for(m, r), data_for)
        h = res["levels"]["field"]["all"]["with_offset"]["headline_chi2_snr_ref_ge5"]
        pr = SP.paired_run(sp, "dither", [m], m, mask_for, pred_for, data_for, log=quiet)
        hm = pr["methods"][m]["levels"]["field"]["all"]["with_offset"]["headline"]["score_method"]
        cols[m] = (h, hm)
        ok = np.allclose([h["value"], h["lo"], h["hi"]], [hm["value"], hm["lo68"], hm["hi68"]], rtol=1e-12, atol=0)
        check("headline chi2/N S/N>=5 and its 68%% interval for %s equal score.py: %.10g [%.10g, %.10g]" % (m, h["value"], h["lo"], h["hi"]), ok,
              "paired code: %.10g [%.10g, %.10g]" % (hm["value"], hm["lo68"], hm["hi68"]))
    RESULTS["test1_headline"] = {m: dict(score_py=c[0], score_paired=c[1]["value"]) for m, c in cols.items()}
    # paired pixel sets identical to each method's own set (nothing missing here)
    pr = SP.paired_run(sp, "dither", methods[:1], methods[1], mask_for, pred_for, data_for, log=quiet)
    f = pr["methods"][methods[0]]["levels"]["field"]["all"]["with_offset"]["headline"]
    check("paired score of each method (pairs on identical pixels) equals score.py's own value (no pixel dropped)",
          abs(f["score_method"]["value"] - cols[methods[0]][0]["value"]) < 1e-12 and abs(f["score_reference"]["value"] - cols[methods[1]][0]["value"]) < 1e-12
          and pr["methods"][methods[0]]["n_pixels_dropped_not_claimed_by_both"] == 0)
    check("pooled difference equals pixel-wise form", abs(f["diff"]["value"] - f["diff_pixelwise"]["value"]) < 1e-10,
          "%.10g vs %.10g" % (f["diff"]["value"], f["diff_pixelwise"]["value"]))
    # how close are the MAST _cal files to the protocol's own data? compare offsets with the published score file
    pub = json.load(open(os.path.join(os.path.dirname(HERE), "SMACS_fieldop_fix_200.json")))["per_exposure"]
    diffs = []
    for r in held:
        pr_ = pred_for("fieldop_fix_200", r)
        st = SP.pixel_state(data_for(r), pr_["model"], mask_for(r), pr_["model_valid"])
        diffs.append((r, st["b"], pub[r]["offset"], st["info"]["n_scored"], pub[r]["n_scored"]))
    RESULTS["test1_offsets_vs_published"] = diffs
    print("  offsets (this run with MAST _cal data, published): max |diff| = %.3g; n_scored identical: %s" % (
        max(abs(x[1] - x[2]) for x in diffs), all(x[3] == x[4] for x in diffs)))
    return sp, mask_for, pred_for, data_for


# ------------------------------------------------------------------ synthetic fields
def synth(models, nexp=8, shape=(128, 160), seed=0, holes=0.0, sigma_const=True, data_fn=None):
    """models: {name: fn(x, y, k)->model array}.  data = sigma * white noise.  Returns loaders."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    exps, masks, preds, datas = [], {}, {m: {} for m in models}, {}
    for k in range(nexp):
        root = "syn%02d" % k
        sig = np.ones(shape) if sigma_const else 0.8 + 0.4 * (0.5 + 0.5 * np.sin(x / 23.0 + k) * np.cos(y / 31.0))
        snr = np.where(rng.random(shape) < 0.25, 0.5, 20.0).astype(np.float32)
        valid = np.ones(shape, bool)
        if holes:
            valid &= rng.random(shape) > holes
        masks[root] = dict(valid=valid, snr_ref=snr, sigma=sig.astype(np.float32), noise_factor=np.ones(shape, np.float32), box_id=np.zeros(shape, np.int16))
        datas[root] = sig * rng.standard_normal(shape)
        for m, fn in models.items():
            preds[m][root] = dict(model=fn(x, y, k).astype(np.float32), model_valid=np.ones(shape, bool))
        exps.append(dict(rootname=root, filter="F150W", detector="NRCB1" if k % 2 else "NRCBLONG", dither_group="g%d" % k))
    sp = dict(field="SYN", exposures=exps, splits={"dither": {"heldout": [e["rootname"] for e in exps]}})
    return sp, (lambda r: masks[r]), (lambda m, r: preds[m][r]), (lambda r: datas[r])


def run_syn(models, ref, **kw):
    sp, mf, pf, df = synth(models, **kw)
    ms = [m for m in models if m != ref]
    pr = SP.paired_run(sp, "dither", ms, ref, mf, pf, df, nboot=300, log=quiet)
    bs = SP.byscale_run(sp, "dither", list(models), mf, pf, df, nboot=300, log=quiet)
    return pr, bs


def band_of(lam):
    return int(np.floor(np.log2(lam))) - 1


def test2():
    print("== test 2: synthetic")
    shape = (128, 160)
    A = 0.5
    for lam in (6.0, 20.0, 40.0, 60.0):
        pat = lambda x, y, k, lam=lam: A * np.sin(2 * np.pi * (0.8 * x + 0.6 * y) / lam)       # wavelength lam along the wave vector
        pr, bs = run_syn({"flat": lambda x, y, k: 0 * x, "patterned": pat}, ref="flat", shape=shape)
        h = pr["methods"]["patterned"]["levels"]["field"]["all"]["with_offset"]["headline"]
        exp = A * A / 2
        check("injected pattern lambda=%g px: paired diff = A^2/2 = %.3f (got %.4f [%.4f, %.4f]), interval excludes 0, P(better)=0" % (lam, exp, h["diff"]["value"], h["diff"]["lo68"], h["diff"]["hi68"]),
              abs(h["diff"]["value"] - exp) < 0.02 and h["diff"]["lo68"] > 0 and h["diff"]["frac_better"] == 0.0)
        lv = bs["levels"]["all"]
        dp = np.array([np.nan if v is None else v for v in lv["diff"]["flat|patterned"]["value"]])
        pk = int(np.nanargmax(dp))
        others = np.delete(dp, pk)
        # expected band power: total power N*A^2/2 shared over the modes of the band (modes counted for this shape)
        bm = SP.band_map(*shape)
        nmodes = (bm == band_of(lam)).sum()
        exp_p = shape[0] * shape[1] * A * A / 2 / nmodes
        w_ok = np.nanmax(np.where(np.arange(len(dp)) != pk, dp, np.nan)) < 0.1 * dp[pk] + 1.2 * 0   # others are small relative to peak (noise-free)
        if lam <= 20:
            check("difference-power peak at band %g-%g px for lambda=%g px; peak %.3g (expected ~%.3g), all other bands < 10%% of the peak" % (
                SP.BAND_LO[pk], SP.BAND_HI[pk], lam, dp[pk], exp_p), pk == band_of(lam) and abs(dp[pk] / exp_p - 1) < 0.2 and w_ok)
        else:   # wavelength is a third or more of the 128 x 160 cut-out: only 2-4 cycles fit, the taper spreads the line into the neighbouring (few-mode) bands
            check("KNOWN LIMIT lambda=%g px is %.0f%% of the cut-out size: peak band %g-%g px is within one band of the true band %g-%g px (peak %.3g, naive expectation %.3g)" % (
                lam, 100 * lam / 160, SP.BAND_LO[pk], SP.BAND_HI[pk], SP.BAND_LO[band_of(lam)], SP.BAND_HI[band_of(lam)], dp[pk], exp_p), abs(pk - band_of(lam)) <= 1)
        RESULTS["synthetic_lambda_%g" % lam] = dict(diff=h["diff"], peak_band=[SP.BAND_LO[pk], SP.BAND_HI[pk]], peak_power=dp[pk], expected_peak=exp_p)
        ident = np.allclose(np.array([x or 0 for x in lv["diff"]["flat|patterned"]["value"]]),
                            np.array([x or 0 for x in lv["auto"]["flat"]["value"]]) + np.array([x or 0 for x in lv["auto"]["patterned"]["value"]])
                            - 2 * np.array([x or 0 for x in lv["cross"]["flat|patterned"]["value"]]), rtol=1e-6, atol=1e-9)
        check("identity diff = auto_1 + auto_2 - 2 cross (lambda=%g)" % lam, ident)
    # identical models
    same = lambda x, y, k: 0.1 * np.sin(x / 9.0) * np.cos(y / 7.0)
    pr, bs = run_syn({"a": same, "b": same}, ref="a")
    h = pr["methods"]["b"]["levels"]["field"]["all"]["with_offset"]["headline"]
    zero = all(h["diff"][k] == 0.0 for k in ("value", "lo68", "hi68", "lo95", "hi95")) and h["diff"]["frac_equal"] == 1.0 and h["diff"]["frac_better"] == 0.0
    b0 = all(all(v == 0.0 for v in bn["diff"]["value"]) and bn["diff"]["lo68"] == bn["diff"]["hi68"] == 0.0 for bn in
             pr["methods"]["b"]["levels"]["field"]["all"]["with_offset"]["bins"] if bn["diff"]["value"] is not None) if False else True
    bins0 = all(bn["diff"]["value"] in (0.0, None) and bn["diff"]["lo68"] in (0.0, None) and bn["diff"]["hi68"] in (0.0, None)
                for bn in pr["methods"]["b"]["levels"]["field"]["all"]["with_offset"]["bins"])
    dz = np.nanmax(np.abs(np.array([np.nan if v is None else v for v in bs["levels"]["all"]["diff"]["a|b"]["value"]])))
    check("identical models: paired difference exactly 0 with zero-width 68% and 95% intervals (pooled and per bin), difference power exactly 0", zero and bins0 and dz == 0.0,
          "diff=%r interval [%r, %r]" % (h["diff"]["value"], h["diff"]["lo68"], h["diff"]["hi68"]))
    # white noise, with and without holes
    for holes in (0.0, 0.3):
        pr, bs = run_syn({"zero1": lambda x, y, k: 0 * x, "zero2": lambda x, y, k: 0 * x + 0.0}, ref="zero1", shape=(192, 192), holes=holes, nexp=40)
        lv = bs["levels"]["all"]
        v = np.array([np.nan if x is None else x for x in lv["auto"]["zero1"]["value"]])
        n = np.array(lv["n_cutouts_per_band"]) > 0
        nm = np.array([(SP.band_map(192, 192) == j).sum() for j in range(SP.NBANDS)])[n]
        tol = 4.0 / np.sqrt(0.5 * nm * 40)                      # 4 sigma of the scatter of a mean over (modes x 40 exposures), halved for taper/hole correlation
        flat = np.all(np.abs(v[n] - 1) < tol)
        c = np.array([np.nan if x is None else x for x in lv["cross"]["zero1|zero2"]["value"]])[n]
        check("white noise (%.0f%% of pixels masked at random, 40 exposures): residual power = 1 in every band within 4 sigma of its scatter (range %.3f-%.3f; lowest band has %d modes); cross-power of two identical models also ~1 (noise is common)" % (
            100 * holes, np.nanmin(v[n]), np.nanmax(v[n]), nm[-1]), flat and np.all(np.abs(c - 1) < tol))
        RESULTS["white_holes_%g" % holes] = [float(x) for x in v[n]]
    # red-noise / mask caveat: a pattern with 30% holes keeps its peak but the peak power is biased
    pat = lambda x, y, k: 0.5 * np.sin(2 * np.pi * (0.8 * x + 0.6 * y) / 20.0)
    pr, bs = run_syn({"flat": lambda x, y, k: 0 * x, "patterned": pat}, ref="flat", shape=(128, 160), holes=0.3)
    dp = np.array([np.nan if v is None else v for v in bs["levels"]["all"]["diff"]["flat|patterned"]["value"]])
    nm = (SP.band_map(128, 160) == band_of(20)).sum()
    check("pattern lambda=20 with 30%% random holes: peak still in the right band; mask-correction bias of the peak power %.0f%% (known, see report)" % (
        100 * (dp[band_of(20)] / (128 * 160 * 0.125 / nm) - 1)), int(np.nanargmax(dp)) == band_of(20))
    RESULTS["synthetic_holes_peak_bias"] = float(dp[band_of(20)] / (128 * 160 * 0.125 / nm) - 1)


# ------------------------------------------------------------------ test 3 + real-sample results
def test3(loaders):
    print("== test 3: paired interval versus separate intervals, real sample")
    sp, mask_for, pred_for, data_for = loaders
    pr = SP.paired_run(sp, "dither", ["fieldop_fix_200"], "stage3_blot", mask_for, pred_for, data_for, log=quiet)
    h = pr["methods"]["fieldop_fix_200"]["levels"]["field"]["all"]["with_offset"]["headline"]
    wp, wm, wr, wi = h["width68_paired_diff"], h["width68_method_alone"], h["width68_reference_alone"], h["width68_if_independent"]
    check("paired 68%% interval (width %.3f) narrower than the separate intervals (method %.3f, reference %.3f; quadrature sum %.3f): factor %.2f narrower than quadrature sum" % (
        wp, wm, wr, wi, wi / wp), wp < min(wm, wr, wi))
    RESULTS["test3"] = dict(width_paired=wp, width_method=wm, width_reference=wr, width_quadrature=wi, ratio_to_quadrature=wi / wp,
                            diff=h["diff"], score_method=h["score_method"], score_reference=h["score_reference"], pooled_pixels=h["n_pixels"])
    json.dump(pr, open(os.path.join(OUT, "real_sample_paired.json"), "w"), indent=1)
    open(os.path.join(OUT, "real_sample_paired.md"), "w").write(SP.paired_markdown(pr))
    bs = SP.byscale_run(sp, "dither", ["fieldop_fix_200", "stage3_blot"], mask_for, pred_for, data_for, nboot=200, log=quiet)
    json.dump(bs, open(os.path.join(OUT, "real_sample_byscale.json"), "w"), indent=1)
    open(os.path.join(OUT, "real_sample_byscale.md"), "w").write(SP.byscale_markdown(bs))
    SP.byscale_figures(bs, OUT)
    worst = 0.0
    for c in bs["per_cutout"]:
        a1, a2, cr, df = (np.array([np.nan if v is None else v for v in x]) for x in (c["auto"]["fieldop_fix_200"], c["auto"]["stage3_blot"],
                                                                                    c["cross"]["fieldop_fix_200|stage3_blot"], c["diff"]["fieldop_fix_200|stage3_blot"]))
        worst = max(worst, np.nanmax(np.abs(df - (a1 + a2 - 2 * cr)) / np.nanmax(np.abs(df))))
    check("real data: diff power = auto_1 + auto_2 - 2 cross for every cut-out (max relative deviation %.2g)" % worst, worst < 1e-8)
    RESULTS["real_byscale_identity_max_rel_dev"] = worst


if __name__ == "__main__":
    loaders = test1()
    test2()
    test3(loaders)
    json.dump(RESULTS, open(os.path.join(OUT, "test_results.json"), "w"), indent=1, default=float)
    print("FAILED %d" % RESULTS.get("failed", 0) if RESULTS.get("failed") else "ALL PASSED")
    sys.exit(1 if RESULTS.get("failed") else 0)
