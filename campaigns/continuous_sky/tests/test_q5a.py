"""Q5a: name parser, test/validation selection, per-exposure bootstrap (no data needed). Run: python3 test_q5a.py"""
import sys, math, random
import numpy as np, torch
sys.path.insert(0, "/home/claude/work/scripts")
import smacs_four_v22 as S

BANDS = S.BANDS
SW, LW = BANDS[:3], BANDS[3:]


def mkname(idx, grp, exp, det, prefix=True, vis="001001"):
    core = f"jw02736{vis}_{grp}_{exp:05d}_{det}_o001_crf_rshift_tweakreg.fits"
    return (f"{idx:03d}_" + core) if prefix else core


# ---------------------------------------------------------------- (c) parser
def test_parser():
    p = S.parse_exposure_name("034_jw02736001001_02101_00009_nrcb3_o001_crf_rshift_tweakreg.fits")
    assert p["prog"] == "02736" and p["obs"] == "001" and p["vis"] == "001" and p["grp"] == "02101" and p["exp"] == "00009"
    assert p["det"] == "nrcb3" and p["idx"] == 34 and p["visit"] == ("001001", "02101") and p["dither"] == (("001001", "02101"), "00009")
    q = S.parse_exposure_name("jw02736001001_02103_00003_nrcblong_o001_crf.fits")  # no index prefix, LW detector
    assert q["det"] == "nrcblong" and q["idx"] is None and q["visit"] == ("001001", "02103")
    # detectors of one exposure share a dither key; different exposure / group / visit do not
    a, b = mkname(0, "02101", 1, "nrcb1"), mkname(1, "02101", 1, "nrcb4")
    assert S.dither_key(a) == S.dither_key(b)
    assert S.dither_key(a) != S.dither_key(mkname(4, "02101", 2, "nrcb1"))
    assert S.dither_key(a) != S.dither_key(mkname(0, "02103", 1, "nrcb1"))
    assert S.visit_key(mkname(0, "02101", 1, "nrcb1", vis="002001")) != S.visit_key(a)
    assert S.visit_key(a) == S.visit_key(mkname(9, "02101", 7, "nrcblong"))
    # path, upper case, a non-matching name: robust, no exception
    assert S.parse_exposure_name("/data/x/" + a)["exp"] == "00001"
    assert S.parse_exposure_name("JW02736001001_02101_00001_NRCB2_cal.fits")["det"] == "nrcb2"
    for bad in ("", "foo.fits", "jw123_x.fits", "001_something_nrcb1.fits"):
        assert S.parse_exposure_name(bad) is None
        assert S.dither_key(bad) == (("?", "?"), bad) and S.visit_key(bad) == ("?", "?")
    print("parser ok")


# ---------------------------------------------------------------- (d) selection
def synth(visits_per_band=3, n_dith=9, seed=0, all_det_equal=False):
    """135-like exposure list: per band `visits_per_band` visit groups x n_dith dithers; SW 4 detectors, LW 1."""
    rng = random.Random(seed)
    exps, idx = [], 0
    for bi, b in enumerate(BANDS):
        for v in range(visits_per_band):
            grp = f"02{101 + 2 * (3 * bi + v):03d}"[:5]
            for d in range(1, n_dith + 1):
                for det in (["nrcb1", "nrcb2", "nrcb3", "nrcb4"] if b in SW else ["nrcblong"]):
                    n = 5000 if all_det_equal else rng.randint(2000, 9000)
                    exps.append({"name": mkname(idx, grp, d, det), "band": b, "d": torch.zeros(n)})
                    idx += 1
    return exps


def old_select(exps, n_val):
    for e in exps:
        e["holdout"] = False
    for b in BANDS:
        bl = [e for e in exps if e["band"] == b]
        if len(bl) >= 3:
            max(bl, key=lambda e: e["d"].numel())["holdout"] = True
    for e in exps:
        e["val"] = False
    for _ in range(n_val):
        for b in BANDS:
            bl = [e for e in exps if e["band"] == b and not e["holdout"] and not e["val"]]
            if len(bl) >= 4:
                max(bl, key=lambda e: e["d"].numel())["val"] = True
    return [e["name"] for e in exps if e["holdout"]], [e["name"] for e in exps if e["val"]]


def test_selection():
    for nval in (0, 1, 2):
        ex = synth(seed=nval)
        oh, ov = old_select(ex, nval)
        ex2 = synth(seed=nval)
        r = S.select_exposures(ex2, n_test=1, n_val=nval)
        assert r["holdout"] == oh and r["validation"] == ov and len(oh) == 6, (nval,)
    # N=1 with sparse bands (<3 images) -> nothing held out for that band, as before
    ex = synth(n_dith=1, visits_per_band=1)
    ex = [e for e in ex if e["band"] not in LW] + [e for e in ex if e["band"] in LW][:2]
    oh, _ = old_select(ex, 0)
    assert S.select_exposures(ex, 1, 0)["holdout"] == oh
    # --holdout-from names are used as given, n_test ignored
    ex = synth()
    names = [ex[3]["name"], ex[50]["name"]]
    r = S.select_exposures(ex, n_test=3, n_val=0, holdout_names=names)
    assert r["holdout"] == names
    r = S.select_exposures(ex, n_test=3, n_val=1, no_holdout=True)
    assert r["holdout"] == [] and r["validation"] == []

    # N=3: 3 dithers per band, from 3 different visits, all detectors of each dither
    ex = synth()
    r = S.select_exposures(ex, n_test=3, n_val=1)
    for b in BANDS:
        h = [e for e in ex if e["band"] == b and e["holdout"]]
        dk = {S.dither_key(e["name"]) for e in h}
        assert len(dk) == 3, (b, len(dk))
        assert len({k[0] for k in dk}) == 3, b  # 3 different visits
        want = 4 if b in SW else 1
        assert len(h) == 3 * want
        for k in dk:  # whole dither held out
            assert all(e["holdout"] for e in ex if e["band"] == b and S.dither_key(e["name"]) == k)
        # each chosen dither is the most-pixel dither of its visit
        for v in {k[0] for k in dk}:
            px = {}
            for e in ex:
                if e["band"] == b and S.visit_key(e["name"]) == v:
                    px[S.dither_key(e["name"])] = px.get(S.dither_key(e["name"]), 0) + e["d"].numel()
            best = max(px, key=lambda k: (px[k], tuple(reversed(k))))
            assert best in dk or px[best] == max(px[k] for k in dk if k[0] == v)
        v_ = [e for e in ex if e["band"] == b and e["val"]]
        assert len(v_) == 1 and not v_[0]["holdout"]  # validation drawn from what is left
    assert set(r["holdout"]).isdisjoint(r["validation"])
    # deterministic
    ex_b = synth()
    assert S.select_exposures(ex_b, 3, 1)["holdout"] == r["holdout"]
    # N=4 with 3 visits: round-robin wraps (2,1,1) dithers; N=2: two different visits
    ex = synth(); S.select_exposures(ex, 2, 0)
    for b in BANDS:
        dk = {S.dither_key(e["name"]) for e in ex if e["band"] == b and e["holdout"]}
        assert len(dk) == 2 and len({k[0] for k in dk}) == 2
    ex = synth(); S.select_exposures(ex, 4, 0)
    for b in BANDS:
        dk = [S.dither_key(e["name"]) for e in ex if e["band"] == b and e["holdout"]]
        cnt = {}
        for k in set(dk):
            cnt[k[0]] = cnt.get(k[0], 0) + 1
        assert sorted(cnt.values()) == [1, 1, 2]
    # one visit per band (the real data layout): dithers are taken by pixel count, no crash
    ex = synth(visits_per_band=1)
    r = S.select_exposures(ex, 3, 0)
    for b in BANDS:
        assert len({S.dither_key(e["name"]) for e in ex if e["band"] == b and e["holdout"]}) == 3
    # too few dithers: keeps >= 2 for training and warns
    ex = synth(visits_per_band=1, n_dith=3)
    r = S.select_exposures(ex, 3, 0)
    assert r["warnings"] and all(len({S.dither_key(e["name"]) for e in ex if e["band"] == b and e["holdout"]}) == 1 for b in BANDS)
    print("selection ok")


# ---------------------------------------------------------------- bootstrap
def synth_records(rng, nexp_per_band, bands=BANDS, npull=40):
    recs = []
    for bi, b in enumerate(bands):
        for j in range(nexp_per_band):
            n = int(rng.integers(20000, 60000)); ns = int(rng.integers(500, 3000))
            recs.append({"name": mkname(len(recs), f"02{101 + 2 * bi}", j + 1, "nrcb1"), "band": b, "det": 0, "n": n, "n_src": ns,
                         "chi2_sum": n * rng.normal(1.5, 0.3), "chi2_src_sum": ns * rng.normal(2.5, 0.8),
                         "robust_sum": n * rng.normal(1.2, 0.2), "robust_src_sum": ns * rng.normal(2.0, 0.6),
                         "pulls": list(rng.normal(0, 1.4, npull))})
    return recs


def direct_pool(recs):
    n = sum(r["n"] for r in recs); ns = sum(r["n_src"] for r in recs)
    v = torch.tensor([x for r in recs for x in r["pulls"]], dtype=torch.float32)
    return {"chi2_all": sum(r["chi2_sum"] for r in recs) / n, "chi2_src": sum(r["chi2_src_sum"] for r in recs) / ns,
            "robust_all": sum(r["robust_sum"] for r in recs) / n, "robust_src": sum(r["robust_src_sum"] for r in recs) / ns,
            "flux_err": float(1.4826 * (v - v.median()).abs().median())}


def test_bootstrap_point():
    rng = np.random.default_rng(3)
    recs = synth_records(rng, 4)
    bs = S.bootstrap_scores(recs, B=200, seed=0)
    d = direct_pool(recs)
    for k, v in d.items():
        assert abs(bs[k]["est"] - v) < 1e-12 * max(1, abs(v)), (k, bs[k]["est"], v)
        assert bs[k]["p2.5"] <= bs[k]["p16"] <= bs[k]["p84"]
        assert bs[k]["p2.5"] <= bs[k]["p16"] <= bs[k]["p84"] <= bs[k]["p97.5"]
    assert bs["single_exposure_bands"] == [] and bs["n_exposures"]["F090W"] == 4 and bs["n_dithers"]["F090W"] == 4
    # reproducible for a fixed seed, different for another
    assert S.bootstrap_scores(recs, B=200, seed=0)["chi2_src"] == bs["chi2_src"]
    assert S.bootstrap_scores(recs, B=200, seed=1)["chi2_src"] != bs["chi2_src"]
    # a single-exposure band is flagged, and a band without any resampling variance adds none
    one = synth_records(rng, 1, bands=["F090W"]) + synth_records(rng, 5, bands=["F150W"])
    b1 = S.bootstrap_scores(one, B=100)
    assert b1["single_exposure_bands"] == ["F090W"] and b1["single_dither_bands"] == ["F090W"]
    assert "SINGLE-DITHER" in S.boot_line("m", b1) and b1["n_dithers"] == {"F090W": 1, "F150W": 5}
    # a band with 4 detector images of ONE dither: not a single exposure, but a single dither
    four = [dict(r, name=mkname(i, "02101", 1, f"nrcb{i + 1}")) for i, r in enumerate(synth_records(rng, 4, bands=["F090W"]))]
    b4 = S.bootstrap_scores(four, B=100)
    assert b4["single_dither_bands"] == ["F090W"] and b4["single_exposure_bands"] == [] and b4["n_dithers"]["F090W"] == 1
    assert b4["chi2_src"]["p84"] == b4["chi2_src"]["p16"]  # one cluster: replicates are identical, variance not seen
    # no source pixels at all -> NaN, no crash
    ns0 = [dict(r, n_src=0, chi2_src_sum=0.0, robust_src_sum=0.0) for r in recs]
    assert math.isnan(S.bootstrap_scores(ns0, B=20)["chi2_src"]["est"])
    print("bootstrap point estimate ok")


def test_cluster_wider():
    """4 detectors per dither, perfectly correlated within a dither: cluster interval wider than the per-image interval."""
    rng = np.random.default_rng(7)
    recs = []
    for bi, b in enumerate(BANDS[:3]):
        for dI in range(8):
            base = dict(n=40000, n_src=2000, chi2_sum=40000 * rng.normal(1.5, 0.4), chi2_src_sum=2000 * rng.normal(2.5, 0.8),
                        robust_sum=40000 * rng.normal(1.2, 0.3), robust_src_sum=2000 * rng.normal(2.0, 0.6))
            base["chi2_sum"] = 40000 * (1.5 + 0.4 * rng.normal()); 
            for det in range(4):
                recs.append(dict(base, name=mkname(len(recs), f"02{101 + 2 * bi}", dI + 1, f"nrcb{det + 1}"), band=b, det=det,
                                 pulls=list(np.full(10, 0.0) + 1.4 * rng.normal())))  # identical-ish across a dither is enough for sums
    # make pulls identical across detectors of a dither too
    for i in range(0, len(recs), 4):
        for k in range(1, 4):
            recs[i + k]["pulls"] = list(recs[i]["pulls"])
    c = S.bootstrap_scores(recs, B=1500, seed=0, cluster="dither")
    im = S.bootstrap_scores(recs, B=1500, seed=0, cluster="image")
    assert c["n_dithers"] == {b: 8 for b in BANDS[:3]} and im["n_exposures"] == {b: 32 for b in BANDS[:3]}
    for k in S._BOOT_STATS:
        assert abs(c[k]["est"] - im[k]["est"]) < 1e-12
    for k in ("chi2_all", "chi2_src", "robust_all", "robust_src"):
        wc, wi = c[k]["p84"] - c[k]["p16"], im[k]["p84"] - im[k]["p16"]
        print(f"  {k:11s} cluster width {wc:.4f} vs per-image width {wi:.4f} (ratio {wc / wi:.2f}, expected ~2)")
        assert wc > 1.4 * wi
    print("cluster wider than per-image ok")


def test_bootstrap_scaling():
    """interval width ~ 1/sqrt(n_exposures): iid exposures from one distribution, n and 4n exposures per band."""
    ratios = {k: [] for k in S._BOOT_STATS}
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        w = {}
        for m in (6, 24):
            bs = S.bootstrap_scores(synth_records(rng, m, bands=BANDS[:3], npull=60), B=600, seed=seed)
            w[m] = {k: bs[k]["p84"] - bs[k]["p16"] for k in S._BOOT_STATS}
        for k in S._BOOT_STATS:
            ratios[k].append(w[6][k] / w[24][k])
    for k, r in ratios.items():
        mr = float(np.mean(r))
        print(f"  width(6 exp/band)/width(24 exp/band) {k:11s} mean {mr:.2f} (expected 2.0)")
        assert 1.4 < mr < 2.8, (k, mr)
    print("bootstrap 1/sqrt(n) scaling ok")


if __name__ == "__main__":
    test_parser(); test_selection(); test_bootstrap_point(); test_cluster_wider(); test_bootstrap_scaling()
    print("ALL OK")
