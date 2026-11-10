"""CPU smoke tests of scripts/plan_tiles.py + scripts/tilefit.py + the v40 table transfer, on the SMACS test data (see tests/make_tilefit_testdata.py: 5 consistent
'dithers' per file by cropping; NOT a science test).
  1. plan 3 tiles of 4 arcsec on the generated set;
  2. tilefit: 2 training tiles + 1 held-out tile, --selfcheck, validation every 2 cycles, random order, disk store, --ht-control, --score-coadds, pixel dumps:
     exit 0, finite per-tile and pooled scores (T1, T5, control), instrument_state.pt content, held-out dithers excluded from training, tables updated, kernel dump;
  3. resume: the same command with more cycles continues from ckpt.pt;
  4. v40 --load-tables-from (frozen and fine-tuned) on testdata/nircam (SMACS test crop) with the E2-data flags.
Run: python3 tests/test_tilefit.py [--skip-tilefit]"""
import json, math, os, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
RUNS = os.path.join(ROOT, "runs", "tf_test")
SVO = os.path.join(ROOT, "testdata", "svo_cache")


def sh(cmd, log):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
    open(log, "w").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    print(f"  exit {p.returncode} in {time.time() - t0:.0f} s: {' '.join(cmd[1:4])} ...")
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(1)
    return p.stdout


def fin(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def main(skip_tilefit=False):
    os.makedirs(RUNS, exist_ok=True)
    sys.path.insert(0, HERE)
    import make_tilefit_testdata as M
    data = M.main()
    plan = os.path.join(RUNS, "plan.json")
    sh([PY, os.path.join(ROOT, "scripts", "plan_tiles.py"), "--inputs", data, "--out", plan, "--png", os.path.join(RUNS, "plan.png"), "--n-tiles", "3", "--size-arcsec", "4",
        "--n-min", "4", "--gap-arcsec", "1", "--cells-per-tile", "16", "--cand-stride", "2"], os.path.join(RUNS, "plan.log"))
    out = os.path.join(RUNS, "out2")
    base = [PY, "-u", os.path.join(ROOT, "scripts", "tilefit.py"), "--plan", plan, "--out", out, "--halo-px", "16", "--inner-sky", "1", "--svo-cache", SVO, "--spec-q", "3", "--boot-b", "20",
            "--quick", "--heldout-tiles", "1", "--ht-steps", "5", "--no-fallback", "--val-every-cycles", "2", "--ht-control", "--score-coadds", "--selfcheck", "--tile-order", "random",
            "--tile-store", "disk", "--dump-heldout-pixels", "--seed", "1", "--log-every", "1"]
    if not skip_tilefit:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
        so = sh(base + ["--steps", "4"], os.path.join(RUNS, "tilefit_run1.log"))
        assert "SELFCHECK tile 0" in so and ": ok" in so.split("SELFCHECK")[1].splitlines()[0], "selfcheck"
        tm = json.load(open(os.path.join(out, "tile_metrics.json")))
        for tid, ent in tm["train_tiles"].items():
            T1 = ent["T1"]
            assert fin(T1["chi2_all"], T1["chi2_src"], T1["robust_all"], T1["robust_src"]), (tid, T1)
            assert ent["heldout_dithers_exposures"] == 12, ent
        assert fin(tm["pooled_T1"]["chi2_all"], tm["pooled_T1"]["chi2_src"]) and tm["pooled_T1"]["n_exposures"] == 24
        h = tm["heldout_tiles"]["2"]
        assert fin(h["T5"]["chi2_all"], h["T5"]["chi2_src"], h["T5_control"]["chi2_all"]) and fin(tm["pooled_T5"]["chi2_src"], tm["pooled_T5_control"]["chi2_src"])
        assert "T1_swarp" in tm["train_tiles"]["0"] and "T5_drizzle" in h
        st = torch.load(os.path.join(out, "instrument_state.pt"), weights_only=False)
        assert st["format"] == "tilefit-instrument" and len(st["tables"]) == 6 and set(st["dets"]) == {"NRCB3", "NRCBLONG"}, st["dets"]
        assert all(v["n_updates"] > 0 and torch.isfinite(v["theta"]).all() and abs(float(v["table"].sum()) - 1) < 1e-4 for v in st["tables"].values())
        met = json.load(open(os.path.join(out, "metrics.json")))
        held = set(met["tilefit"]["held_out_dithers"])
        assert held and not (held & set(st["per_exposure"])), "held-out dither used by the instrument"
        assert len(st["per_exposure"]) == 18, len(st["per_exposure"])
        # tables moved from the generic start
        d_ = max(float((v["table"] - v["table"].flip(0).flip(1)).abs().max()) for v in st["tables"].values())
        assert os.path.exists(os.path.join(out, "explicit_lib_3d_psf_groups.npy"))
        g = np.load(os.path.join(out, "explicit_lib_3d_psf_groups.npy"), allow_pickle=True).item()["detector_frame"]
        assert "NRCB3/F090W" in g and "table" in g["NRCB3/F090W"]
        assert os.path.isdir(os.path.join(out, "tile0", "heldout_pixels", "tilefit_lib")) and os.path.isdir(os.path.join(out, "tile2", "heldout_pixels", "tilefit_lib_control"))
        assert os.path.exists(os.path.join(out, "tile_store", "tile0.pt"))
        print("tilefit run 1 ok: pooled T1 S/N>5", round(tm["pooled_T1"]["chi2_src"], 3), "T5", round(tm["pooled_T5"]["chi2_src"], 3), "T5 control", round(tm["pooled_T5_control"]["chi2_src"], 3))
        so2 = sh(base + ["--steps", "6"], os.path.join(RUNS, "tilefit_run2_resume.log"))
        assert "resumed from" in so2 and "cycle 4" in so2.split("resumed from")[1].splitlines()[0], "resume"
        print("resume ok")
    # ---- v40
    st_path = os.path.join(out, "instrument_state.pt")
    NIR = os.path.join(ROOT, "testdata", "nircam")
    e2 = ["--data-dir", NIR, "--size", "128", "--quick", "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "20", "--budget-explicit", "100000", "--stripes", "1", "--eval-every-min", "0",
          "--lam-lo", "0.35", "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3", "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--lib-source", "gauss",
          "--lib-learn-table", "--lib-res", "off", "--psf-window-arcsec", "3.0", "--sample-mode", "bicubic", "--halo-px", "16", "--skip-implicit", "--explicit-configs", "explicit", "--seed", "0"]
    res = {}
    for tag, extra in (("frozen", ["--load-tables-from", st_path, "--freeze-tables"]), ("finetune", ["--load-tables-from", st_path]), ("none", [])):
        o = os.path.join(RUNS, f"v40_{tag}")
        sh([PY, "-u", os.path.join(ROOT, "scripts", "smacs_four_v40.py"), "--out", o] + e2 + extra, os.path.join(RUNS, f"v40_{tag}.log"))
        res[tag] = (o, torch.load(os.path.join(o, "explicit_lib_3d_state.pt"), map_location="cpu"), json.load(open(os.path.join(o, "metrics.json"))))
    lt = res["frozen"][2]["load_tables"]
    print("v40 load_tables:", {k: (len(v) if isinstance(v, list) else v) for k, v in lt.items()})
    assert len(lt["matched"]) == 6 and not lt["absent"]
    # frozen tables equal the loaded ones; fine-tuned ones moved; the default run is at the generic start
    n_frozen_same = n_ft_moved = 0
    for lab, _ in lt["matched"]:
        det, band = lab.split("/")
        dets = res["frozen"][2]["detectors"]; bands = [r["band"] for r in res["frozen"][2]["band_registry"]]
        key = f"psf.tab.{dets.index(det)}_{bands.index(band)}"
        src = torch.load(st_path, weights_only=False)["tables"][lab]
        th_f, th_t, th_0 = res["frozen"][1][key], res["finetune"][1][key], res["none"][1][key]
        if th_f.shape == src["theta"].shape:
            assert torch.equal(th_f, src["theta"]), "frozen table changed"
        n_frozen_same += 1
        n_ft_moved += int(not torch.equal(th_t, th_f))
        assert not torch.equal(th_f, th_0)
    print(f"v40: frozen tables identical to the loaded ones ({n_frozen_same}/6); fine-tuned tables moved ({n_ft_moved}/6); resized: {lt.get('resized', [])[:2]}")
    assert n_ft_moved == 6
    for tag in res:
        c = res[tag][2]["metrics"]["heldout_chi2"]
        print("  v40", tag, "in-sample chi2", {k: round(v, 3) for k, v in c.items() if k.startswith("explicit")})
    print("PASS")




def test_resize():
    """v40 with a source table of a different size N (padded by 2 samples per side): the loaded table must equal the original one (centre crop)."""
    out = os.path.join(RUNS, "out2")
    st = torch.load(os.path.join(out, "instrument_state.pt"), weights_only=False)
    for lab, ent in st["tables"].items():
        T = ent["table"].double()
        Tp = torch.nn.functional.pad(T, (2, 2, 2, 2))
        Tp = Tp / Tp.sum()
        y = Tp.clamp(min=1e-8 * float(Tp.max()))
        ent["theta"] = torch.log(torch.expm1(y)).float(); ent["table"] = Tp.float(); ent["N"] = Tp.shape[-1]
    p = os.path.join(RUNS, "instrument_state_padded.pt")
    torch.save(st, p)
    NIR = os.path.join(ROOT, "testdata", "nircam")
    o = os.path.join(RUNS, "v40_resize")
    e2 = ["--data-dir", NIR, "--size", "128", "--quick", "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "20", "--budget-explicit", "100000", "--stripes", "1", "--eval-every-min", "0",
          "--lam-lo", "0.35", "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3", "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--lib-source", "gauss",
          "--lib-learn-table", "--lib-res", "off", "--psf-window-arcsec", "3.0", "--sample-mode", "bicubic", "--halo-px", "16", "--skip-implicit", "--explicit-configs", "explicit", "--seed", "0"]
    sh([PY, "-u", os.path.join(ROOT, "scripts", "smacs_four_v40.py"), "--out", o] + e2 + ["--load-tables-from", p, "--freeze-tables"], os.path.join(RUNS, "v40_resize.log"))
    met = json.load(open(os.path.join(o, "metrics.json"))); sd = torch.load(os.path.join(o, "explicit_lib_3d_state.pt"), map_location="cpu")
    rs = met["load_tables"]["resized"]
    assert len(rs) == 6 and all(r[1] == r[2] + 4 and r[3] < 1e-12 for r in rs), rs
    orig = torch.load(os.path.join(out, "instrument_state.pt"), weights_only=False)["tables"]
    dets = met["detectors"]; bands = [r["band"] for r in met["band_registry"]]
    worst = 0.0
    for lab, ent in orig.items():
        d, b = lab.split("/")
        th = sd[f"psf.tab.{dets.index(d)}_{bands.index(b)}"]
        T = torch.nn.functional.softplus(th); T = T / T.sum()
        worst = max(worst, float((T - ent["table"]).abs().max() / ent["table"].max()))
    print("v40 resize (source N + 4 -> crop): max |table - original| / peak =", f"{worst:.2e}")
    assert worst < 1e-4, worst


if __name__ == "__main__":
    main("--skip-tilefit" in sys.argv)
    test_resize()
    print("ALL PASS")
