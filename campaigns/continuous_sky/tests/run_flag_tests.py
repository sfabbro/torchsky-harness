"""Quick end-to-end runs of every NEW opfield flag on testdata (CPU, ~3-4 min each): exit code 0, finite held-out scores, parameter counts, the files
that the flag promises. Results go to runs/flag_tests/<name>/ and runs/flag_tests/summary.json.

    venvT/bin/python tests/run_flag_tests.py [--only name1,name2] [--steps 60]

Data: testdata/nircam (6 exposures: NRCB3 F090W, F150W, F200W and NRCBLONG F277W, F356W, F444W) plus a SYMLINK COPY of each under another exposure number
(_00009_): 12 exposures, every band twice (the second image is a duplicate of the first: a smoke test of the plumbing, not of the science; the loader needs
at least one training exposure per band, which rules out holding out a band's only image). --holdout-from holds out the originals of NRCB3 F090W and
NRCBLONG F356W. The aliasing of untrained (detector, band) cells is covered by tests/test_opfield.py::test_untrained_cells_are_aliased. A synthetic E2 table file
(round Gaussian tables with sigma 1.1 / 1.5 detector px, a different centre per cell) is written for --base-init-from.
Runs: default (+ dumps), eval-only from it (scores must agree with the training run), and the flags one at a time:
  base_instrument, base_detector, base_detector_band_l2zero, shape_head_channel, nang72, cartesian, basetable, basetable_init, lambda_jacobian(_laminband),
  psi_log, importance_sampling, two_scale, all_three (8 Nov: --psi-param log, --importance-sampling, --two-scale, and the three together).
Then compare_operators.py (default vs basetable_init dumps) and null_stacks.py (default dumps) and compare_kernels.py (default run vs the synthetic E2 file) are run.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
OUT = os.path.join(ROOT, "runs", "flag_tests")


def common(data_dir, steps):
    return ["--data-dir", data_dir, "--size", "128", "--halo-px", "16", "--quick", "--no-fallback", "--holdout-from", os.path.join(OUT, "holdout.txt"), "--boot-b", "50", "--stripes", "1",
            "--svo-cache", os.path.join(ROOT, "testdata", "svo_cache"), "--budget-min", "20", "--steps", str(steps), "--nuis-steps", "10", "--batch", "512",
            "--log-every", "30", "--loss", "huber", "--noise-start", "2.0", "--core-radius", "10", "--core-pitch", "0.5", "--dump-heldout-pixels"]


def make_data_dir():
    open(os.path.join(OUT, "holdout.txt"), "w").write("jw02736001001_02101_00002_nrcb3_cal.fits\njw02736001001_02103_00004_nrcblong_cal.fits\n")
    d = os.path.join(OUT, "data")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(ROOT, "testdata", "nircam")
    for f in sorted(os.listdir(src)):
        if f.endswith("_cal.fits"):
            for name in (f, "_".join(f.split("_")[:2] + ["00009"] + f.split("_")[3:])):
                t = os.path.join(d, name)
                if not os.path.exists(t):
                    os.symlink(os.path.join(src, f), t)
    return d


def make_e2(dets_bands, path):
    os.makedirs(path, exist_ok=True)
    os_, N = 4, 2 * 40 * 4 + 1
    ax = (np.arange(N) - N // 2) / os_
    xx, yy = np.meshgrid(ax, ax)
    g = {}
    for k, (d, b) in enumerate(dets_bands):
        sig = 1.1 if "LONG" not in d else 1.5
        K = np.exp(-0.5 * ((xx - 0.05 * k) ** 2 + (yy + 0.03 * k) ** 2) / sig ** 2)
        K = K / K.sum()
        g[f"{d}/{b}"] = {"det": d, "band": b, "node_index": 1, "lam_um": 1.0, "oversample": 4, "half_window_det_px": 40.0, "library": K, "learned": K, "table": K}
    np.save(os.path.join(path, "explicit_lib_3d_psf_groups.npy"), {"detector_frame": g}, allow_pickle=True)


RUNS = {
    "default": [],
    "base_instrument": ["--base-per", "instrument"],
    "base_detector": ["--base-per", "detector"],
    "base_detector_band_l2zero": ["--base-per", "detector_band", "--base-l2", "0"],
    "shape_head_channel": ["--shape-head-per", "channel"],
    "nang72": ["--wing-nang", "72"],
    "cartesian": ["--wings", "cartesian"],
    "basetable": ["--base-table", "detector_band"],
    "basetable_init": ["--base-table", "detector_band", "--base-init-from", "E2DIR"],
    "lambda_jacobian": ["--lambda-gauge", "jacobian"],
    "lambda_jacobian_laminband": ["--lambda-gauge", "jacobian", "--lam-in-band"],
    # 8 Nov 2026: wings / shoulders (tests/test_opfield_wings.py has the unit tests and tests/synth_wings.py the synthetic check)
    "psi_log": ["--psi-param", "log"],
    "importance_sampling": ["--importance-sampling", "1,3,8,15,30", "--is-refresh", "20", "--is-snr", "10"],
    "two_scale": ["--two-scale", "0.3", "--wing-mod", "1"],
    "all_three": ["--psi-param", "log", "--importance-sampling", "1,3,8,15,30", "--is-refresh", "20", "--is-snr", "10", "--two-scale", "0.3", "--wing-mod", "1",
                  "--lambda-gauge", "jacobian", "--lam-in-band"],
}


def run(name, args, env_note=""):
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    t = time.time()
    p = subprocess.run([PY, "-u", os.path.join(ROOT, "scripts", "opfield.py"), "--out", out] + args, capture_output=True, text=True, cwd=ROOT)
    open(os.path.join(OUT, name + ".log"), "w").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    rec = {"name": name, "exit": p.returncode, "seconds": round(time.time() - t, 1), "args": args}
    if p.returncode == 0 and os.path.exists(os.path.join(out, "metrics.json")):
        M = json.load(open(os.path.join(out, "metrics.json")))
        m = M["metrics"]
        rec.update(chi2=m["heldout_chi2"]["opfield"], chi2_src=m["heldout_chi2_sources"]["opfield"], robust=m["heldout_robust"]["opfield"], robust_src=m["heldout_robust_sources"]["opfield"],
                   per_band={b: v["chi2_src"] for b, v in m["heldout_per_band"]["opfield"].items()}, param_counts=M.get("param_counts"), base_rows=M.get("base_rows", {}).get("n_rows"),
                   aliased=M.get("base_rows", {}).get("aliased_untrained"), stencil_points=M["stencil"]["points_total"], nk=[v["nk"] for v in M["stencil"]["per_channel"].values()],
                   n_dump=len([f for f in os.listdir(os.path.join(out, "heldout_pixels", "opfield"))]) if os.path.isdir(os.path.join(out, "heldout_pixels", "opfield")) else 0,
                   base_init=M.get("base_init_from", {}).get("matched"), eval_only=M.get("eval_only"),
                   kdiag=M.get("kernel_diagnostics", {}).get("cells"))
        rec["finite"] = bool(all(np.isfinite(rec[k]) for k in ("chi2", "chi2_src", "robust", "robust_src")))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = make_data_dir()
    # det / band pairs of the data
    pairs = []
    for f in sorted(os.listdir(data)):
        if not f.endswith("_cal.fits"):
            continue
        det = f.split("_")[3].upper()
        band = {"02101": ("F090W", "F277W"), "02103": ("F150W", "F356W"), "02105": ("F200W", "F444W")}[f.split("_")[1]][1 if "LONG" in det else 0]
        pairs.append((det, band))
    e2 = os.path.join(OUT, "e2_synth")
    make_e2(pairs, e2)
    sel = a.only.split(",") if a.only else list(RUNS) + ["eval_only"]
    summary = {}
    sj = os.path.join(OUT, "summary.json")
    if os.path.exists(sj):
        summary = json.load(open(sj))
    for name in sel:
        if name == "eval_only":
            args = ["--args-from", os.path.join(OUT, "default", "metrics.json"), "--eval-only", "--init-state", os.path.join(OUT, "default", "opfield_state.pt"), "--dump-heldout-pixels"]
        else:
            args = common(data, a.steps) + [x if x != "E2DIR" else e2 for x in RUNS[name]]
        rec = run(name, args)
        summary[name] = rec
        json.dump(summary, open(sj, "w"), indent=1, default=float)
        print(f"{name}: exit {rec['exit']} {rec['seconds']} s  chi2 {rec.get('chi2')} src {rec.get('chi2_src')} rows {rec.get('base_rows')} nk {rec.get('nk')} params {None if not rec.get('param_counts') else rec['param_counts']['total']}", flush=True)
        assert rec["exit"] == 0, open(os.path.join(OUT, name + ".log")).read()[-3000:]
        assert rec["finite"] and rec["param_counts"] and rec["n_dump"] > 0
        kd = rec["kdiag"]
        assert kd and all(np.isfinite(v["exp_a_mean"]) and np.isfinite(v["exp_a_std"]) and set(v["ee_core_arcsec"]) == {"0.1", "0.2", "0.5"} for v in kd.values()), "kernel_diagnostics"
        if name.startswith("lambda_jacobian"):
            z = np.load(os.path.join(OUT, name, "kernels", "effective_kernels.npz"))
            for k in z.files:
                if k.endswith("|Lc"):
                    assert np.array_equal(z[k], np.eye(2, dtype=np.float32)), (k, z[k])
                if k.endswith("|M"):
                    assert np.array_equal(z[k], z[k[:-2] + "|J"]), k          # M == J exactly
            print(f"   {name}: Lambda == I and M == J in all {sum(k.endswith('|Lc') for k in z.files)} kernel groups; kernel_diagnostics cells: {len(kd)}")
    if "default" in summary and "eval_only" in summary:
        d, e = summary["default"], summary["eval_only"]
        rel = abs(d["chi2"] - e["chi2"]) / d["chi2"]
        rels = abs(d["chi2_src"] - e["chi2_src"]) / d["chi2_src"]
        print(f"eval-only vs training run: chi2 {d['chi2']:.5f} vs {e['chi2']:.5f} (rel {rel:.2e}); src {d['chi2_src']:.5f} vs {e['chi2_src']:.5f} (rel {rels:.2e})")
        assert rel < 2e-3 and rels < 2e-3
    d0 = os.path.join(OUT, "default")
    if os.path.isdir(os.path.join(d0, "heldout_pixels")):
        subprocess.run([PY, os.path.join(ROOT, "scripts", "null_stacks.py"), d0, "--out", os.path.join(OUT, "null_stacks_default")], check=True, cwd=ROOT, capture_output=True)
        print("null_stacks.py on the opfield dump: ok")
        if os.path.isdir(os.path.join(OUT, "eval_only", "heldout_pixels")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "compare_operators.py"), d0, os.path.join(OUT, "eval_only"), "--out", os.path.join(OUT, "cmp_ops_train_vs_evalonly"),
                            "--splits", "test"], check=True, cwd=ROOT)
        if os.path.isdir(os.path.join(OUT, "basetable_init", "heldout_pixels")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "compare_operators.py"), d0, os.path.join(OUT, "basetable_init"), "--out", os.path.join(OUT, "cmp_ops_default_vs_basetable_init"),
                            "--splits", "test"], check=True, cwd=ROOT)
        subprocess.run([PY, os.path.join(ROOT, "scripts", "compare_kernels.py"), d0, e2, "--out", os.path.join(OUT, "cmp_kernels_default_vs_synthE2")], check=True, cwd=ROOT)
    print("ALL OK")


if __name__ == "__main__":
    main()
