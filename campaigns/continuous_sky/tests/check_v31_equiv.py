"""v32 without --sky-coeffs splat2 must behave as v31. Runs smacs_four_v31.py and smacs_four_v32.py on the same NIRCam cal files with the same
arguments and the same torch seed (1 CPU thread), then compares: stdout (timestamps stripped), every cube FITS (bitwise), every state
dict tensor (bitwise), the saved .npy products, and metrics.json (all v31 keys, exact; timings and args excluded).
Run: python3 tests/check_v31_equiv.py  (needs torchsky, i.e. the venv with the built wheel)."""
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NIR = os.path.join(ROOT, "testdata", "nircam")
SVO = os.path.join(ROOT, "testdata", "svo_cache")
CODE = ("import sys, runpy, torch; torch.manual_seed(0); torch.set_num_threads(1); "
        "sys.argv = sys.argv[1:]; sys.path.insert(0, __import__('os').path.dirname(sys.argv[0])); runpy.run_path(sys.argv[0], run_name='__main__')")
BASE = ["--data-dir", NIR, "--size", "128", "--quick", "--no-fallback", "--no-holdout", "--score-all", "--boot-b", "50",
        "--budget-explicit", "100000", "--budget-implicit", "100000", "--stripes", "1", "--eval-every-min", "0"]
VARIANTS = {
    "default": [],
    "hybrid_spectral": ["--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3", "--sky-coeffs", "hybrid", "--skip-implicit", "--explicit-configs", "fixed"],
    "chrom_spectral": ["--explicit-psf", "chrom", "--psf-model", "chrom", "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3"],
}


def run(ver, tag, extra):
    out = os.path.join(ROOT, "runs", f"equiv_{tag}_v{ver}")
    os.makedirs(out, exist_ok=True)
    script = os.path.join(ROOT, "scripts", f"smacs_four_v{ver}.py")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-u", "-c", CODE, script, "--out", out] + BASE + extra, capture_output=True, text=True)
    print(f"  v{ver} {tag}: exit {p.returncode} in {time.time() - t0:.0f} s")
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
        raise SystemExit(1)
    return out, p.stdout


def strip(stdout):
    keep = []
    for ln in stdout.splitlines():
        ln = re.sub(r"^\[\s*[\d.]+s\] ", "", ln)
        ln = re.sub(r"\d+\.\d+ s\b", "<t> s", ln) if "timing" in ln.lower() else ln
        keep.append(ln)
    return keep


def same(a, b, path=""):
    """exact comparison, nan == nan; returns a list of differing paths"""
    if isinstance(a, dict) and isinstance(b, dict):
        bad = []
        for k in a:
            if k not in b:
                bad.append(f"{path}/{k}: missing in v32")
            else:
                bad += same(a[k], b[k], f"{path}/{k}")
        return bad
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} vs {len(b)}"]
        return [x for i, (p, q) in enumerate(zip(a, b)) for x in same(p, q, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return [] if np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) else [f"{path}: arrays differ"]
    return [] if a == b else [f"{path}: {str(a)[:60]} vs {str(b)[:60]}"]


def compare(tag, o31, o32, so31, so32):
    n_bad = 0
    l31, l32 = strip(so31), strip(so32)
    # every v31 line must appear, in order, unchanged
    extra = [ln for ln in l32 if ln not in set(l31)]
    miss = [ln for ln in l31 if ln not in set(l32)]
    order_ok = [ln for ln in l32 if ln in set(l31)] == [ln for ln in l31 if ln in set(l32)]
    print(f"  stdout: {len(l31)} v31 lines, {len(l32)} v32 lines; v31 lines missing in v32: {len(miss)}; extra v32 lines: {len(extra)}; same order: {order_ok}")
    for ln in miss[:10]:
        print("    MISSING:", ln[:200])
    for ln in extra[:10]:
        print("    EXTRA  :", ln[:200])
    n_bad += len(miss) + (not order_ok)
    nfile = 0
    for root, _, files in os.walk(o31):
        for f in sorted(files):
            p31 = os.path.join(root, f)
            p32 = os.path.join(o32, os.path.relpath(p31, o31))
            if f in ("metrics.json", "metrics_partial.json", "stdout.log"):
                continue
            nfile += 1
            if not os.path.exists(p32):
                print("    MISSING FILE", p32); n_bad += 1; continue
            if f.endswith(".fits"):
                from astropy.io import fits
                a, b = fits.getdata(p31), fits.getdata(p32)
                ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
            elif f.endswith(".pt"):
                a, b = torch.load(p31, weights_only=False), torch.load(p32, weights_only=False)
                ok = list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
            elif f.endswith(".npy"):
                a, b = np.load(p31, allow_pickle=True), np.load(p32, allow_pickle=True)
                ok = (not same(a.item(), b.item())) if a.dtype == object else np.array_equal(a, b, equal_nan=True)
            else:
                ok = open(p31, "rb").read() == open(p32, "rb").read()
            if not ok:
                print("    DIFFERENT:", os.path.relpath(p31, o31)); n_bad += 1
    print(f"  {nfile} output files compared bitwise (cubes, states, npy), differences: {n_bad - len(miss) - (not order_ok)}")
    m31, m32 = json.load(open(os.path.join(o31, "metrics.json"))), json.load(open(os.path.join(o32, "metrics.json")))
    skip = ("timings", "args")
    d = same({k: v for k, v in m31.items() if k not in skip}, {k: v for k, v in m32.items() if k not in skip})
    newkeys = sorted(set(m32) - set(m31))
    print(f"  metrics.json: {len(m31)} top-level keys in v31; {len(d)} differences in them (timings/args excluded); keys only in v32: {newkeys}")
    for x in d[:10]:
        print("    ", x)
    n_bad += len(d)
    print(f"  args only in v32: {sorted(set(m32['args']) - set(m31['args']))}; changed defaults: "
          f"{ {k: (m31['args'][k], m32['args'][k]) for k in m31['args'] if m31['args'][k] != m32['args'].get(k)} }")
    return n_bad


if __name__ == "__main__":
    total = 0
    only = sys.argv[1:]
    for tag, extra in VARIANTS.items():
        if only and tag not in only:
            continue
        print(f"== variant {tag}: {' '.join(extra) or '(defaults)'}")
        o31, s31 = run(31, tag, extra)
        o32, s32 = run(32, tag, extra)
        total += compare(tag, o31, o32, s31, s32)
    print("\nEQUIVALENT" if total == 0 else f"\nNOT EQUIVALENT ({total} differences)")
    sys.exit(1 if total else 0)
