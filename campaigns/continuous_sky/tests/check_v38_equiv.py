"""v38 without the new flags must behave as v37. Runs smacs_four_v37.py and smacs_four_v38.py on the same NIRCam cal files with the same
arguments and the same torch seed (1 CPU thread), then compares: stdout (timestamps stripped), every cube FITS (bitwise), every state
dict tensor (bitwise), the saved .npy products, and metrics.json (all v37 keys, exact; timings and args excluded).
Run: python3 tests/check_v38_equiv.py  (needs torchsky, i.e. the venv with the built wheel)."""
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
    "chrom_spectral": ["--explicit-psf", "chrom", "--psf-model", "chrom", "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3"],
    # the E2-data configuration (NIRCam + HST, library PSF with learnable Gaussian tables); run explicitly: python3 tests/check_v38_equiv.py e2data
    "e2data": ["--hst-dir", os.path.join(ROOT, "testdata", "hst_all"), "--lam-lo", "0.35", "--spectral", "rbf", "--svo-cache", SVO, "--spec-q", "3",
               "--psf-frame", "detector", "--psf-model", "lib", "--explicit-psf", "lib", "--lib-source", "gauss", "--lib-learn-table", "--lib-res", "off",
               "--psf-window-arcsec", "3.0", "--sample-mode", "bicubic", "--halo-px", "16", "--skip-implicit", "--explicit-configs", "explicit"],
}
DEFAULT_VARIANTS = ("default", "chrom_spectral")


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
                bad.append(f"{path}/{k}: missing in v38")
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


def compare(tag, o37, o38, so37, so38):
    n_bad = 0
    l37, l38 = strip(so37), strip(so38)
    # every v37 line must appear, in order, unchanged
    extra = [ln for ln in l38 if ln not in set(l37)]
    miss = [ln for ln in l37 if ln not in set(l38)]
    order_ok = [ln for ln in l38 if ln in set(l37)] == [ln for ln in l37 if ln in set(l38)]
    print(f"  stdout: {len(l37)} v37 lines, {len(l38)} v38 lines; v37 lines missing in v38: {len(miss)}; extra v38 lines: {len(extra)}; same order: {order_ok}")
    for ln in miss[:10]:
        print("    MISSING:", ln[:200])
    for ln in extra[:10]:
        print("    EXTRA  :", ln[:200])
    n_bad += len(miss) + (not order_ok)
    nfile = 0
    for root, _, files in os.walk(o37):
        for f in sorted(files):
            p37 = os.path.join(root, f)
            p38 = os.path.join(o38, os.path.relpath(p37, o37))
            if f in ("metrics.json", "metrics_partial.json", "stdout.log"):
                continue
            nfile += 1
            if not os.path.exists(p38):
                print("    MISSING FILE", p38); n_bad += 1; continue
            if f.endswith(".fits"):
                from astropy.io import fits
                a, b = fits.getdata(p37), fits.getdata(p38)
                ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
            elif f.endswith(".pt"):
                a, b = torch.load(p37, weights_only=False), torch.load(p38, weights_only=False)
                ok = list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
            elif f.endswith(".npy"):
                a, b = np.load(p37, allow_pickle=True), np.load(p38, allow_pickle=True)
                ok = (not same(a.item(), b.item())) if a.dtype == object else np.array_equal(a, b, equal_nan=True)
            else:
                ok = open(p37, "rb").read() == open(p38, "rb").read()
            if not ok:
                print("    DIFFERENT:", os.path.relpath(p37, o37)); n_bad += 1
    print(f"  {nfile} output files compared bitwise (cubes, states, npy), differences: {n_bad - len(miss) - (not order_ok)}")
    m37, m38 = json.load(open(os.path.join(o37, "metrics.json"))), json.load(open(os.path.join(o38, "metrics.json")))
    skip = ("timings", "args")
    d = same({k: v for k, v in m37.items() if k not in skip}, {k: v for k, v in m38.items() if k not in skip})
    newkeys = sorted(set(m38) - set(m37))
    print(f"  metrics.json: {len(m37)} top-level keys in v37; {len(d)} differences in them (timings/args excluded); keys only in v38: {newkeys}")
    for x in d[:10]:
        print("    ", x)
    n_bad += len(d)
    print(f"  args only in v38: {sorted(set(m38['args']) - set(m37['args']))}; changed defaults: "
          f"{ {k: (m37['args'][k], m38['args'][k]) for k in m37['args'] if m37['args'][k] != m38['args'].get(k)} }")
    return n_bad


if __name__ == "__main__":
    total = 0
    only = sys.argv[1:]
    for tag, extra in VARIANTS.items():
        if (only and tag not in only) or (not only and tag not in DEFAULT_VARIANTS):
            continue
        print(f"== variant {tag}: {' '.join(extra) or '(defaults)'}")
        o37, s37 = run(37, tag, extra)
        o38, s38 = run(38, tag, extra)
        total += compare(tag, o37, o38, s37, s38)
    print("\nEQUIVALENT" if total == 0 else f"\nNOT EQUIVALENT ({total} differences)")
    sys.exit(1 if total else 0)
