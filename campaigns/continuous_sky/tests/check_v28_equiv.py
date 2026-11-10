"""v29 without any --hst-* flag must behave as v28. Runs smacs_four_v28.py and smacs_four_v29.py on the same NIRCam cal files with the same
arguments and the same torch seed (1 CPU thread), then compares: stdout (timestamps stripped), every cube FITS (bitwise), every state
dict tensor (bitwise), the saved .npy products, and metrics.json (all v28 keys, exact; timings and args excluded).
Run: python3 tests/check_v28_equiv.py  (needs torchsky, i.e. the venv with the built wheel)."""
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
                bad.append(f"{path}/{k}: missing in v29")
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


def compare(tag, o28, o29, so28, so29):
    n_bad = 0
    l28, l29 = strip(so28), strip(so29)
    # v29 prints one extra line (the band registry); every v28 line must appear, in order, unchanged
    extra = [ln for ln in l29 if ln not in set(l28)]
    miss = [ln for ln in l28 if ln not in set(l29)]
    order_ok = [ln for ln in l29 if ln in set(l28)] == [ln for ln in l28 if ln in set(l29)]
    print(f"  stdout: {len(l28)} v28 lines, {len(l29)} v29 lines; v28 lines missing in v29: {len(miss)}; extra v29 lines: {len(extra)}; same order: {order_ok}")
    for ln in miss[:10]:
        print("    MISSING:", ln[:200])
    for ln in extra[:10]:
        print("    EXTRA  :", ln[:200])
    n_bad += len(miss) + (not order_ok)
    nfile = 0
    for root, _, files in os.walk(o28):
        for f in sorted(files):
            p28 = os.path.join(root, f)
            p29 = os.path.join(o29, os.path.relpath(p28, o28))
            if f in ("metrics.json", "metrics_partial.json", "stdout.log"):
                continue
            nfile += 1
            if not os.path.exists(p29):
                print("    MISSING FILE", p29); n_bad += 1; continue
            if f.endswith(".fits"):
                from astropy.io import fits
                a, b = fits.getdata(p28), fits.getdata(p29)
                ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
            elif f.endswith(".pt"):
                a, b = torch.load(p28, weights_only=False), torch.load(p29, weights_only=False)
                ok = list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
            elif f.endswith(".npy"):
                a, b = np.load(p28, allow_pickle=True), np.load(p29, allow_pickle=True)
                ok = (not same(a.item(), b.item())) if a.dtype == object else np.array_equal(a, b, equal_nan=True)
            else:
                ok = open(p28, "rb").read() == open(p29, "rb").read()
            if not ok:
                print("    DIFFERENT:", os.path.relpath(p28, o28)); n_bad += 1
    print(f"  {nfile} output files compared bitwise (cubes, states, npy), differences: {n_bad - len(miss) - (not order_ok)}")
    m28, m29 = json.load(open(os.path.join(o28, "metrics.json"))), json.load(open(os.path.join(o29, "metrics.json")))
    skip = ("timings", "args")
    d = same({k: v for k, v in m28.items() if k not in skip}, {k: v for k, v in m29.items() if k not in skip})
    newkeys = sorted(set(m29) - set(m28))
    print(f"  metrics.json: {len(m28)} top-level keys in v28; {len(d)} differences in them (timings/args excluded); keys only in v29: {newkeys}")
    for x in d[:10]:
        print("    ", x)
    n_bad += len(d)
    print(f"  args only in v29: {sorted(set(m29['args']) - set(m28['args']))}; changed defaults: "
          f"{ {k: (m28['args'][k], m29['args'][k]) for k in m28['args'] if m28['args'][k] != m29['args'].get(k)} }")
    return n_bad


if __name__ == "__main__":
    total = 0
    only = sys.argv[1:]
    for tag, extra in VARIANTS.items():
        if only and tag not in only:
            continue
        print(f"== variant {tag}: {' '.join(extra) or '(defaults)'}")
        o28, s28 = run(28, tag, extra)
        o29, s29 = run(29, tag, extra)
        total += compare(tag, o28, o29, s28, s29)
    print("\nEQUIVALENT" if total == 0 else f"\nNOT EQUIVALENT ({total} differences)")
    sys.exit(1 if total else 0)
