#!/usr/bin/env python3
"""Fair crossmatch performance: torchsky vs scipy cKDTree (and astropy at small N).

Each (engine, workload) runs in its own subprocess, so peak RSS belongs to that
engine alone. Every engine gets the same thread count: ``torch.set_num_threads``
for torchsky and ``workers=`` for cKDTree. cKDTree pairs are flattened with
NumPy rather than a Python loop. Build and query are timed separately, and their
sum is the end-to-end time.

Pair-ID parity is checked before any timing is reported. Pairs on which the
engines disagree are recomputed in float64. Disagreements outside the 1e-8 arcsec
tie band around the radius mark the row ``FAIL``.

Layouts:

- ``sparse``: uniform all-sky, right = left jittered by 0.3 arcsec.
- ``dense``: a 100 deg^2 field, same jitter (the xmatcher audit setup).
- ``crowded``: Plummer clusters; right drawn independently.

Modes: ``best`` (nearest within radius), ``all`` (every pair) and ``knn2``
(two nearest, unbounded).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR

TIE_BAND_ARCSEC = 1e-8


def _layout(name: str, n: int, seed: int) -> tuple[np.ndarray, ...]:
    g = np.random.default_rng(seed)
    if name == "sparse":
        ra = g.uniform(0, 360, n)
        dec = np.degrees(np.arcsin(g.uniform(-1, 1, n)))
    elif name == "dense":
        ra = 150.0 + g.uniform(0, 10, n)
        dec = 2.0 + g.uniform(0, 10, n)
    elif name == "crowded":
        centers = g.uniform([0, -60], [360, 60], size=(max(1, n // 20000), 2))
        k = g.integers(0, len(centers), n)
        u = g.uniform(1e-6, 1, n)
        r = np.minimum(30.0 / np.sqrt(u ** (-2 / 3) - 1), 1800.0) / 3600.0
        t = g.uniform(0, 2 * math.pi, n)
        dec = np.clip(centers[k, 1] + r * np.sin(t), -90, 90)
        ra = (centers[k, 0] + r * np.cos(t) / np.cos(np.radians(dec))) % 360
        g2 = np.random.default_rng(seed + 1)
        k2 = g2.integers(0, len(centers), n)
        u2 = g2.uniform(1e-6, 1, n)
        r2 = np.minimum(30.0 / np.sqrt(u2 ** (-2 / 3) - 1), 1800.0) / 3600.0
        t2 = g2.uniform(0, 2 * math.pi, n)
        rdec = np.clip(centers[k2, 1] + r2 * np.sin(t2), -90, 90)
        rra = (centers[k2, 0] + r2 * np.cos(t2) / np.cos(np.radians(rdec))) % 360
        return ra, dec, rra, rdec
    else:
        raise ValueError(name)
    j = np.random.default_rng(seed + 1)
    s = 0.3 / 3600.0
    rdec = np.clip(dec + j.normal(0, s, n), -90, 90)
    rra = (ra + j.normal(0, s, n) / np.maximum(np.cos(np.radians(dec)), 1e-9)) % 360
    return ra, dec, rra, rdec


def _unit(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], 1)


def _peak_rss_bytes() -> int:
    """This process's peak RSS.

    On Linux ``ru_maxrss`` survives fork + exec, so a worker would report its
    parent's peak; ``VmHWM`` belongs to the current address space and resets on exec.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


# ------------------------------------------------------------------ workers


def _worker(args: argparse.Namespace) -> None:
    la, lb, ra, rb = _layout(args.layout, args.n, args.seed)
    out = Path(args.out)
    t_build = 0.0
    if args.engine == "torchsky":
        import torch

        import torchsky
        from torchsky.catalogs import SkyIndex, crossmatch_sky

        torch.set_num_threads(args.threads)
        dev = torch.device(args.device)
        L = (torch.from_numpy(la).to(dev), torch.from_numpy(lb).to(dev))
        R = (torch.from_numpy(ra).to(dev), torch.from_numpy(rb).to(dev))

        def sync() -> None:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        def build():
            index = SkyIndex.build(*R) if args.mode == "knn2" else None
            sync()
            return index

        def query(index):
            if args.mode == "knn2":
                res = index.query_nearest(*L, 2)
                sync()
                return res.index.cpu().numpy(), None
            res = crossmatch_sky(*L, *R, radius_arcsec=args.radius, find=args.mode)
            sync()
            return res.left_index.cpu().numpy(), res.right_index.cpu().numpy()

        version = torchsky.__version__
    elif args.engine == "ckdtree":
        import scipy
        from scipy.spatial import cKDTree

        A, B = _unit(la, lb), _unit(ra, rb)
        chord = 2 * math.sin(math.radians(args.radius / 3600) / 2)

        def build():
            return cKDTree(B)

        def query(tree):
            if args.mode == "knn2":
                _, j = tree.query(A, k=2, workers=args.threads)
                return j, None
            if args.mode == "best":
                d, j = tree.query(A, k=1, distance_upper_bound=chord, workers=args.threads)
                ok = np.isfinite(d)
                return np.nonzero(ok)[0], j[ok]
            lists = tree.query_ball_point(A, chord, workers=args.threads, return_sorted=False)
            counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if counts.sum() else np.empty(0, np.int64)
            return np.repeat(np.arange(len(lists)), counts), flat

        version = scipy.__version__
    elif args.engine == "astropy":
        import astropy
        import astropy.units as u
        from astropy.coordinates import SkyCoord, search_around_sky

        def build():
            return SkyCoord(ra * u.deg, rb * u.deg)

        left = SkyCoord(la * u.deg, lb * u.deg)

        def query(right):
            if args.mode == "knn2":
                j2 = left.match_to_catalog_sky(right, nthneighbor=2)[0]
                j1 = left.match_to_catalog_sky(right, nthneighbor=1)[0]
                return np.stack([j1, j2], 1), None
            if args.mode == "best":
                j, d, _ = left.match_to_catalog_sky(right)
                ok = d.arcsec <= args.radius
                return np.nonzero(ok)[0], j[ok]
            il, ir, _, _ = search_around_sky(left, right, args.radius * u.arcsec)
            return il, ir

        version = astropy.__version__
    else:
        raise ValueError(args.engine)

    samples = []
    for rep in range(args.runs + 1):  # first call warms up
        t0 = time.perf_counter()
        index = build()
        t1 = time.perf_counter()
        li, rj = query(index)
        t2 = time.perf_counter()
        if rep:
            samples.append((t1 - t0, t2 - t1))
    t_build = float(np.median([s[0] for s in samples]))
    t_query = float(np.median([s[1] for s in samples]))
    np.save(out / f"{args.engine}_left.npy", np.asarray(li))
    if rj is not None:
        np.save(out / f"{args.engine}_right.npy", np.asarray(rj))
    (out / f"{args.engine}.json").write_text(json.dumps({
        "engine": args.engine, "engine_version": version, "build_s": t_build,
        "query_s": t_query, "time_s": t_build + t_query, "peak_rss_bytes": _peak_rss_bytes(),
        "pairs": int(np.asarray(li).shape[0]) if rj is not None else int(np.asarray(li).shape[0]),
    }))


# ------------------------------------------------------------------ parent


def _sep_arcsec(la, lb, ra, rb, li, rj) -> np.ndarray:
    a, b = _unit(la[li], lb[li]), _unit(ra[rj], rb[rj])
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))) * 3600


def _parity(args, workload_dir: Path, engines: list[str]) -> dict[str, Any]:
    la, lb, ra, rb = _layout(args.layout, args.n, args.seed)
    ref = engines[0]
    out: dict[str, Any] = {}
    if args.mode == "knn2":
        a = np.load(workload_dir / f"{ref}_left.npy")
        for other in engines[1:]:
            b = np.load(workload_dir / f"{other}_left.npy")
            rows = np.nonzero((a != b).any(axis=1))[0]
            bad = 0
            for i in rows:  # float64 near-ties between neighbours are ambiguous
                sa = _sep_arcsec(la, lb, ra, rb, np.full(2, i), a[i])
                sb = _sep_arcsec(la, lb, ra, rb, np.full(2, i), b[i])
                bad += int(np.max(np.abs(np.sort(sa) - np.sort(sb))) > TIE_BAND_ARCSEC)
            out[other] = {"differing_rows": int(rows.size), "mismatches": bad}
        return out
    def key(engine: str) -> np.ndarray:
        left = np.load(workload_dir / f"{engine}_left.npy").astype(np.int64)
        return left * args.n + np.load(workload_dir / f"{engine}_right.npy")

    base = key(ref)
    for other in engines[1:]:
        k = key(other)
        diff = np.setxor1d(base, k)
        li, rj = diff // args.n, diff % args.n
        sep = _sep_arcsec(la, lb, ra, rb, li, rj)
        if args.mode == "best":
            # A different nearest partner is fine only if equidistant within the band.
            bad_rows = np.unique(li)
            bad = 0
            for i in bad_rows:
                s = sep[li == i]
                bad += int(s.size != 2 or abs(s[0] - s[1]) > TIE_BAND_ARCSEC)
        else:
            bad = int(np.count_nonzero(np.abs(sep - args.radius) > TIE_BAND_ARCSEC))
        out[other] = {"differing_pairs": int(diff.size), "mismatches": bad}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=lambda s: [int(float(x)) for x in s.split(",")], default=[10_000, 100_000, 1_000_000])
    p.add_argument("--layouts", type=lambda s: s.split(","), default=["sparse", "dense", "crowded"])
    p.add_argument("--modes", type=lambda s: s.split(","), default=["best", "all", "knn2"])
    p.add_argument("--threads", type=lambda s: [int(x) for x in s.split(",")], default=[os.cpu_count() or 1])
    p.add_argument("--radius-arcsec", type=float, default=2.0)
    p.add_argument("--device", default="cpu", help="torchsky device (cKDTree and astropy run on CPU)")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--astropy-max-n", type=int, default=100_000)
    p.add_argument("--min-ratio-vs-ckdtree", type=float, default=None,
                   help="fail rows where cKDTree time / torchsky time is below this")
    p.add_argument("--min-ratio-size", type=int, default=1_000_000,
                   help="apply --min-ratio-vs-ckdtree only from this size up")
    p.add_argument("--max-rss-ratio-vs-ckdtree", type=float, default=None)
    p.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "catalog_match_perf.json")
    # internal worker arguments
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--engine", help=argparse.SUPPRESS)
    p.add_argument("--layout", help=argparse.SUPPRESS)
    p.add_argument("--mode", help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, help=argparse.SUPPRESS)
    p.add_argument("--radius", type=float, help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        args.threads = args.threads[0]
        _worker(args)
        return 0

    rows: list[dict[str, Any]] = []
    for threads in args.threads:
        for layout in args.layouts:
            for n in args.sizes:
                for mode in args.modes:
                    engines = ["torchsky", "ckdtree"] + (["astropy"] if n <= args.astropy_max_n else [])
                    with tempfile.TemporaryDirectory(prefix="xm_perf_") as tmp:
                        results = {}
                        for engine in engines:
                            cmd = [sys.executable, "-m", "torchsky_harness.benchmarks.bench_catalog_match_perf",
                                   "--worker", "--engine", engine, "--layout", layout, "--mode", mode,
                                   "--n", str(n), "--radius", str(args.radius_arcsec), "--seed", str(args.seed),
                                   "--runs", str(args.runs), "--threads", str(threads), "--out", tmp,
                                   "--device", args.device]
                            env = dict(os.environ, OMP_NUM_THREADS=str(threads))
                            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                            if proc.returncode != 0:
                                results[engine] = {"error": proc.stderr.strip().splitlines()[-1:] or ["failed"]}
                            else:
                                results[engine] = json.loads((Path(tmp) / f"{engine}.json").read_text())
                        ok_engines = [e for e in engines if "error" not in results[e]]
                        parity = _parity(argparse.Namespace(layout=layout, n=n, seed=args.seed, mode=mode,
                                                            radius=args.radius_arcsec), Path(tmp), ok_engines) \
                            if "torchsky" in ok_engines and len(ok_engines) > 1 else {}
                    ts = results.get("torchsky", {})
                    for engine in engines:
                        r = results[engine]
                        row: dict[str, Any] = {"case": f"{layout}_{mode}_n{n}_t{threads}", "layout": layout,
                                               "mode": mode, "n_left": n, "n_right": n, "threads": threads,
                                               "radius_arcsec": args.radius_arcsec, "library": engine,
                                               "device": args.device if engine == "torchsky" else "cpu"}
                        if "error" in r:
                            row.update(status="FAIL" if engine == "torchsky" else "UNAVAILABLE", reason=r["error"])
                            rows.append(row)
                            continue
                        row.update(r)
                        row["rows_per_s"] = n / r["time_s"]
                        row["pairs_per_s"] = r["pairs"] / r["time_s"]
                        row["candidate_reduction"] = "unavailable: subject exposes no public candidate counters"
                        row["status"] = "OK"
                        if engine != "torchsky" and "time_s" in ts:
                            row["speed_ratio_torchsky_vs_this"] = r["time_s"] / ts["time_s"]
                            row["rss_ratio_torchsky_vs_this"] = ts["peak_rss_bytes"] / r["peak_rss_bytes"]
                            row.update({f"parity_{k}": v for k, v in parity.get(engine, {}).items()})
                            if parity.get(engine, {}).get("mismatches", 0):
                                row["status"] = "FAIL"
                                row["reason"] = "pair-ID parity mismatch outside the tie band"
                            if engine == "ckdtree" and n >= args.min_ratio_size:
                                if args.min_ratio_vs_ckdtree is not None and row["speed_ratio_torchsky_vs_this"] < args.min_ratio_vs_ckdtree:
                                    row["status"] = "FAIL"
                                    row["reason"] = f"speed ratio below {args.min_ratio_vs_ckdtree}"
                                if args.max_rss_ratio_vs_ckdtree is not None and row["rss_ratio_torchsky_vs_this"] > args.max_rss_ratio_vs_ckdtree:
                                    row["status"] = "FAIL"
                                    row["reason"] = f"RSS ratio above {args.max_rss_ratio_vs_ckdtree}"
                        rows.append(row)
                        print(f"{row['case']:<28} {engine:<9} {row.get('status')} "
                              f"t={row.get('time_s', float('nan')):.3f}s (build {row.get('build_s', float('nan')):.3f}) "
                              f"rss={row.get('peak_rss_bytes', 0)/2**20:.0f}MiB "
                              f"ratio={row.get('speed_ratio_torchsky_vs_this', float('nan')):.2f} "
                              f"parity={ {k: v for k, v in row.items() if k.startswith('parity_')} }", flush=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))
    failed = [r for r in rows if r.get("status") == "FAIL"]
    print(f"{len(failed)} failing row(s). JSON: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
