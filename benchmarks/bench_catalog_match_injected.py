#!/usr/bin/env python3
"""Injected-truth association: torchsky associate / associate_many vs nway.

A seeded scene in a 0.5 x 0.5 deg box:

- a primary catalogue;
- a secondary catalogue in which a fraction (completeness) of primaries have a
  counterpart, plus uniform background sources;
- for N-way, a third catalogue built the same way.

Positions are drawn from per-row isotropic Gaussian errors (nway's error model),
so both tools see the same likelihood. nway (Salvato et al. 2018) runs out of
process in the ``nway`` Pixi environment, reading FITS written with torchfits,
with the same prior completeness. torchsky runs with that completeness and
with completeness fitted by EM.

Metrics against the truth:

- completeness and purity of the best counterpart at p_any >= 0.5 and >= 0.9;
- calibration of p_any (expected calibration error, 10 bins);
- correctness of the full N-way tuple;
- wall time.

Rows fail only on execution errors; the numbers are the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchfits import table as fits_table

import torchsky
from torchsky.catalogs import Astrometry, Catalog, associate, associate_many
from torchsky.sphere import MOC

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR

D = torch.float64
BOX = (150.0, 150.5, 2.0, 2.5)
ROOT = Path(__file__).resolve().parents[1]


def _box_area_deg2() -> float:
    ra0, ra1, de0, de1 = BOX
    return math.degrees(math.radians(ra1 - ra0)) * math.degrees(
        math.sin(math.radians(de1)) - math.sin(math.radians(de0))
    )


def _offset(ra, dec, de_mas, dn_mas):
    mas = math.radians(1 / 3.6e6)
    r, d = np.radians(ra), np.radians(dec)
    v = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], 1)
    e = np.stack([-np.sin(r), np.cos(r), np.zeros_like(r)], 1)
    n = np.cross(v, e)
    w = v + mas * (de_mas[:, None] * e + dn_mas[:, None] * n)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return np.degrees(np.arctan2(w[:, 1], w[:, 0])) % 360, np.degrees(np.arcsin(w[:, 2]))


def make_scene(seed: int, n_primary: int, n_background: int, completeness: list[float], err_ranges):
    g = np.random.default_rng(seed)
    ra0, ra1, de0, de1 = BOX
    sin_d = g.uniform(math.sin(math.radians(de0)), math.sin(math.radians(de1)), n_primary)
    true_ra = g.uniform(ra0, ra1, n_primary)
    true_dec = np.degrees(np.arcsin(sin_d))
    cats, truth = [], []
    for c, (lo, hi) in enumerate(err_ranges):
        if c == 0:
            src = np.arange(n_primary)
            n_bg = 0
        else:
            src = np.nonzero(g.random(n_primary) < completeness[c - 1])[0]
            n_bg = n_background
        ra = np.concatenate([true_ra[src], g.uniform(ra0, ra1, n_bg)])
        dec = np.concatenate([true_dec[src], np.degrees(np.arcsin(g.uniform(math.sin(math.radians(de0)), math.sin(math.radians(de1)), n_bg)))])
        sig = g.uniform(lo, hi, len(ra))  # arcsec, 1-sigma per axis
        ra, dec = _offset(ra, dec, g.normal(0, 1, len(ra)) * sig * 1000, g.normal(0, 1, len(ra)) * sig * 1000)
        # Primary rows stay in true-source order; others are shuffled.
        perm = np.arange(len(ra)) if c == 0 else g.permutation(len(ra))
        ra, dec, sig = ra[perm], dec[perm], sig[perm]
        t = np.full(n_primary, -1)
        inv = np.argsort(perm)
        t[src] = inv[np.arange(len(src))]
        cats.append({"ra": ra, "dec": dec, "sig": sig})
        truth.append(t)
    return cats, truth


def _torchsky_catalog(c) -> Catalog:
    sig_mas = torch.from_numpy(c["sig"] * 1000.0)
    return Catalog(lon=torch.from_numpy(c["ra"]), lat=torch.from_numpy(c["dec"]), epoch=2016.0,
                   astrometry=Astrometry.from_errors(sig_mas, sig_mas))


def _metrics(best: np.ndarray, p_any: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    has = truth >= 0
    out: dict[str, float] = {}
    for thr in (0.5, 0.9):
        claim = p_any >= thr
        correct = claim & (best == truth) & has
        out[f"completeness_p{int(thr * 100)}"] = float(correct.sum() / max(has.sum(), 1))
        out[f"purity_p{int(thr * 100)}"] = float(correct.sum() / max(claim.sum(), 1))
    # p_any calibration: is the primary's best counterpart its true counterpart?
    hit = (best == truth) & has
    bins = np.clip((p_any * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        sel = bins == b
        if sel.sum():
            ece += sel.mean() * abs(p_any[sel].mean() - hit[sel].mean())
    out["ece_p_any"] = float(ece)
    return out


def _nway_metrics(members: list[np.ndarray], p_any: np.ndarray, truth: list[np.ndarray]) -> dict[str, float]:
    """MAP tuple and component accuracy (absences included), p_any vs any counterpart."""
    k = len(members)
    out: dict[str, float] = {}
    comps = [members[c] == truth[c + 1] for c in range(k)]
    out["map_tuple_accuracy"] = float(np.all(comps, axis=0).mean())
    for c in range(k):
        out[f"cat{c + 1}_component_accuracy"] = float(comps[c].mean())
    has_any = np.any([truth[c + 1] >= 0 for c in range(k)], axis=0)
    for thr in (0.5, 0.9):
        claim = p_any >= thr
        out[f"any_completeness_p{int(thr * 100)}"] = float((claim & has_any).sum() / max(has_any.sum(), 1))
        out[f"any_purity_p{int(thr * 100)}"] = float((claim & has_any).sum() / max(claim.sum(), 1))
    bins = np.clip((p_any * 10).astype(int), 0, 9)
    out["ece_p_any"] = float(sum(
        (bins == b).mean() * abs(p_any[bins == b].mean() - has_any[bins == b].mean())
        for b in range(10) if (bins == b).any()
    ))
    return out


def run_torchsky(cats, truth, completeness, box: MOC, fitted: bool) -> dict[str, Any]:
    tcats = [_torchsky_catalog(c) for c in cats]
    n0 = len(cats[0]["ra"])
    rows: dict[str, Any] = {}
    t0 = time.perf_counter()
    if len(cats) == 2:
        res = associate(
            tcats[0], tcats[1], gate_pvalue=1e-6, max_radius_arcsec=15.0, right_footprint=box,
            completeness="em" if fitted else completeness[0], assign="none",
        )
        rows["time_s"] = time.perf_counter() - t0
        best = np.full(n0, -1)
        pbest = np.zeros(n0)
        li, rj, post = res.left_index.numpy(), res.right_index.numpy(), res.posterior.numpy()
        order = np.lexsort((-post, li))
        first = np.unique(li[order], return_index=True)[1]
        best[li[order][first]] = rj[order][first]
        pbest[li[order][first]] = post[order][first]
        p_any = 1.0 - res.p_none.numpy()
        rows.update(_metrics(best, p_any, truth[1]))
        rows["completeness_fit"] = res.completeness
    else:
        res = associate_many(
            tcats, gate_pvalue=1e-6, max_radius_arcsec=15.0, footprints=[box] * len(tcats),
            completeness="em" if fitted else list(completeness),
        )
        rows["time_s"] = time.perf_counter() - t0
        members = res.members[res.best].numpy()
        rows.update(_nway_metrics([members[:, c] for c in range(1, len(cats))], res.p_any.numpy(), truth))
        rows["completeness_fit"] = res.completeness[1:]
    return rows


def run_nway(cats, truth, completeness, tmp: Path) -> dict[str, Any]:
    names = ["PRIMARY", "SECOND", "THIRD"][: len(cats)]
    args = []
    for name, c in zip(names, cats):
        path = tmp / f"{name}.fits"
        fits_table.write(
            str(path),
            {"ID": np.arange(len(c["ra"]), dtype=np.int64), "RA": c["ra"], "DEC": c["dec"], "pos_err": c["sig"]},
            header={"SKYAREA": _box_area_deg2()},
            extname=name,
            overwrite=True,
        )
        args += [str(path), ":pos_err"]
    out = tmp / "nway_out.fits"
    prior = float(np.prod(completeness))
    cmd = ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", "nway", "nway.py",
           *args, "--radius", "15", "--prior-completeness", str(prior), "--out", str(out)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"nway failed: {proc.stderr[-500:]}")
    tab = fits_table.read(str(out))
    cols = {k: np.asarray(v) for k, v in tab.items()} if isinstance(tab, dict) else {k: np.asarray(tab[k]) for k in tab.column_names}
    flag = cols["match_flag"] == 1
    n0 = len(cats[0]["ra"])
    prim = cols["PRIMARY_ID"][flag].astype(int)
    p_any = np.zeros(n0)
    p_any[prim] = cols["p_any"][flag]
    rows: dict[str, Any] = {"time_s": elapsed}
    members = []
    for c, name in enumerate(names[1:], start=1):
        best = np.full(n0, -1)
        ids = cols[f"{name}_ID"][flag]
        valid = ids >= 0
        best[prim[valid]] = ids[valid].astype(int)
        members.append(best)
        if len(cats) == 2:
            rows.update(_metrics(best, p_any, truth[c]))
    if len(cats) > 2:
        rows.update(_nway_metrics(members, p_any, truth))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-primary", type=int, default=3000)
    p.add_argument("--n-background", type=int, default=20000)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "catalog_match_injected.json")
    p.add_argument("--no-nway-gate", action="store_true",
                   help="record only: do not fail torchsky rows worse than nway (by > 0.01 on the gated metrics)")
    args = p.parse_args()
    ra0, ra1, de0, de1 = BOX
    box = MOC.from_polygon([ra0, ra1, ra1, ra0], [de0, de0, de1, de1], max_order=16)
    scenarios = {
        "pairwise_dense": ([0.7], [(0.05, 0.3), (0.3, 1.0)]),
        "nway_three": ([0.7, 0.6], [(0.05, 0.3), (0.3, 1.0), (0.2, 0.6)]),
    }
    rows: list[dict[str, Any]] = []
    for name, (comp, errs) in scenarios.items():
        cats, truth = make_scene(args.seed, args.n_primary, args.n_background, comp, errs)
        with tempfile.TemporaryDirectory(prefix="xm_inj_") as tmp:
            for label, fn in (
                ("torchsky", lambda: run_torchsky(cats, truth, comp, box, fitted=False)),
                ("torchsky_em", lambda: run_torchsky(cats, truth, comp, box, fitted=True)),
                ("nway", lambda: run_nway(cats, truth, comp, Path(tmp))),
            ):
                row: dict[str, Any] = {"case": name, "library": label, "n_primary": args.n_primary,
                                       "n_background": args.n_background, "completeness_true": comp}
                try:
                    row.update(fn())
                    row["status"] = "OK"
                except Exception as exc:  # recorded per row
                    row.update(status="FAIL" if label.startswith("torchsky") else "UNAVAILABLE",
                               reason=f"{type(exc).__name__}: {exc}")
                row["subject_version"] = torchsky.__version__
                rows.append(row)
                print(json.dumps(row, default=float), flush=True)
    if not args.no_nway_gate:
        # Gate: torchsky (given completeness) vs nway on the same scene.
        for case in scenarios:
            ts = next(r for r in rows if r["case"] == case and r["library"] == "torchsky")
            nw = next((r for r in rows if r["case"] == case and r["library"] == "nway"), None)
            if nw is None or nw.get("status") != "OK" or ts.get("status") != "OK":
                ts.update(status="FAIL", reason="nway comparator unavailable or torchsky failed")
                continue
            checks = (
                [("completeness_p90", 1, 0.01), ("ece_p_any", -1, 0.01)]
                if case.startswith("pairwise")
                else [("map_tuple_accuracy", 1, 0.01), ("any_completeness_p90", 1, 0.01), ("ece_p_any", -1, 0.01)]
            )
            worse = [k for k, sign, tol in checks if sign * (ts[k] - nw[k]) < -tol]
            if worse:
                ts.update(status="FAIL", reason=f"worse than nway on {worse}")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2, default=float))
    failed = [r for r in rows if r["status"] == "FAIL"]
    print(f"{len(failed)} torchsky row(s) failed. JSON: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
