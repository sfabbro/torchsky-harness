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
with completeness fitted by maximum likelihood, in its exclusive (one-to-one,
default) and independent (nway's) models.

Crowded scenes (``--scenes crowded``, 0.05 x 0.05 deg):

- ``crowded_miscal``: a dense, precise primary (0.5 per arcsec^2, 3 mas) and a
  shallow secondary holding 20 % of them, whose published errors (0.15-0.4")
  are half the true ones (CatWISE in the bulge). nway gets the published
  errors and the true completeness; torchsky fits completeness and the error
  scale/floor (``calibrate="scale_floor"``).
- ``blend_shallow``: WISE-like blends. Primaries follow a G luminosity
  function; each secondary detection is seeded by a primary brighter than
  the W1 limit with no brighter detection within the deblending radius, at
  the PSF- and flux-weighted centroid of the primaries around it, plus
  noise. Truth is the seed. torchsky adds a pair-colour term (W1 - G); nway
  gets its magnitude prior on W1 (``--mag ... auto``).
- ``nway_crowded``: dense primary, two shallow secondaries (20 %, 15 %),
  correct errors: N-way exclusivity alone.

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
from scipy.spatial import cKDTree
from torchfits import table as fits_table

import torchsky
from torchsky.catalogs import Astrometry, Catalog, HistogramRatio, associate, associate_many
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


CROWD_BOX = (150.0, 150.05, 2.0, 2.05)


def _uniform(g, n, box):
    ra0, ra1, de0, de1 = box
    ra = g.uniform(ra0, ra1, n)
    dec = np.degrees(np.arcsin(g.uniform(math.sin(math.radians(de0)), math.sin(math.radians(de1)), n)))
    return ra, dec


def make_crowded_scene(seed: int, n_primary: int, fracs: list[float], n_background: int,
                       pub_range=(0.15, 0.4), true_scale: float = 2.0):
    """Dense precise primary; shallow secondaries with under-estimated errors."""
    g = np.random.default_rng(seed)
    ra, dec = _uniform(g, n_primary, CROWD_BOX)
    cats = [{"ra": ra, "dec": dec, "sig": np.full(n_primary, 0.003)}]
    truth = [np.arange(n_primary)]
    for frac in fracs:
        src = np.nonzero(g.random(n_primary) < frac)[0]
        bra, bdec = _uniform(g, n_background, CROWD_BOX)
        r0 = np.concatenate([ra[src], bra])
        d0 = np.concatenate([dec[src], bdec])
        pub = g.uniform(*pub_range, len(r0))
        sig_true = np.hypot(true_scale * pub, 0.0)
        r1, d1 = _offset(r0, d0, g.normal(0, 1, len(r0)) * sig_true * 1000, g.normal(0, 1, len(r0)) * sig_true * 1000)
        perm = g.permutation(len(r0))
        t = np.full(n_primary, -1)
        t[src] = np.argsort(perm)[: len(src)]
        cats.append({"ra": r1[perm], "dec": d1[perm], "sig": pub[perm]})
        truth.append(t)
    return cats, truth


def make_blend_scene(seed: int, n_primary: int, w1_limit: float = 16.5, deblend_arcsec: float = 6.0,
                     psf_sigma_arcsec: float = 2.6, n_background: int = 200, pub_range=(0.08, 0.3),
                     true_scale: float = 1.5):
    """WISE-like blended secondary of a deep primary (see the module docstring)."""
    g = np.random.default_rng(seed)
    ra, dec = _uniform(g, n_primary, CROWD_BOX)
    # dN/dG ~ 10^(0.35 G) on [12, 21]
    u = g.random(n_primary)
    a = 0.35 * math.log(10)
    gmag = np.log(np.exp(a * 12) + u * (np.exp(a * 21) - np.exp(a * 12))) / a
    w1 = gmag - g.normal(1.2, 0.4, n_primary)
    flux = 10 ** (-0.4 * w1)
    xyz = np.stack([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                    np.cos(np.radians(dec)) * np.sin(np.radians(ra)), np.sin(np.radians(dec))], 1)
    tree = cKDTree(xyz)
    rad = math.radians(1 / 3600)
    order = np.argsort(w1)
    seeds: list[int] = []
    det_xyz: list[np.ndarray] = []
    det_w1: list[float] = []
    for i in order:
        if w1[i] > w1_limit:
            break
        if det_xyz:
            d = np.linalg.norm(np.asarray(det_xyz) - xyz[i], axis=1) / rad
            if (d < deblend_arcsec).any():
                continue
        near = tree.query_ball_point(xyz[i], 3 * psf_sigma_arcsec * rad)
        d = np.linalg.norm(xyz[near] - xyz[i], axis=1) / rad
        wgt = flux[near] * np.exp(-0.5 * (d / psf_sigma_arcsec) ** 2)
        c = (wgt[:, None] * xyz[near]).sum(0)
        seeds.append(int(i))
        det_xyz.append(c / np.linalg.norm(c))
        det_w1.append(float(-2.5 * np.log10(wgt.sum())))
    c = np.asarray(det_xyz)
    r0 = np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 360
    d0 = np.degrees(np.arcsin(c[:, 2]))
    bra, bdec = _uniform(g, n_background, CROWD_BOX)
    r0 = np.concatenate([r0, bra])
    d0 = np.concatenate([d0, bdec])
    w1_obs = np.concatenate([np.asarray(det_w1), g.uniform(w1_limit - 2, w1_limit, n_background)])
    pub = g.uniform(*pub_range, len(r0))
    sig = true_scale * pub
    r1, d1 = _offset(r0, d0, g.normal(0, 1, len(r0)) * sig * 1000, g.normal(0, 1, len(r0)) * sig * 1000)
    perm = g.permutation(len(r0))
    t = np.full(n_primary, -1)
    t[np.asarray(seeds)] = np.argsort(perm)[: len(seeds)]
    cats = [{"ra": ra, "dec": dec, "sig": np.full(n_primary, 0.003), "mag": gmag},
            {"ra": r1[perm], "dec": d1[perm], "sig": pub[perm], "mag": w1_obs[perm]}]
    return cats, [np.arange(n_primary), t]


def _crowd_box_area_deg2() -> float:
    ra0, ra1, de0, de1 = CROWD_BOX
    return math.degrees(math.radians(ra1 - ra0)) * math.degrees(
        math.sin(math.radians(de1)) - math.sin(math.radians(de0))
    )


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


def run_torchsky(cats, truth, completeness, box: MOC, fitted: bool, exclusive: bool = True,
                 calibrate: str = "none", colour: bool = False) -> dict[str, Any]:
    tcats = [_torchsky_catalog(c) for c in cats]
    n0 = len(cats[0]["ra"])
    rows: dict[str, Any] = {}
    t0 = time.perf_counter()
    if len(cats) == 2:
        terms = ()
        if colour:
            terms = (HistogramRatio(torch.from_numpy(cats[1]["mag"]), [torch.linspace(-6.0, 4.0, 41, dtype=D)],
                                    left_values=torch.from_numpy(cats[0]["mag"])),)
        res = associate(
            tcats[0], tcats[1], gate_pvalue=1e-6, max_radius_arcsec=15.0, right_footprint=box,
            completeness="em" if fitted else completeness[0], assign="none", exclusive=exclusive,
            calibrate=calibrate, terms=terms,
        )
        rows["time_s"] = time.perf_counter() - t0
        best = np.full(n0, -1)
        pbest = np.zeros(n0)
        li, rj, post = res.left_index.numpy(), res.right_index.numpy(), res.posterior.detach().numpy()
        order = np.lexsort((-post, li))
        first = np.unique(li[order], return_index=True)[1]
        best[li[order][first]] = rj[order][first]
        pbest[li[order][first]] = post[order][first]
        p_any = 1.0 - res.p_none.detach().numpy()
        rows.update(_metrics(best, p_any, truth[1]))
        rows["completeness_fit"] = res.completeness
        if calibrate != "none":
            rows["right_scale_fit"] = float(res.right_scale)
            rows["floor_mas_fit"] = float(res.floor_mas2.diagonal().mean()) ** 0.5
    else:
        res = associate_many(
            tcats, gate_pvalue=1e-6, max_radius_arcsec=15.0, footprints=[box] * len(tcats),
            completeness="em" if fitted else list(completeness), exclusive=exclusive,
        )
        rows["time_s"] = time.perf_counter() - t0
        members = res.members[res.best].numpy()
        rows.update(_nway_metrics([members[:, c] for c in range(1, len(cats))], res.p_any.numpy(), truth))
        rows["completeness_fit"] = res.completeness[1:]
    return rows


def run_nway(cats, truth, completeness, tmp: Path, area_deg2: float | None = None, mag: bool = False,
             timeout_s: float | None = None) -> dict[str, Any]:
    names = ["PRIMARY", "SECOND", "THIRD"][: len(cats)]
    args = []
    for name, c in zip(names, cats):
        path = tmp / f"{name}.fits"
        cols = {"ID": np.arange(len(c["ra"]), dtype=np.int64), "RA": c["ra"], "DEC": c["dec"], "pos_err": c["sig"]}
        if "mag" in c:
            cols["MAG"] = c["mag"]
        fits_table.write(
            str(path),
            cols,
            header={"SKYAREA": area_deg2 if area_deg2 is not None else _box_area_deg2()},
            extname=name,
            overwrite=True,
        )
        args += [str(path), ":pos_err"]
    out = tmp / "nway_out.fits"
    prior = float(np.prod(completeness))
    extra = ["--mag", "SECOND:MAG", "auto"] if mag else []
    cmd = ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", "nway", "nway.py",
           *args, "--radius", "15", "--prior-completeness", str(prior), *extra, "--out", str(out)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"nway did not finish within {timeout_s:.0f} s") from exc
    mag_note = None
    if proc.returncode != 0 and mag and "too few secure matches" in proc.stdout + proc.stderr:
        # nway's automatic magnitude histogram needs enough secure matches.
        mag_note = "nway --mag auto refused (too few secure matches); run without the magnitude prior"
        cmd = [c for c in cmd if c not in extra]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"nway failed: {(proc.stdout + proc.stderr)[-800:]}")
    tab = fits_table.read(str(out))
    cols = {k: np.asarray(v) for k, v in tab.items()} if isinstance(tab, dict) else {k: np.asarray(tab[k]) for k in tab.column_names}
    flag = cols["match_flag"] == 1
    n0 = len(cats[0]["ra"])
    prim = cols["PRIMARY_ID"][flag].astype(int)
    p_any = np.zeros(n0)
    p_any[prim] = cols["p_any"][flag]
    rows: dict[str, Any] = {"time_s": elapsed}
    if mag:
        rows["mag_prior"] = mag_note or "nway --mag SECOND:MAG auto"
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


def _run_rows(case: str, base: dict[str, Any], runners) -> list[dict[str, Any]]:
    rows = []
    for label, fn in runners:
        row: dict[str, Any] = {"case": case, "library": label, **base}
        try:
            row.update(fn())
            row["status"] = "OK"
        except Exception as exc:  # recorded per row
            row.update(status="FAIL" if label.startswith("torchsky") else "UNAVAILABLE",
                       reason=f"{type(exc).__name__}: {exc}")
        row["subject_version"] = torchsky.__version__
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    return rows


def _gate(rows, case: str, label: str, checks, require_nway: bool = True) -> None:
    ts = next(r for r in rows if r["case"] == case and r["library"] == label)
    nw = next((r for r in rows if r["case"] == case and r["library"] == "nway"), None)
    if not require_nway and ts.get("status") == "OK" and (nw is None or nw.get("status") != "OK"):
        ts["nway_gate"] = "not applied: nway did not complete"
        return
    if nw is None or nw.get("status") != "OK" or ts.get("status") != "OK":
        ts.update(status="FAIL", reason="nway comparator unavailable or torchsky failed")
        return
    worse = [k for k, sign, tol in checks if sign * (ts[k] - nw[k]) < -tol]
    if worse:
        ts.update(status="FAIL", reason=f"worse than nway on {worse}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-primary", type=int, default=3000)
    p.add_argument("--n-background", type=int, default=20000)
    p.add_argument("--n-crowded", type=int, default=12000, help="primaries in the crowded 0.05 deg box")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--scenes", type=lambda v: v.split(","), default=["uniform", "crowded"],
                   help="uniform (pairwise_dense, nway_three) and/or crowded")
    p.add_argument("--nway-timeout", type=float, default=1200.0,
                   help="seconds before an nway run on a crowded scene is recorded as unfinished")
    p.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "catalog_match_injected.json")
    p.add_argument("--no-nway-gate", action="store_true",
                   help="record only: do not fail torchsky rows worse than nway (by > 0.01 on the gated metrics)")
    args = p.parse_args()
    rows: list[dict[str, Any]] = []
    gates: list[tuple[str, str, list, bool]] = []
    pair_checks = [("completeness_p90", 1, 0.01), ("ece_p_any", -1, 0.01)]
    nway_checks = [("map_tuple_accuracy", 1, 0.01), ("any_completeness_p90", 1, 0.01), ("ece_p_any", -1, 0.01)]
    if "uniform" in args.scenes:
        ra0, ra1, de0, de1 = BOX
        box = MOC.from_polygon([ra0, ra1, ra1, ra0], [de0, de0, de1, de1], max_order=16)
        scenarios = {
            "pairwise_dense": ([0.7], [(0.05, 0.3), (0.3, 1.0)]),
            "nway_three": ([0.7, 0.6], [(0.05, 0.3), (0.3, 1.0), (0.2, 0.6)]),
        }
        for name, (comp, errs) in scenarios.items():
            cats, truth = make_scene(args.seed, args.n_primary, args.n_background, comp, errs)
            base = {"n_primary": args.n_primary, "n_background": args.n_background, "completeness_true": comp}
            with tempfile.TemporaryDirectory(prefix="xm_inj_") as tmp:
                rows += _run_rows(name, base, (
                    ("torchsky", lambda: run_torchsky(cats, truth, comp, box, fitted=False)),
                    ("torchsky_em", lambda: run_torchsky(cats, truth, comp, box, fitted=True)),
                    ("torchsky_indep", lambda: run_torchsky(cats, truth, comp, box, fitted=False, exclusive=False)),
                    ("nway", lambda: run_nway(cats, truth, comp, Path(tmp))),
                ))
            gates.append((name, "torchsky", pair_checks if name.startswith("pairwise") else nway_checks, True))
    if "crowded" in args.scenes:
        ra0, ra1, de0, de1 = CROWD_BOX
        cbox = MOC.from_polygon([ra0, ra1, ra1, ra0], [de0, de0, de1, de1], max_order=18)
        area = _crowd_box_area_deg2()
        n = args.n_crowded
        crowd = {
            "crowded_miscal": make_crowded_scene(args.seed, n, [0.2], 300),
            "blend_shallow": make_blend_scene(args.seed, n),
            "nway_crowded": make_crowded_scene(args.seed + 1, n, [0.2, 0.15], 300, true_scale=1.0),
        }
        for name, (cats, truth) in crowd.items():
            comp = [float((t >= 0).mean()) for t in truth[1:]]
            base = {"n_primary": n, "n_secondary": [len(c["ra"]) for c in cats[1:]], "completeness_true": comp,
                    "primary_per_arcsec2": n / (area * 3600**2)}
            cal = "none" if name == "nway_crowded" else "scale_floor"
            colour = name == "blend_shallow"
            with tempfile.TemporaryDirectory(prefix="xm_inj_") as tmp:
                rows += _run_rows(name, base, (
                    ("torchsky", lambda: run_torchsky(cats, truth, comp, cbox, fitted=True, calibrate=cal,
                                                      colour=colour)),
                    ("torchsky_indep", lambda: run_torchsky(cats, truth, comp, cbox, fitted=True, exclusive=False,
                                                            calibrate=cal, colour=colour)),
                    ("torchsky_plain", lambda: run_torchsky(cats, truth, comp, cbox, fitted=False, exclusive=False)),
                    ("nway", lambda: run_nway(cats, truth, comp, Path(tmp), area_deg2=area, mag=colour,
                                              timeout_s=args.nway_timeout)),
                ))
            gates.append((name, "torchsky", pair_checks if len(cats) == 2 else nway_checks, False))
    if not args.no_nway_gate:
        # Gate: torchsky (exclusive, default) vs nway on the same scene.
        for case, label, checks, strict in gates:
            _gate(rows, case, label, checks, require_nway=strict)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2, default=float))
    failed = [r for r in rows if r["status"] == "FAIL"]
    print(f"{len(failed)} torchsky row(s) failed. JSON: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
