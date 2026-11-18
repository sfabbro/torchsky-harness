#!/usr/bin/env python3
"""Exact crossmatch correctness: torchsky public API against independent oracles.

The harness owns the oracles: brute-force float64 atan2 on NumPy, an mpmath
boundary sample, scipy ``cKDTree`` on chord radii, and astropy
``search_around_sky`` / ``match_to_catalog_sky``. Every named adversarial case
must reproduce the oracle exactly outside a float64 tie band:

- poles, including exactly +-90 deg
- the RA wrap, and the RA wrap at the pole
- duplicates and equidistant ties
- a radius ladder from 1 mas to 1 deg
- per-row radii spanning 5 dex
- a Plummer cluster
- two-sided sum, quadrature and max rules

Subject APIs that do not exist are reported as ``UNAVAILABLE``, which fails
the claim's required rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torchsky
from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR

from torchsky_harness.benchmarks._catalog_match_oracle import (  # noqa: E402
    ARCSEC_PER_RAD,
    Case,
    adversarial_cases,
    compare_all,
    compare_best,
    mpmath_separation,
    nearest,
    separations,
    threshold,
)

D = torch.float64


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))


def _api(name: str) -> Any | None:
    import torchsky.catalogs as cat

    return getattr(cat, name, None)


def _pairs(res: Any) -> set[tuple[int, int]]:
    return set(zip(res.left_index.tolist(), res.right_index.tolist(), strict=True))


def _row(case: str, api: str, cmp=None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"case": case, "library": "torchsky", "api": api}
    if cmp is not None:
        row.update(
            status="OK" if cmp.ok else "FAIL",
            missing=cmp.missing,
            extra=cmp.extra,
            ambiguous=cmp.ambiguous,
            expected=cmp.expected,
        )
    row.update(extra)
    return row


def _sep_error(res: Any, sep: np.ndarray) -> float:
    if res.left_index.numel() == 0:
        return 0.0
    got = res.separation_arcsec.detach().cpu().numpy()
    want = sep[res.left_index.numpy(), res.right_index.numpy()]
    return float(np.max(np.abs(got - want)))


def run_case(case: Case) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    (la, lb), (ra, rb) = case.left, case.right
    n, m = len(la), len(ra)
    sep = separations(la, lb, ra, rb)
    thr = threshold(n, m, case.radius_left, case.radius_right, case.combine)
    L, R = (_t(la), _t(lb)), (_t(ra), _t(rb))
    rl = case.radius_left if np.isscalar(case.radius_left) else _t(np.asarray(case.radius_left))

    if case.radius_right is None:
        cs = _api("crossmatch_sky")
        res = cs(*L, *R, radius_arcsec=rl, find="all")
        rows.append(_row(case.name, "crossmatch_sky:all", compare_all(_pairs(res), sep, thr),
                         max_sep_err_arcsec=_sep_error(res, sep)))
        res = cs(*L, *R, radius_arcsec=rl, find="best")
        best = dict(zip(res.left_index.tolist(), res.right_index.tolist(), strict=True))
        rows.append(_row(case.name, "crossmatch_sky:best", compare_best(best, sep, thr)))
    elif case.combine == "sum":
        res = _api("crossmatch_sky")(*L, *R, left_err_arcsec=_t(np.asarray(case.radius_left)),
                                     right_err_arcsec=_t(case.radius_right), find="all")
        rows.append(_row(case.name, "crossmatch_sky:skyerr", compare_all(_pairs(res), sep, thr)))

    mw = _api("match_within")
    if mw is None:
        rows.append(_row(case.name, "match_within", status="UNAVAILABLE",
                         reason="torchsky.catalogs.match_within not provided by subject"))
    else:
        kw = {} if case.radius_right is None else {
            "right_radius_arcsec": _t(case.radius_right), "combine": case.combine}
        res = mw(*L, *R, rl, **kw)
        rows.append(_row(case.name, "match_within", compare_all(_pairs(res), sep, thr),
                         max_sep_err_arcsec=_sep_error(res, sep)))
        cw = _api("count_within")
        counts = cw(*L, *R, rl, **kw).numpy()
        inside = sep <= thr
        amb = (np.abs(sep - thr) <= 1e-8).any(axis=1)
        bad = int(np.count_nonzero((counts != inside.sum(axis=1)) & ~amb))
        rows.append(_row(case.name, "count_within", status="OK" if bad == 0 else "FAIL",
                         mismatched_rows=bad, ambiguous=int(amb.sum())))

    if case.radius_right is None:
        mn = _api("match_nearest")
        if mn is None:
            rows.append(_row(case.name, "match_nearest", status="UNAVAILABLE",
                             reason="torchsky.catalogs.match_nearest not provided by subject"))
        else:
            for k in (1, 2):
                got = mn(*L, *R, k=k)
                want_i, want_s, amb = nearest(sep, k)
                gi = got.index.numpy()
                bad = int(np.count_nonzero((gi != want_i).any(axis=1) & ~amb))
                err = np.abs(np.where(np.isfinite(want_s), got.separation_arcsec.numpy() - want_s, 0.0))[~amb]
                rows.append(_row(case.name, f"match_nearest:k{k}", status="OK" if bad == 0 else "FAIL",
                                 mismatched_rows=bad, ambiguous=int(amb.sum()),
                                 max_sep_err_arcsec=float(err.max()) if err.size else 0.0))
    return rows


def comparator_rows(case: Case) -> list[dict[str, Any]]:
    """Cross-check the oracles themselves against scipy and astropy."""
    rows: list[dict[str, Any]] = []
    if case.radius_right is not None or not np.isscalar(case.radius_left):
        return rows
    (la, lb), (ra, rb) = case.left, case.right
    sep = separations(la, lb, ra, rb)
    thr = threshold(len(la), len(ra), case.radius_left, None, "sum")
    from scipy.spatial import cKDTree

    from torchsky_harness.benchmarks._catalog_match_oracle import unit

    chord = 2.0 * math.sin(math.radians(case.radius_left / 3600.0) / 2.0)
    lists = cKDTree(unit(ra, rb)).query_ball_point(unit(la, lb), chord)
    kd = {(i, int(j)) for i, js in enumerate(lists) for j in js}
    rows.append({"case": case.name, "library": "scipy_ckdtree", "api": "query_ball_point",
                 **_cmp_dict(compare_all(kd, sep, thr))})
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord, search_around_sky

        il, ir, _, _ = search_around_sky(SkyCoord(la * u.deg, lb * u.deg), SkyCoord(ra * u.deg, rb * u.deg),
                                         case.radius_left * u.arcsec)
        ap = set(zip(il.tolist(), ir.tolist()))
        rows.append({"case": case.name, "library": "astropy", "api": "search_around_sky",
                     **_cmp_dict(compare_all(ap, sep, thr))})
    except ImportError as exc:  # pragma: no cover - comparator missing
        rows.append({"case": case.name, "library": "astropy", "status": "UNAVAILABLE", "reason": str(exc)})
    return rows


def _cmp_dict(cmp) -> dict[str, Any]:
    # Comparators are recorded, not gated: their own boundary rules may differ
    # inside the tie band and astropy uses a different separation formula.
    return {"status": "OK", "agrees_with_oracle": cmp.ok, "missing": cmp.missing,
            "extra": cmp.extra, "ambiguous": cmp.ambiguous, "expected": cmp.expected}


def mpmath_rows(seed: int, samples: int) -> list[dict[str, Any]]:
    """Oracle separation accuracy against 50-digit arithmetic near thresholds."""
    from torchsky_harness.benchmarks._catalog_match_oracle import jitter, pole_field, wrap_field

    rng = np.random.default_rng(seed)
    worst = 0.0
    worst_subject = 0.0
    mw = _api("match_within")
    for gen in (pole_field, wrap_field):
        la, lb = gen(samples, int(rng.integers(1 << 30)))
        ra, rb = jitter(la, lb, 0.5, int(rng.integers(1 << 30)))
        sep = separations(la, lb, ra, rb)
        idx = np.arange(samples)
        ref = np.array([mpmath_separation(la[i], lb[i], ra[i], rb[i]) for i in idx])
        worst = max(worst, float(np.max(np.abs(sep[idx, idx] - ref))))
        if mw is not None:
            res = mw(_t(la), _t(lb), _t(ra), _t(rb), 5.0)
            same = res.left_index == res.right_index
            got = res.separation_arcsec[same].numpy()
            worst_subject = max(worst_subject, float(np.max(np.abs(got - ref[res.left_index[same].numpy()]))))
    ok = worst <= 1e-6 and worst_subject <= 1e-6
    return [{"case": "mpmath_boundary_separation", "library": "torchsky", "api": "match_within",
             "status": "OK" if ok else "FAIL", "oracle_max_err_arcsec": worst,
             "subject_max_err_arcsec": worst_subject, "tolerance_arcsec": 1e-6,
             "tolerance_uas": 1.0, "unit_note": f"{ARCSEC_PER_RAD:.0f} arcsec/rad"}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scale", type=int, default=1, help="multiply case sizes")
    parser.add_argument("--mpmath-samples", type=int, default=200)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "catalog_match_exact.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for case in adversarial_cases(args.seed, args.scale):
        rows.extend(run_case(case))
        rows.extend(comparator_rows(case))
    rows.extend(mpmath_rows(args.seed, args.mpmath_samples))
    for row in rows:
        row.setdefault("subject_version", torchsky.__version__)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))
    failed = [r for r in rows if r.get("library") == "torchsky" and r.get("status") != "OK"]
    for r in rows:
        flag = r.get("status")
        extra = f" missing={r.get('missing')} extra={r.get('extra')}" if "missing" in r else ""
        print(f"{r['case']:<32} {r['library']:<14} {r.get('api', ''):<22} {flag}{extra}")
    print(f"{len(failed)} torchsky row(s) not OK. JSON: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
