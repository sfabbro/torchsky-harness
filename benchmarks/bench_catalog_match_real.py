#!/usr/bin/env python3
"""Real-data association: Gaia DR3 x 2MASS PSC x CatWISE2020 from HATS trees.

Reads cones from HATS collections (default ``/arc/projects/hats``) and runs:

- ``associate(gaia, 2mass)``: Gaia (J2016, full 5-parameter covariance, RV
  where measured) propagated to the 2MASS epoch, against 2MASS error ellipses;
- ``associate(gaia, catwise)``: CatWISE2020 motion solution (MJD 57170, error
  ellipse and proper-motion errors);
- ``associate_many([gaia, 2mass, catwise])``.

Rows without proper motion (Gaia 2-parameter solutions, all of 2MASS) take a
``MotionPrior`` built from the field's Gaia 5-parameter stars.

There is no ground truth here, so the rows report diagnostics a correct
association must satisfy:

- **chi^2 calibration:** ``d^2`` of confident matches (posterior > 0.9) follows
  the truncated chi^2_2. Mean and tail fractions are reported against the theory.
- **completeness:** the EM fit per catalogue.
- **high proper motion:** Gaia stars above 200 mas/yr recovered with posterior
  > 0.9, compared with a naive 2 arcsec nearest match without propagation.
- **N-way:** consistency of N-way with pairwise MAP matches, and the certified
  residual mass.

Timings cover reading, propagation and association. A field's status is
``OK`` when everything runs, the catalogue covariances are valid, and the
confident-match calibration stays within broad sanity bounds. Exact numbers
are recorded for scorecards.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

import torchsky
from torchsky.catalogs import (
    Astrometry,
    Catalog,
    HistogramRatio,
    MotionPrior,
    PartitionPlan,
    count_within,
    associate,
    associate_many,
    match_nearest,
)
from torchsky.sphere import great_circle_distance

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR

D = torch.float64
FIELDS = {
    "baade_window": (270.92, -30.03, 0.12),
    "lmc_bar": (80.9, -69.75, 0.2),
    "north_pole": (0.0, 89.95, 0.25),
    "south_pole": (180.0, -89.95, 0.25),
    "ra_wrap": (0.02, 2.0, 0.4),
    "pleiades": (56.75, 24.12, 0.6),
    "barnards_star": (269.452, 4.6934, 0.2),
}
GAIA_COLS = [
    "source_id", "ra", "dec", "ra_error", "dec_error", "parallax", "parallax_error",
    "pmra", "pmra_error", "pmdec", "pmdec_error", "ra_dec_corr", "ra_parallax_corr",
    "ra_pmra_corr", "ra_pmdec_corr", "dec_parallax_corr", "dec_pmra_corr",
    "dec_pmdec_corr", "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
    "radial_velocity", "radial_velocity_error", "phot_g_mean_mag", "ruwe",
]
TMASS_COLS = ["ra", "decl", "err_maj", "err_min", "err_ang", "jdate", "k_m", "ph_qual"]
CATWISE_COLS = [
    "source_id", "ra_pm", "dec_pm", "sigra_pm", "sigdec_pm", "sigradec_pm",
    "PMRA", "PMDec", "sigPMRA", "sigPMDec", "w1mpro_pm",
]
CATWISE_EPOCH = 2000.0 + (57170.0 + 2400000.5 - 2451545.0) / 365.25


def _cone_tables(root: Path, ra: float, dec: float, radius_deg: float, columns: list[str], ra_col: str, dec_col: str):
    parts = np.loadtxt(root / "partition_info.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    plan = PartitionPlan.from_pixels(torch.from_numpy(parts[:, 0]), torch.from_numpy(parts[:, 1]))
    lon_c, lat_c = plan.centers()
    sep = great_circle_distance(torch.tensor(ra, dtype=D), torch.tensor(dec, dtype=D), lon_c, lat_c) * 3600
    hit = torch.nonzero(sep <= radius_deg * 3600 + plan.radii_arcsec(), as_tuple=True)[0]
    tables = []
    for i in hit.tolist():
        o, p = int(plan.orders[i]), int(plan.pixels[i])
        path = root / "dataset" / f"Norder={o}" / f"Dir={(p // 10000) * 10000}" / f"Npix={p}.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=columns)
        lo = torch.from_numpy(t[ra_col].to_numpy(zero_copy_only=False).astype("f8"))
        la = torch.from_numpy(t[dec_col].to_numpy(zero_copy_only=False).astype("f8"))
        keep = (great_circle_distance(torch.tensor(ra, dtype=D), torch.tensor(dec, dtype=D), lo, la) <= radius_deg).numpy()
        tables.append(t.filter(keep))
    import pyarrow as pa

    return pa.concat_tables(tables) if tables else None, len(hit)


def _col(table, name) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64))


def _gaia(table) -> tuple[Catalog, int]:
    cols = {k: _col(table, k) for k in GAIA_COLS if k not in ("source_id",)}
    cov = Astrometry.gaia_covariance(cols, radial_velocity_error=cols["radial_velocity_error"])
    ok = Astrometry.validity(cov)
    r = torch.nonzero(ok, as_tuple=True)[0]
    cat = Catalog(
        lon=cols["ra"][r], lat=cols["dec"][r], epoch=2016.0,
        pm_lon_coslat_mas_yr=cols["pmra"][r], pm_lat_mas_yr=cols["pmdec"][r],
        parallax_mas=cols["parallax"][r], radial_velocity_km_s=cols["radial_velocity"][r],
        astrometry=Astrometry(cov[r]), columns={"g": cols["phot_g_mean_mag"][r]},
    )
    return cat, int((~ok).sum())


def _tmass(table) -> Catalog:
    jd = _col(table, "jdate")
    ast = Astrometry.from_ellipse(_col(table, "err_maj") * 1000, _col(table, "err_min") * 1000, _col(table, "err_ang"))
    return Catalog(
        lon=_col(table, "ra"), lat=_col(table, "decl"),
        epoch=2000.0 + (jd - 2451545.0) / 365.25, astrometry=ast, columns={"k": _col(table, "k_m")},
    )


def _catwise(table) -> tuple[int, Catalog]:
    n = table.num_rows
    sa, sd = _col(table, "sigra_pm") * 1000, _col(table, "sigdec_pm") * 1000
    co = _col(table, "sigradec_pm") * 1000
    cov = torch.full((n, 6, 6), math.nan, dtype=D)
    cov[:, 0, 0], cov[:, 1, 1] = sa * sa, sd * sd
    cov[:, 0, 1] = cov[:, 1, 0] = torch.sign(co) * co * co  # signed co-sigma
    cov[:, 2, 2] = (_col(table, "sigPMRA") * 1000) ** 2
    cov[:, 3, 3] = (_col(table, "sigPMDec") * 1000) ** 2
    cov[:, 2, 3] = cov[:, 3, 2] = 0.0
    cov[:, :2, 2:4] = 0.0
    cov[:, 2:4, :2] = 0.0
    good = Astrometry.validity(cov) & torch.isfinite(cov[:, :4, :4]).all(-1).all(-1)
    rows = torch.nonzero(good, as_tuple=True)[0]
    return int((~good).sum()), Catalog(
        lon=_col(table, "ra_pm")[rows], lat=_col(table, "dec_pm")[rows], epoch=CATWISE_EPOCH,
        pm_lon_coslat_mas_yr=_col(table, "PMRA")[rows] * 1000, pm_lat_mas_yr=_col(table, "PMDec")[rows] * 1000,
        astrometry=Astrometry(cov[rows]), columns={"w1": _col(table, "w1mpro_pm")[rows]},
    )


def _calibration(res, k2: float) -> dict[str, float]:
    conf = res.posterior > 0.9
    d2 = res.d2[conf]
    if d2.numel() == 0:
        return {"n_confident": 0}
    # Truncated chi^2_2 (gate k^2): mean and P(d^2 > 4.61 | gate).
    trunc = 1 - math.exp(-k2 / 2)
    mean_theory = (2 - (k2 + 2) * math.exp(-k2 / 2)) / trunc
    tail_theory = (math.exp(-4.605 / 2) - math.exp(-k2 / 2)) / trunc
    return {
        "n_confident": int(conf.sum()),
        "d2_mean": float(d2.mean()),
        "d2_mean_theory_all_true_pairs": mean_theory,
        "d2_tail_p90": float((d2 > 4.605).float().mean()),
        "d2_tail_p90_theory_all_true_pairs": tail_theory,
    }


def run_field(name: str, root: Path, ra: float, dec: float, r: float) -> dict[str, Any]:
    row: dict[str, Any] = {"case": name, "library": "torchsky", "ra": ra, "dec": dec, "radius_deg": r}
    t0 = time.perf_counter()
    gtab, gp = _cone_tables(root / "gaia" / "dr3", ra, dec, r, GAIA_COLS, "ra", "dec")
    ttab, tp = _cone_tables(root / "2mass" / "psc", ra, dec, r, TMASS_COLS, "ra", "decl")
    ctab, cp = _cone_tables(root / "catwise" / "2020", ra, dec, r, CATWISE_COLS, "ra_pm", "dec_pm")
    row["read_s"] = time.perf_counter() - t0
    row["partitions_read"] = {"gaia": gp, "2mass": tp, "catwise": cp}
    if gtab is None or ttab is None or ctab is None:
        row.update(status="UNAVAILABLE", reason="a catalogue has no rows in this cone")
        return row
    gaia, gaia_bad = _gaia(gtab)
    tmass = _tmass(ttab)
    cw_bad, cw = _catwise(ctab)
    row["n"] = {"gaia": len(gaia), "2mass": len(tmass), "catwise": len(cw)}
    row["dropped_invalid_covariance"] = {"gaia": gaia_bad, "catwise": cw_bad}
    level = gaia.to_state().motion_level()
    row["gaia_levels"] = {str(v): int((level == v).sum()) for v in (2, 4, 6)}
    has_pm = level >= 4
    prior = MotionPrior.from_reference(
        gaia.lon[has_pm], gaia.lat[has_pm], gaia.pm_lon_coslat_mas_yr[has_pm], gaia.pm_lat_mas_yr[has_pm],  # type: ignore[index]
        order=8, min_count=32,
    )
    k2 = -2 * math.log(1e-4)
    t_epoch = float(torch.median(torch.as_tensor(tmass.epoch)))
    t1 = time.perf_counter()
    # Pair epochs: each Gaia row is propagated to the 2MASS row's own date.
    colour_edges = [torch.linspace(-10.0, 6.0, 65, dtype=D)]
    g2t = associate(
        gaia, tmass, epoch="right", motion_prior=prior, max_radius_arcsec=60.0,
        assign="optimal", calibrate="scale_floor",
        terms=(HistogramRatio(tmass.columns["k"], colour_edges, left_values=gaia.columns["g"]),),
    )
    row["gaia_2mass_s"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    # CatWISE errors depend on blending: calibrate per number of Gaia sources
    # within the WISE beam (6 arcsec) of each CatWISE source.
    blend = count_within(cw.lon, cw.lat, gaia.lon, gaia.lat, 6.0).to(D)
    g2c = associate(
        gaia, cw, epoch="right", motion_prior=prior, max_radius_arcsec=60.0,
        assign="optimal", calibrate="scale_floor", calibrate_by=blend,
        calibrate_edges=torch.tensor([0.0, 1.5, 2.5, 4.5, 1e9], dtype=D),
        terms=(HistogramRatio(cw.columns["w1"], colour_edges, left_values=gaia.columns["g"]),),
    )
    row["gaia_catwise_s"] = time.perf_counter() - t1
    # Variant: CatWISE errors calibrated per W1 magnitude bin (in crowded fields
    # every source is blended, so the blend-count bins collapse to one).
    g2c_mag = associate(
        gaia, cw, epoch="right", motion_prior=prior, max_radius_arcsec=60.0,
        assign="optimal", calibrate="scale_floor", calibrate_by=cw.columns["w1"],
        calibrate_edges=torch.tensor([0.0, 11.0, 12.5, 13.5, 14.5, 15.5, 30.0], dtype=D),
        terms=(HistogramRatio(cw.columns["w1"], colour_edges, left_values=gaia.columns["g"]),),
    )
    for key, res in (("gaia_2mass", g2t), ("gaia_catwise", g2c), ("gaia_catwise_magbins", g2c_mag)):
        row[key] = {
            "completeness": res.completeness, "matched": int(res.match.sum()), "capped": res.n_capped,
            "fitted_right_error_scale": res.right_scale.tolist() if isinstance(res.right_scale, torch.Tensor) else res.right_scale,
            "fitted_floor_mas": (
                (res.floor_mas2.diagonal(dim1=-2, dim2=-1).mean(-1) ** 0.5).tolist()
                if res.floor_mas2 is not None else 0.0
            ),
            **_calibration(res, k2),
        }
    # High proper motion: association vs naive 2 arcsec nearest at stored epochs.
    pm = torch.hypot(gaia.pm_lon_coslat_mas_yr.nan_to_num(0), gaia.pm_lat_mas_yr.nan_to_num(0))  # type: ignore[union-attr]
    fast = torch.nonzero(pm > 200.0, as_tuple=True)[0]
    if fast.numel():
        confident = torch.zeros(len(gaia), dtype=torch.bool)
        confident[g2t.left_index[g2t.match & (g2t.posterior > 0.9)]] = True
        naive = match_nearest(gaia.lon[fast], gaia.lat[fast], tmass.lon, tmass.lat, k=1, max_radius_arcsec=2.0)
        row["high_pm"] = {
            "n": int(fast.numel()),
            "assoc_recovered": int(confident[fast].sum()),
            "naive_2arcsec_recovered": int((naive.index[:, 0] >= 0).sum()),
            "max_pm_mas_yr": float(pm[fast].max()),
        }
    t1 = time.perf_counter()
    many = associate_many([gaia, tmass, cw], motion_prior=prior, max_radius_arcsec=60.0, budget=4096)
    row["nway_s"] = time.perf_counter() - t1
    best = many.members[many.best]
    pair_t = torch.full((len(gaia),), -1, dtype=torch.int64)
    pair_t[g2t.left_index[g2t.match]] = g2t.right_index[g2t.match]
    agree = (best[:, 1] == pair_t)
    row["nway"] = {
        "completeness": many.completeness,
        "map_agrees_with_pairwise_2mass": float(agree.float().mean()),
        "max_residual_mass": float(many.residual_mass.max()),
        "n_hypotheses": int(many.members.shape[0]),
    }
    cal = row["gaia_2mass"]
    sane = cal.get("n_confident", 0) > 0 and 0.5 < cal.get("d2_mean", 0.0) < 4.0
    row["status"] = "OK" if sane else "FAIL"
    if not sane:
        row["reason"] = "confident 2MASS matches are not chi^2-calibrated (catalogue errors or model)"
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hats-root", type=Path, default=Path("/arc/projects/hats"))
    p.add_argument("--fields", default=",".join(FIELDS))
    p.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "catalog_match_real.json")
    args = p.parse_args()
    rows = []
    for name in args.fields.split(","):
        ra, dec, r = FIELDS[name]
        try:
            row = run_field(name, args.hats_root, ra, dec, r)
        except Exception as exc:  # recorded, not swallowed: the row fails
            row = {"case": name, "library": "torchsky", "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
        row["subject_version"] = torchsky.__version__
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2, default=float))
    failed = [r for r in rows if r.get("status") != "OK"]
    print(f"{len(failed)} field(s) not OK. JSON: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
