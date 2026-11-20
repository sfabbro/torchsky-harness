"""Tests for benchmarks/updc/compare_catalogs_vs_reference.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from torchsky_harness.benchmarks.updc.compare_catalogs_vs_reference import (  # noqa: E402
    compare_catalogs,
    main,
)


def _native_ops_healthy() -> bool:
    try:
        import torch

        import torchsky._cpp  # noqa: F401 - registers torch.ops.torchsky

        torch.ops.torchsky.healpix_ring2nest(4, torch.arange(12))
        return True
    except Exception:
        return False


# crossmatch_sky dispatches through native HEALPix ops; skip cleanly on
# platforms where the extension ABI is currently broken (macOS arm64 +
# conda torch 2.13 at main; linux CI builds and runs these).
pytestmark = pytest.mark.skipif(
    not _native_ops_healthy(),
    reason="torchsky native extension non-functional on this platform",
)


def _synthetic_pair(
    n_matched: int = 25,
    seed: int = 3,
    *,
    extra_candidates: int = 4,
    missing_reference: int = 6,
    offset_arcsec: float = 0.05,
):
    rng = np.random.default_rng(seed)
    ra0, dec0 = 150.0, 2.0
    ra = ra0 + rng.uniform(-0.01, 0.01, size=n_matched)
    dec = dec0 + rng.uniform(-0.01, 0.01, size=n_matched)
    flux = rng.uniform(0.5, 10.0, size=n_matched)

    # Candidate = reference shifted by a small offset (well inside radius).
    dra_arcsec = offset_arcsec / np.cos(np.deg2rad(dec))
    cand_ra = np.concatenate([ra + dra_arcsec / 3600.0, ra0 + 0.5 * np.ones(extra_candidates)])
    cand_dec = np.concatenate([dec + offset_arcsec / 3600.0, dec0 + 0.5 * np.ones(extra_candidates)])
    cand_flux = np.concatenate([flux * 1.02, np.full(extra_candidates, 0.1)])

    ref_ra = np.concatenate([ra, ra[:missing_reference] + 0.5])
    ref_dec = np.concatenate([dec, dec[:missing_reference] + 0.5])
    ref_flux = np.concatenate([flux, flux[:missing_reference]])
    return cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux


def test_counts_and_statistics_on_synthetic_pairs() -> None:
    cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux = _synthetic_pair()
    report = compare_catalogs(
        cand_ra,
        cand_dec,
        cand_flux,
        ref_ra,
        ref_dec,
        ref_flux,
        radius_arcsec=0.5,
    )
    assert report["matched"] == 25
    assert report["candidate_only"] == 4
    assert report["reference_only"] == 6
    stats = report["separation_arcsec"]
    assert stats["median"] == pytest.approx(0.05 * np.sqrt(2), rel=0.2)
    assert stats["p95"] < 0.5
    ratio = report["flux_ratio"]
    assert ratio["median"] == pytest.approx(1.02, rel=1e-3)


def test_boundary_radius_discriminates() -> None:
    cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux = _synthetic_pair(
        offset_arcsec=0.09, seed=5
    )
    inside = compare_catalogs(
        cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux, radius_arcsec=0.2
    )
    outside = compare_catalogs(
        cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux, radius_arcsec=0.1
    )
    assert inside["matched"] == 25
    assert outside["matched"] == 0


def test_association_is_one_to_one_when_candidates_share_a_reference() -> None:
    reference_ra = np.array([150.0])
    reference_dec = np.array([2.0])
    reference_flux = np.array([1.0])
    candidate_ra = np.array([150.0, 150.0 + 0.02 / 3600.0])
    candidate_dec = np.array([2.0, 2.0])
    candidate_flux = np.array([1.0, 1.0])
    report = compare_catalogs(
        candidate_ra,
        candidate_dec,
        candidate_flux,
        reference_ra,
        reference_dec,
        reference_flux,
        radius_arcsec=0.5,
    )
    assert report["association"] == "maximum_cardinality_one_to_one"
    assert report["matched"] == 1
    assert report["candidate_only"] == 1
    assert report["reference_only"] == 0


def test_snr_bins_reported_when_reference_snr_given() -> None:
    cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux = _synthetic_pair()
    snr = np.linspace(1.0, 80.0, num=ref_ra.size)
    report = compare_catalogs(
        cand_ra,
        cand_dec,
        cand_flux,
        ref_ra,
        ref_dec,
        ref_flux,
        radius_arcsec=0.5,
        reference_snr=snr,
    )
    labels = [entry["snr_bin"] for entry in report["snr_bins"]]
    assert labels == ["<5", "5-10", "10-20", "20-50", ">=50"]
    assert sum(entry["matched"] for entry in report["snr_bins"]) == 25


def test_main_writes_json_report(tmp_path: Path) -> None:
    import torchfits

    cand_ra, cand_dec, cand_flux, ref_ra, ref_dec, ref_flux = _synthetic_pair()
    candidate_path = tmp_path / "candidate.fits"
    reference_path = tmp_path / "reference.fits"
    torchfits.table.write(
        str(candidate_path),
        {"ra": cand_ra, "dec": cand_dec, "flux": cand_flux},
    )
    torchfits.table.write(
        str(reference_path),
        {"ra": ref_ra, "dec": ref_dec, "flux": ref_flux},
    )
    out_path = tmp_path / "qa.json"

    rc = main(
        [
            "--candidate",
            str(candidate_path),
            "--reference",
            str(reference_path),
            "--radius-arcsec",
            "0.5",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["matched"] == 25
