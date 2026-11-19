"""Declared evaluation claims and the verdicts an execution record may carry.

An execution record never guesses scientific validity. A run can only be
``established`` for a claim the harness has declared here, and only when the
declared required rows are present and passing. Evaluations without a declared
claim stay ``not_assessed``; the record says why.

Claim scopes:

``synthetic_truth``
    The evaluation compares against an injected or analytic truth on synthetic
    data. It says nothing about real detector data, real sky, or other regimes.
``reference_comparison``
    The evaluation compares the subject against an independent reference
    implementation on the declared cases.
``real_data``
    The evaluation uses real observations with an explicit, checked reference.
``contract``
    The evaluation checks structural/domain/tensor/concurrency behaviour rather
    than scientific parity: malformed input must be rejected, invalid physical
    domains must stay invalid, and crashes/hangs fail the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from torchsky_harness.subject import sha256

ClaimScope = Literal[
    "synthetic_truth", "reference_comparison", "real_data", "contract"
]


@dataclass(frozen=True)
class EvaluationClaim:
    """The bounded claim a release gate is allowed to certify."""

    scope: ClaimScope
    required_rows: tuple[str, ...]
    comparators: tuple[str, ...] = ()
    required_row_selectors: tuple[tuple[tuple[str, str], ...], ...] = ()
    required_arguments: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = field(default_factory=tuple)
    summary_failures_key: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "required_rows": list(self.required_rows),
            "comparators": list(self.comparators),
            "required_row_selectors": [dict(selector) for selector in self.required_row_selectors],
            "required_arguments": dict(self.required_arguments),
            "limitations": list(self.limitations),
            "summary_failures_key": self.summary_failures_key,
        }


CLAIMS: dict[str, EvaluationClaim] = {
    "bench_native_contracts": EvaluationClaim(
        scope="contract",
        required_rows=(
            "empty_ring_synthesis",
            "empty_healpix_arrays",
            "invalid_healpix_coordinates",
            "tan_back_hemisphere",
            "mol_polar_roundtrip",
            "generalized_hpx",
            "invalid_tpv_metadata",
            "scalar_sht_high_order",
            "spin_sht_concurrency",
            "tiny_arc_geometry",
            "noncontiguous_wcs",
            "dispatcher_healpix_contract",
        ),
        limitations=(
            "fixed adversarial counterexamples, each in a subprocess with a hard timeout",
            "no randomized differential fuzz, sanitizer, or multi-device matrix yet",
            "domain agreement is checked against WCSLIB/ERFA where applicable;"
            " the remaining probes assert documented contracts or numerical bounds",
            "a pass is necessary but not sufficient for release: scientific gates,"
            " packaging, and device evidence remain separate claims",
        ),
    ),
    "bench_domain_fuzz": EvaluationClaim(
        scope="contract",
        required_rows=(
            "wcs_domain_TAN",
            "wcs_domain_SIN",
            "wcs_domain_ARC",
            "wcs_domain_ZEA",
            "wcs_domain_STG",
            "wcs_domain_ZPN",
            "wcs_domain_AZP",
            "wcs_domain_SZP",
            "wcs_domain_AIR",
            "wcs_domain_AIT",
            "wcs_domain_MOL",
            "wcs_domain_HPX",
            "wcs_domain_XPH",
            "wcs_domain_CEA",
            "wcs_domain_MER",
            "wcs_domain_CAR",
            "wcs_domain_SFL",
            "wcs_domain_PAR",
            "wcs_domain_COP",
            "wcs_domain_COE",
            "wcs_domain_COD",
            "wcs_domain_COO",
            "wcs_domain_BON",
            "wcs_domain_PCO",
            "wcs_domain_NCP",
            "hpx_generalized_hk",
            "tpv_forward_random_polynomial",
        ),
        comparators=("astropy",),
        limitations=(
            "one seeded random header per projection plus generalized HPX and TPV;"
            " this searches for domain leaks and systematic coverage loss, not every"
            " parameter combination",
            "pixels are sampled in and just outside the image footprint;"
            " the comparison also tolerates conservative subject invalid masks",
            "no multi-device or 32-bit floating-point fuzz yet",
        ),
    ),
    "bench_coords_astropy": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "icrs_to_galactic",
            "icrs_to_fk5_j2000",
            "icrs_to_barycentric_mean_ecliptic_j2000",
        ),
        comparators=("astropy",),
        limitations=(
            "geometric frame graph only (ICRS, FK5 J2000, Galactic, mean"
            " ecliptic of J2000); observed-place frames are covered separately"
            " by the ERFA comparison benchmark",
            "random finite-sky points; no all-sky pole/seam stress in this row",
            "tolerance 1e-6 asec; astropy is the oracle",
        ),
    ),
    "bench_wcs": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "TAN", "SIN", "ARC", "ZEA", "STG", "ZPN", "AZP", "SZP", "AIR", "AIT",
            "MOL", "HPX", "XPH", "CEA", "MER", "CAR", "SFL", "PAR", "COP", "COE",
            "COD", "COO", "BON", "PCO", "TSC", "CSC", "QSC", "NCP", "TAN_SIP", "TPV",
            "MER_OFF", "CYP_OFF", "AIT_OFF", "HPX_OFF", "SFL_OFF", "XPH_OFF",
            "BON_OFF", "PAR_OFF",
        ),
        comparators=("astropy",),
        limitations=(
            "astropy reference on CPU float64, 20k samples per case",
            "throughput ratios are same-machine, not a pinned-runner claim",
            "reference-latitude-offset cases exercise the center-native rotation"
            " for MER, CYP, AIT, HPX, SFL, XPH, BON, and PAR at CRVAL2 = 20 deg",
            "AIR parity floor ~1e-8 deg comes from the shared AIR formula;"
            " all-sky pole and quad-cube seam inverse conditioning is excluded"
            " from the round-trip thresholds, not from the angular parity gate",
        ),
    ),
    "bench_map_real_cutout": EvaluationClaim(
        scope="reference_comparison",
        required_rows=("resample_bilinear_real",),
        comparators=("reproject",),
        limitations=(
            "single real 256x256 CFHT MegaPipe g-band cutout with its original"
            " TAN WCS; raw survey pixels beyond this cutout are not redistributed",
            "interior of the shared footprint is compared (edge erosion); the"
            " coverage-boundary convention is recorded, not gated",
        ),
    ),
    "bench_map_drizzle_required": EvaluationClaim(
        scope="reference_comparison",
        required_rows=("upstream_drizzle_crpix_translation",),
        comparators=("drizzle",),
        required_row_selectors=(
            (
                ("case", "upstream_drizzle_crpix_translation"),
                ("library", "drizzle"),
                ("variant", "point"),
            ),
            (
                ("case", "upstream_drizzle_crpix_translation"),
                ("library", "drizzle"),
                ("variant", "square"),
            ),
        ),
        required_arguments=(
            ("--acceptance-json", "benchmarks/baselines/map_drizzle_acceptance.json"),
            ("--require-comparators", "drizzle"),
        ),
        limitations=(
            "tiny synthetic 32x32 CRPIX translation with two inputs only",
            "point and square low-level kernels are checked against drizzle 2.2.0"
            " using the committed absolute acceptance contract",
            "not real-data replacement parity and not rejection, DQ, context,"
            " variance, rotation, scale, or general distortion parity",
        ),
    ),
    "bench_sphere_real_map": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "ang2pix_real",
            "interp_real",
            "ud_grade_512_256",
            "ud_grade_512_128",
            "map2alm_real",
            "alm2map_real",
            "alm2cl_real",
            "smooth_real",
        ),
        comparators=("healpy",),
        limitations=(
            "real-sky fixture derived from a locally held 2 GB CFHT MegaPipe"
            " g-band mosaic (stride 8, nside 512, 106 occupied pixels); raw"
            " survey pixels are not redistributed",
            "maps are a single real survey footprint, not a full-sky survey"
            " product; full-sky synthetic cases remain in bench_healpix",
        ),
    ),
    "bench_wcs_real_headers": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "int_wfc_zpn", "spitzer_irac_sip", "ptf_tpv", "sip_only",
            "zpn_edge", "hst_wfpc2_sip", "sip_header", "sip2_header",
            "header_newlines_sip", "validate_tan", "tabular_time",
        ),
        comparators=("astropy",),
        limitations=(
            "vendored real instrument headers from the astropy WCS test suite"
            " (BSD-3, provenance in benchmarks/fixtures/real_wcs/PROVENANCE.md)",
            "lookup-distortion fixtures (D2IM/CPDIS/WCSDVARR) are SKIPPED with a"
            " recorded reason until their parity is wired; they are not claimed",
            "the DSS cutout is skipped because astropy rejects its PC matrix",
        ),
    ),
    "bench_healpix": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "ang2pix_ring", "ang2pix_nested", "pix2ang_ring", "pix2ang_nested",
            "ring2nest", "nest2ring",
        ),
        comparators=("healpy",),
        limitations=(
            "index parity and angle parity at nside 512 only",
            "throughput is one CPU, not a pinned-runner performance claim",
        ),
    ),
    "bench_healpix_advanced": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "neighbors_ring", "neighbors_nested", "interp_weights_ring",
            "interp_weights_nested", "interp_val_ring", "interp_val_nested",
        ),
        comparators=("healpy",),
        limitations=(
            "neighbors and bilinear interpolation parity at nside 512 only",
            "throughput is one CPU, not a pinned-runner performance claim",
        ),
    ),
    "bench_sphere_matrix": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(),
        comparators=("healpy",),
        summary_failures_key="failed_thresholds",
        limitations=(
            "index and angle parity at nside 512; median healpy ratios are"
            " recorded but not gated on macOS (hardware-fingerprinted"
            " performance gates belong on the pinned runner)",
        ),
    ),
    "bench_spin_matrix": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(),
        comparators=("healpy",),
        summary_failures_key="failures",
        limitations=(
            "spin -2 harmonic transform accuracy and median throughput ratio",
            "throughput is one CPU; small-bucket ratios only with the default"
            " case set",
        ),
    ),
    "bench_catalog_compat": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "synthetic_clustered", "real_astropy_votable_catalogs",
            "real_astropy_votable_fits_roundtrip",
        ),
        comparators=("astropy",),
        limitations=(
            "synthetic and astropy-table fixtures, not real survey catalogs",
            "separation parity and throughput only; no completeness or"
            " false-positive claim beyond the evaluator rows",
        ),
    ),

    "bench_catalog_match_exact": EvaluationClaim(
        scope="reference_comparison",
        required_rows=(
            "pole_north", "pole_south", "ra_wrap", "antimeridian_at_pole",
            "coincident_duplicates", "equidistant_ties",
            "radius_ladder_0.001arcsec", "radius_ladder_0.01arcsec",
            "radius_ladder_0.1arcsec", "radius_ladder_1arcsec",
            "radius_ladder_10arcsec", "radius_ladder_60arcsec",
            "radius_ladder_3600arcsec", "per_row_radius_5dex", "plummer_cluster",
            "two_sided_sum", "two_sided_quadrature", "two_sided_max",
            "mpmath_boundary_separation",
        ),
        comparators=("bruteforce", "mpmath", "scipy", "astropy"),
        limitations=(
            "synthetic adversarial geometry at bounded sizes (<= ~2k rows per"
            " side by default); exact set equality outside a 1e-8 arcsec tie band",
            "positions only: no epoch propagation, ellipses or probabilistic"
            " association",
        ),
    ),
    "bench_catalog_match_injected": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("pairwise_dense", "nway_three"),
        comparators=("nway",),
        limitations=(
            "one seeded 0.25 deg^2 scene with isotropic errors (nway's error model)",
            "gated: torchsky not worse than nway by > 0.01 on completeness@0.9,"
            " p_any ECE and N-way MAP tuple accuracy",
        ),
    ),
    "bench_map_cube": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("cube_constant_bands",),
        limitations=(
            "synthetic 2D-band cube assembly only; not a 3D IFU or detector cube",
            "no dense spectral interpolation claim",
        ),
    ),
    "bench_map_ifu": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("ifu_matched_sparse_spectrum",),
        limitations=(
            "matched spectral sampling only; dense interpolation is not claimed",
            "synthetic wavelength-map IFU, not real IFU data",
        ),
    ),
    "bench_map_ifu_spectral_axis": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=(
            "ifu_nonuniform_axis_line_localization",
            "ifu_nonuniform_axis_constant_spectrum",
            "ifu_regular_axis_spectrum_interpolation",
            "ifu_nonuniform_axis_interpolation_envelope",
        ),
        limitations=(
            "line localization and normalization only on irregular axes; the"
            " current kernel interpolates in index space",
            "irregular-axis smooth-spectrum recovery is gated as a recorded"
            " error envelope, not as dense physical interpolation",
            "no flux-conserving spectral redistribution claim",
        ),
    ),
    "bench_map_healpix_multiband": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=(
            "healpix_multiband_flat_multiscale",
            "healpix_multiband_sed_registration",
            "healpix_multiband_plane_photometry",
            "healpix_multiband_roundtrip_cube",
        ),
        limitations=(
            "coadd_healpix samples image values at HEALPix centres; flux-conserving"
            " reprojection is not claimed",
            "no variance/weight product, harmonic/SHT resampling, or healpy"
            " throughput claim",
        ),
    ),
    "bench_native_response": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("native_gaussian_star",),
        limitations=(
            "linear pixel-integrated Gaussian profile only",
            "not a full likelihood, joint, or real-detector claim",
        ),
    ),
    "bench_native_likelihood": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("native_gaussian_likelihood",),
        limitations=(
            "diagonal native Gaussian likelihood on a synthetic stamp",
            "evidence-staleness rejection is contract behaviour, not calibration",
        ),
    ),
    "bench_native_nuisance": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("native_nuisance_precision",),
        limitations=(
            "profiled/calibrated nuisance precision against dense references",
            "no real-detector nuisance calibration claim",
        ),
    ),
}


def _argument(command: Sequence[str], flag: str) -> str | None:
    values = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == flag
    ]
    return values[-1] if values else None


def _row_identity(row: Mapping[str, Any]) -> str | None:
    """Report-row identity: benchmark rows use ``case``, operation rows ``operation``."""
    for key in ("case", "operation"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def _load_report(path: Path) -> tuple[Any, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"evidence report unreadable: {exc}"
    return payload, None


def _report_failures(payload: Any, key: str) -> tuple[list[Any], str | None]:
    if not isinstance(payload, dict):
        return [], "summary evidence report is not an object"
    failures = payload.get(key)
    if not isinstance(failures, list):
        return [], f"summary evidence report has no {key!r} list"
    return failures, None


def _claim_report_errors(claim: EvaluationClaim, payload: Any) -> list[str]:
    errors: list[str] = []
    if claim.summary_failures_key is not None:
        failures, failure_error = _report_failures(payload, claim.summary_failures_key)
        if failure_error is not None:
            errors.append(failure_error)
        elif failures:
            errors.append(
                f"{len(failures)} reported failure(s) in {claim.summary_failures_key}"
            )
        return errors
    if not isinstance(payload, list):
        return ["evidence report is not a row list"]
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        errors.append("evidence report has no rows")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        identity = _row_identity(row)
        if identity is not None:
            by_case.setdefault(identity, []).append(row)
    for required in claim.required_rows:
        matches = by_case.get(required)
        if not matches:
            errors.append(f"missing required row {required}")
            continue
        failed = [
            str(row.get("status", "OK"))
            for row in matches
            if str(row.get("status", "OK")) != "OK"
        ]
        if failed:
            errors.append(f"required row {required} reported status {failed[0]}")
    for selector_items in claim.required_row_selectors:
        selector = dict(selector_items)
        matches = [
            row
            for row in rows
            if all(str(row.get(key)) == value for key, value in selector.items())
        ]
        label = ", ".join(f"{key}={value}" for key, value in selector.items())
        if not matches:
            errors.append(f"missing required row selector {label}")
            continue
        failed = [
            str(row.get("status")) for row in matches if row.get("status") != "OK"
        ]
        if failed:
            errors.append(f"required row selector {label} reported status {failed[0]}")
    return errors


def verdict(
    name: str,
    *,
    returncode: int | None,
    command: Sequence[str],
    working_directory: Path,
) -> dict[str, Any]:
    """Derive the only verdict fields an execution record may carry."""
    claim = CLAIMS.get(name)
    if claim is None:
        return {
            "scientific_validity": "not_assessed",
            "comparison_eligibility": "not_assessed",
            "verdict_reasons": ["no declared claim for this evaluation"],
        }

    reasons: list[str] = []
    comparison_notes: list[str] = []
    record: dict[str, Any] = {
        "claim": claim.as_record(),
    }
    if returncode != 0:
        reasons.append(f"evaluation exited with return code {returncode}")
    for flag, expected in claim.required_arguments:
        actual = _argument(command, flag)
        if actual != expected:
            reasons.append(f"evaluation requires {flag} {expected}")

    report_argument = _argument(command, "--json-out")
    if report_argument is None:
        reasons.append("evaluation declared no --json-out evidence report")
    else:
        report_path = Path(report_argument)
        if not report_path.is_absolute():
            report_path = working_directory / report_path
        payload, error = _load_report(report_path)
        if error is not None:
            reasons.append(error)
        else:
            reasons.extend(_claim_report_errors(claim, payload))
            report_key = (
                "evidence_summary"
                if claim.summary_failures_key is not None
                else "evidence_report"
            )
            record[report_key] = str(report_path.resolve())

    comparison = "suite_owned"
    baseline_argument = _argument(command, "--baseline-json")
    if baseline_argument is not None:
        if "--require-complete-baseline" in command:
            comparison = "baseline_compared"
        else:
            comparison_notes.append(
                "baseline supplied without --require-complete-baseline, so"
                " baseline coverage is not enforced"
            )
    record["scientific_validity"] = "established" if not reasons else "not_established"
    record["comparison_eligibility"] = comparison
    record["verdict_reasons"] = reasons
    record["comparison_notes"] = comparison_notes
    return record


BUNDLE_SCHEMA_VERSION = 1


def _read_record(run_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            (run_dir / "execution.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _digest(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return sha256(path)
    except OSError:
        return None


def _record_report_path(run_dir: Path, record: Mapping[str, Any]) -> Path | None:
    report_value = record.get("evidence_report") or record.get("evidence_summary")
    if not isinstance(report_value, str):
        return None
    report_path = Path(report_value)
    if report_path.is_absolute():
        return report_path
    working_directory = record.get("working_directory")
    if isinstance(working_directory, str):
        return Path(working_directory) / report_path
    return run_dir / report_path


def _run_entry(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    report_path = _record_report_path(run_dir, record)
    subject = record.get("subject") if isinstance(record.get("subject"), dict) else {}
    source = subject.get("source") if isinstance(subject.get("source"), dict) else {}
    logs = record.get("logs") if isinstance(record.get("logs"), list) else []
    entry = {
        "run_dir": str(run_dir.resolve()),
        "evaluation": record.get("evaluation"),
        "execution": record.get("execution"),
        "returncode": record.get("returncode"),
        "scientific_validity": record.get("scientific_validity", "not_assessed"),
        "comparison_eligibility": record.get("comparison_eligibility", "not_assessed"),
        "verdict_reasons": record.get("verdict_reasons", []),
        "claim": record.get("claim"),
        "subject": {
            "distribution": subject.get("distribution"),
            "version": subject.get("version"),
            "import_path": subject.get("import_path"),
            "package_content_sha256": subject.get("package_content_sha256"),
            "source_kind": source.get("kind"),
            "source_sha256": source.get("sha256"),
            "revision": source.get("revision"),
            "dirty": source.get("dirty"),
        },
        "harness": record.get("harness"),
        "execution_record": {
            "path": str((run_dir / "execution.json").resolve()),
            "sha256": _digest(run_dir / "execution.json"),
        },
        "evidence_report": {
            "path": str(report_path.resolve()) if report_path else None,
            "sha256": _digest(report_path),
        },
        "logs": [
            {
                "path": str(item.get("path")),
                "sha256": _digest(run_dir / str(item.get("path")))
                if isinstance(item, dict) and item.get("path")
                else None,
            }
            for item in logs
            if isinstance(item, dict)
        ],
    }
    return entry


def build_bundle(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Collect run records into a release evidence bundle.

    For each declared claim the newest `established` run is selected. Runs that
    cannot be read are reported, never silently dropped.
    """
    entries: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for run_dir in run_dirs:
        record = _read_record(run_dir)
        if record is None:
            unreadable.append(str(run_dir))
            continue
        entries.append(_run_entry(run_dir, record))

    by_evaluation: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_evaluation.setdefault(str(entry.get("evaluation")), []).append(entry)

    selected: dict[str, dict[str, Any]] = {}
    superseded: dict[str, list[str]] = {}
    for evaluation, evaluation_entries in by_evaluation.items():
        established = [
            entry
            for entry in evaluation_entries
            if entry.get("scientific_validity") == "established"
        ]
        pool = established or evaluation_entries
        chosen = max(
            pool,
            key=lambda entry: (
                Path(entry["execution_record"]["path"]).stat().st_mtime
                if Path(entry["execution_record"]["path"]).is_file()
                else 0.0
            ),
        )
        selected[evaluation] = chosen
        others = [
            entry["run_dir"] for entry in pool if entry["run_dir"] != chosen["run_dir"]
        ]
        if others:
            superseded[evaluation] = sorted(others)

    declared = sorted(CLAIMS)
    missing_claims = [
        evaluation
        for evaluation in declared
        if selected.get(evaluation, {}).get("scientific_validity") != "established"
    ]
    extra_established = sorted(
        evaluation
        for evaluation, entry in selected.items()
        if evaluation not in CLAIMS
        and entry.get("scientific_validity") == "established"
    )
    # Readiness is about the declared claims; supporting (undeclared) runs may
    # lag the release subject without blocking it, and are still listed in
    # subject_identities for inspection.
    claim_entries = {
        evaluation: entry
        for evaluation, entry in selected.items()
        if evaluation in CLAIMS
    }
    invalid_reports: dict[str, list[str]] = {}
    for evaluation, entry in claim_entries.items():
        report = entry["evidence_report"]
        report_path_value = report.get("path")
        report_path = (
            Path(report_path_value) if isinstance(report_path_value, str) else None
        )
        errors: list[str] = []
        if report_path is None:
            errors.append("evidence report path missing")
        elif report.get("sha256") is None:
            errors.append("evidence report SHA-256 unavailable")
        if report_path is not None:
            payload, load_error = _load_report(report_path)
            if load_error is not None:
                errors.append(load_error)
            else:
                errors.extend(_claim_report_errors(CLAIMS[evaluation], payload))
        report["revalidated"] = not errors
        report["validation_errors"] = errors
        if errors:
            invalid_reports[evaluation] = errors
    subjects = {
        entry["subject"]["package_content_sha256"] for entry in claim_entries.values()
    }
    harness_revisions = {
        (entry.get("harness") or {}).get("source", {}).get("revision")
        for entry in selected.values()
    }
    claim_harness_revisions = {
        (entry.get("harness") or {}).get("source", {}).get("revision")
        for entry in claim_entries.values()
    }
    incomplete = sorted(
        evaluation
        for evaluation, entry in selected.items()
        if entry.get("execution") != "completed"
    )
    incomplete_claims = sorted(
        evaluation
        for evaluation in CLAIMS
        if selected.get(evaluation, {}).get("execution") != "completed"
    )
    counts: dict[str, int] = {"established": 0, "not_established": 0, "not_assessed": 0}
    for entry in selected.values():
        key = str(entry.get("scientific_validity", "not_assessed"))
        counts[key] = counts.get(key, 0) + 1
    release_ready = (
        not missing_claims
        and not incomplete_claims
        and len(subjects) <= 1
        and not unreadable
        and not invalid_reports
    )
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_ready": release_ready,
        "declared_claims": declared,
        "missing_claims": missing_claims,
        "extra_established": extra_established,
        "subject_identities": sorted(
            {
                (
                    entry["subject"]["package_content_sha256"],
                    entry["subject"]["source_kind"],
                    entry["subject"]["source_sha256"],
                    entry["subject"]["revision"],
                )
                for entry in selected.values()
            },
            key=str,
        ),
        "claim_subject_identities": sorted(
            {
                (
                    entry["subject"]["package_content_sha256"],
                    entry["subject"]["source_kind"],
                    entry["subject"]["source_sha256"],
                    entry["subject"]["revision"],
                )
                for entry in claim_entries.values()
            },
            key=str,
        ),
        "subject_consistent": len(subjects) <= 1,
        "harness_revisions": sorted(str(item) for item in harness_revisions),
        "claim_harness_revisions": sorted(
            str(item) for item in claim_harness_revisions
        ),
        "incomplete_runs": incomplete,
        "incomplete_claims": incomplete_claims,
        "unreadable_runs": unreadable,
        "invalid_reports": invalid_reports,
        "superseded_runs": superseded,
        "verdict_counts": counts,
        "entries": selected,
    }


def render_bundle_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Evidence bundle",
        "",
        f"- release ready: **{bundle['release_ready']}**",
        f"- generated: {bundle['generated_at']}",
        f"- subject consistent: {bundle['subject_consistent']}",
        f"- verdicts: {bundle['verdict_counts']}",
    ]
    performance = bundle.get("performance")
    if isinstance(performance, dict):
        lines.append(
            f"- scientific release ready: **{bundle.get('scientific_release_ready')}**"
        )
        lines.append(
            "- performance ready:"
            f" **{performance.get('release_performance_ready', False)}**"
        )
        performance_errors = performance.get("errors")
        if isinstance(performance_errors, list) and performance_errors:
            lines.append(f"- performance blockers: {len(performance_errors)}")
    if bundle["missing_claims"]:
        lines.append(f"- missing claims: {', '.join(bundle['missing_claims'])}")
    if bundle["incomplete_runs"]:
        lines.append(f"- incomplete runs: {', '.join(bundle['incomplete_runs'])}")
    if bundle["unreadable_runs"]:
        lines.append(f"- unreadable run dirs: {', '.join(bundle['unreadable_runs'])}")
    lines += [
        "",
        "| evaluation | validity | comparison | subject | run |",
        "|---|---|---|---|---|",
    ]
    for evaluation in sorted(bundle["entries"]):
        entry = bundle["entries"][evaluation]
        subject = entry["subject"]
        identity = subject.get("package_content_sha256") or "n/a"
        lines.append(
            f"| {evaluation} | {entry['scientific_validity']} |"
            f" {entry['comparison_eligibility']} | {str(identity)[:12]} |"
            f" {entry['run_dir']} |"
        )
    return "\n".join(lines) + "\n"


def write_bundle(
    bundle: dict[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> str:
    markdown = render_bundle_markdown(bundle)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    return markdown
