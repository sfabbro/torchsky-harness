import json
from pathlib import Path

import pytest

from torchsky_harness import runner
from torchsky_harness.evidence import CLAIMS, verdict
from torchsky_harness.registry import evaluations

IFU_ROWS = [
    {
        "case": "ifu_matched_sparse_spectrum",
        "library": "torchsky",
        "method": "coadd_cube",
        "status": "OK",
        "reconstruction_rmse": 0.0,
    }
]


def _write_arguments(tmp_path: Path, *, baseline: bool = True) -> list[str]:
    arguments = ["--json-out", str(tmp_path / "report.json")]
    if baseline:
        arguments += [
            "--baseline-json",
            str(tmp_path / "baseline.json"),
            "--require-complete-baseline",
        ]
    return arguments


def _write_report(tmp_path: Path, rows: list[dict]) -> None:
    (tmp_path / "report.json").write_text(json.dumps(rows), encoding="utf-8")


def test_declared_claims_cover_only_release_gates() -> None:
    assert set(CLAIMS) == {
        "bench_catalog_compat",
        "bench_catalog_match_exact",
        "bench_coords_astropy",
        "bench_domain_fuzz",
        "bench_healpix",
        "bench_healpix_advanced",
        "bench_map_cube",
        "bench_map_ifu",
        "bench_map_ifu_spectral_axis",
        "bench_map_healpix_multiband",
        "bench_map_drizzle_required",
        "bench_map_real_cutout",
        "bench_native_contracts",
        "bench_native_likelihood",
        "bench_native_nuisance",
        "bench_native_response",
        "bench_sphere_matrix",
        "bench_sphere_real_map",
        "bench_spin_matrix",
        "bench_wcs",
        "bench_wcs_real_headers",
    }


def test_bounded_drizzle_evaluation_has_a_distinct_registry_entry() -> None:
    assert evaluations()["bench_map_drizzle_required"] == (
        "torchsky_harness.benchmarks.bench_map_drizzle_required"
    )


def test_undeclared_evaluation_stays_unassessed(tmp_path: Path) -> None:
    record = verdict(
        "bench_sphere_sparse",
        returncode=0,
        command=["python", "-m", "bench_sphere_sparse"],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_assessed"
    assert record["comparison_eligibility"] == "not_assessed"
    assert record["verdict_reasons"] == ["no declared claim for this evaluation"]


def test_declared_claim_is_established_only_with_required_rows(tmp_path: Path) -> None:
    _write_report(tmp_path, IFU_ROWS)
    record = verdict(
        "bench_map_ifu",
        returncode=0,
        command=["python", "-m", "bench_map_ifu", *_write_arguments(tmp_path)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "established"
    assert record["comparison_eligibility"] == "baseline_compared"
    assert record["verdict_reasons"] == []
    assert record["claim"]["scope"] == "synthetic_truth"
    assert record["evidence_report"] == str((tmp_path / "report.json").resolve())


def test_missing_required_row_blocks_validity(tmp_path: Path) -> None:
    _write_report(tmp_path, [{"case": "something_else", "status": "OK"}])
    record = verdict(
        "bench_map_ifu",
        returncode=0,
        command=["python", "-m", "bench_map_ifu", *_write_arguments(tmp_path)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert "missing required row ifu_matched_sparse_spectrum" in record["verdict_reasons"]


def test_failed_required_row_blocks_validity(tmp_path: Path) -> None:
    _write_report(tmp_path, [{**IFU_ROWS[0], "status": "FAILED"}])
    record = verdict(
        "bench_map_ifu",
        returncode=0,
        command=["python", "-m", "bench_map_ifu", *_write_arguments(tmp_path)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert any("reported status FAILED" in reason for reason in record["verdict_reasons"])


def test_nonzero_exit_blocks_validity(tmp_path: Path) -> None:
    _write_report(tmp_path, IFU_ROWS)
    record = verdict(
        "bench_map_ifu",
        returncode=1,
        command=["python", "-m", "bench_map_ifu", *_write_arguments(tmp_path)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert record["verdict_reasons"][0] == "evaluation exited with return code 1"


def test_unanchored_baseline_is_reported_not_enforced(tmp_path: Path) -> None:
    _write_report(tmp_path, IFU_ROWS)
    arguments = ["--json-out", str(tmp_path / "report.json"), "--baseline-json", str(tmp_path / "baseline.json")]
    record = verdict(
        "bench_map_ifu",
        returncode=0,
        command=["python", "-m", "bench_map_ifu", *arguments],
        working_directory=tmp_path,
    )
    assert record["comparison_eligibility"] == "suite_owned"
    assert record["comparison_notes"] == [
        "baseline supplied without --require-complete-baseline, so baseline"
        " coverage is not enforced"
    ]


def test_every_row_for_a_required_case_must_pass(tmp_path: Path) -> None:
    rows = [
        {"case": "synthetic_clustered", "library": "torchsky", "status": "FAILED"},
        {"case": "synthetic_clustered", "library": "astropy", "status": "OK"},
        {"case": "real_astropy_votable_catalogs", "library": "torchsky", "status": "OK"},
        {"case": "real_astropy_votable_fits_roundtrip", "library": "torchsky", "status": "OK"},
    ]
    _write_report(tmp_path, rows)
    record = verdict(
        "bench_catalog_compat",
        returncode=0,
        command=["python", "-m", "bench_catalog_compat", *_write_arguments(tmp_path)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert any(
        "synthetic_clustered reported status FAILED" in reason
        for reason in record["verdict_reasons"]
    )


def test_operation_rows_satisfy_a_comparator_claim(tmp_path: Path) -> None:
    rows = [
        {"operation": operation, "mismatches": 0}
        for operation in (
            "ang2pix_ring",
            "ang2pix_nested",
            "pix2ang_ring",
            "pix2ang_nested",
            "ring2nest",
            "nest2ring",
        )
    ]
    _write_report(tmp_path, rows)
    record = verdict(
        "bench_healpix",
        returncode=0,
        command=["python", "-m", "bench_healpix", *_write_arguments(tmp_path)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "established"
    assert record["claim"]["scope"] == "reference_comparison"
    assert "healpy" in record["claim"]["comparators"]


def test_bounded_drizzle_claim_uses_distinct_evaluation_name(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        [
            {
                "case": "upstream_drizzle_crpix_translation",
                "library": "drizzle",
                "variant": variant,
                "status": "OK",
            }
            for variant in ("point", "square")
        ],
    )
    record = verdict(
        "bench_map_drizzle_required",
        returncode=0,
        command=[
            "python",
            "-m",
            "bench_map_drizzle_required",
            "--acceptance-json",
            "benchmarks/baselines/map_drizzle_acceptance.json",
            "--require-comparators",
            "drizzle",
            *_write_arguments(tmp_path, baseline=False),
        ],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "established"
    assert record["claim"]["required_rows"] == [
        "upstream_drizzle_crpix_translation"
    ]
    assert record["claim"]["comparators"] == ["drizzle"]
    assert record["claim"]["required_row_selectors"] == [
        {
            "case": "upstream_drizzle_crpix_translation",
            "library": "drizzle",
            "variant": "point",
        },
        {
            "case": "upstream_drizzle_crpix_translation",
            "library": "drizzle",
            "variant": "square",
        },
    ]
    assert any(
        "not real-data replacement parity" in limitation
        for limitation in record["claim"]["limitations"]
    )


def test_bounded_drizzle_claim_requires_the_acceptance_contract(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        [
            {
                "case": "upstream_drizzle_crpix_translation",
                "library": "drizzle",
                "variant": variant,
                "status": "OK",
            }
            for variant in ("point", "square")
        ],
    )
    record = verdict(
        "bench_map_drizzle_required",
        returncode=0,
        command=["python", "-m", "bench_map_drizzle_required", *_write_arguments(tmp_path, baseline=False)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert "evaluation requires --acceptance-json" in record["verdict_reasons"][0]


@pytest.mark.parametrize("status", [pytest.param(None, id="null"), pytest.param("FAILED", id="failing")])
def test_bounded_drizzle_claim_requires_explicit_ok_status(
    tmp_path: Path, status: str | None
) -> None:
    rows = [
        {
            "case": "upstream_drizzle_crpix_translation",
            "library": "drizzle",
            "variant": variant,
            "status": "OK",
        }
        for variant in ("point", "square")
    ]
    rows[0]["status"] = status
    _write_report(tmp_path, rows)
    record = verdict(
        "bench_map_drizzle_required",
        returncode=0,
        command=[
            "--acceptance-json",
            "benchmarks/baselines/map_drizzle_acceptance.json",
            "--require-comparators",
            "drizzle",
            *_write_arguments(tmp_path, baseline=False),
        ],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert any("reported status" in reason for reason in record["verdict_reasons"])


def test_bounded_drizzle_claim_rejects_missing_status(tmp_path: Path) -> None:
    rows = [
        {
            "case": "upstream_drizzle_crpix_translation",
            "library": "drizzle",
            "variant": variant,
            "status": "OK",
        }
        for variant in ("point", "square")
    ]
    rows[0].pop("status")
    _write_report(tmp_path, rows)
    record = verdict(
        "bench_map_drizzle_required",
        returncode=0,
        command=[
            "--acceptance-json",
            "benchmarks/baselines/map_drizzle_acceptance.json",
            "--require-comparators",
            "drizzle",
            *_write_arguments(tmp_path, baseline=False),
        ],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"


def test_bounded_drizzle_claim_rejects_missing_required_variant(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        [
            {
                "case": "upstream_drizzle_crpix_translation",
                "library": "drizzle",
                "variant": "point",
                "status": "OK",
            }
        ],
    )
    record = verdict(
        "bench_map_drizzle_required",
        returncode=0,
        command=[
            "--acceptance-json",
            "benchmarks/baselines/map_drizzle_acceptance.json",
            "--require-comparators",
            "drizzle",
            *_write_arguments(tmp_path, baseline=False),
        ],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert any("variant=square" in reason for reason in record["verdict_reasons"])


def test_summary_report_claim_requires_an_empty_failure_list(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"failed_thresholds": []}), encoding="utf-8")
    record = verdict(
        "bench_sphere_matrix",
        returncode=0,
        command=["python", "-m", "bench_sphere_matrix", "--json-out", str(report)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "established"
    assert record["claim"]["summary_failures_key"] == "failed_thresholds"
    assert record["evidence_summary"] == str(report.resolve())


def test_summary_report_failures_block_validity(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"failed_thresholds": [{"operation": "ring2nest"}]}),
        encoding="utf-8",
    )
    record = verdict(
        "bench_sphere_matrix",
        returncode=0,
        command=["python", "-m", "bench_sphere_matrix", "--json-out", str(report)],
        working_directory=tmp_path,
    )
    assert record["scientific_validity"] == "not_established"
    assert any("failed_thresholds" in reason for reason in record["verdict_reasons"])


def test_runner_records_verdict_for_declared_claim(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "identify_subject", lambda: {"version": "test"})
    monkeypatch.setattr(runner, "git_identity", lambda _: {"revision": "test"})

    def fake_run(command, **_kwargs):
        report_path = Path(command[command.index("--json-out") + 1])
        report_path.write_text(json.dumps(IFU_ROWS), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    run_dir = tmp_path / "run"
    assert runner.run("bench_map_ifu", _write_arguments(tmp_path), run_dir) == 0
    record = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    assert record["execution"] == "completed"
    assert record["scientific_validity"] == "established"
    assert record["comparison_eligibility"] == "baseline_compared"
    assert record["claim"]["scope"] == "synthetic_truth"
    assert len(record["logs"]) == 2


def test_runner_marks_missing_report_as_unestablished(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "identify_subject", lambda: {})
    monkeypatch.setattr(runner, "git_identity", lambda _: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: type("Completed", (), {"returncode": 0})(),
    )
    run_dir = tmp_path / "run"
    assert runner.run("bench_map_ifu", _write_arguments(tmp_path), run_dir) == 0
    record = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    assert record["scientific_validity"] == "not_established"
    assert "evidence report unreadable" in record["verdict_reasons"][0]


@pytest.mark.parametrize("name", sorted(CLAIMS))
def test_every_declared_claim_lists_rows_and_limitations(name: str) -> None:
    claim = CLAIMS[name]
    assert claim.required_rows or claim.summary_failures_key
    assert claim.limitations
    assert claim.scope in {
        "synthetic_truth",
        "reference_comparison",
        "real_data",
        "contract",
    }
