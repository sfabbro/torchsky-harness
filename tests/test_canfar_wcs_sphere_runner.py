from pathlib import Path

import numpy as np
import torch
from torchfits import table as fits_table
from torchsky_harness.benchmarks.bench_catalog_compat import (
    _load_torchsky_catalog_from_fits,
    _make_real_fits_tables,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canfar_wcs_sphere_runner_contract() -> None:
    runner = (ROOT / "scripts" / "run_canfar_wcs_sphere.sh").read_text(encoding="utf-8")
    assert "TORCHSKY_HARNESS_COMMIT" in runner
    assert "torchsky_install_subject" in runner
    assert "acceptance_wcs_full" in runner
    assert "--no-legacy-wcs-bridge" in runner
    assert "CONTAINER_IMAGE must name an AstroAI image" in runner
    assert "unknown mode" in runner
    assert runner.index("python -m pip install astropy") < runner.index("-k cuda")
    assert "--wcs-device cuda" in runner
    assert "vos:sfabbro/torchsky-runs" in runner
    assert "bench_sphere_spectral.py" in runner
    assert "vcp" in runner
    assert "gist create" in runner
    assert "trap on_exit EXIT" in runner
    assert "vos_upload heartbeat" in runner
    assert 'vmkdir -p "${vos_dest}" 2>/dev/null || true' not in runner
    assert "cuda_rc" in runner
    assert "cpu_rc" in runner
    assert "cuda job failures" in runner
    assert "cpu job failures" in runner
    assert "optimization-cpu)" in runner
    assert "optimization-cuda)" in runner
    assert "optimization_solver_pcg_cuda.json" in runner
    assert "optimization_triplane_cpu.json" in runner
    assert "real-data-gates)" in runner
    assert "map_real_cutout.json" in runner
    assert "sphere_real_map.json" in runner
    assert "wcs_real_headers.json" in runner
    assert runner.index("-k cuda") < runner.index("acceptance_wcs_cuda")
    assert runner.index("--no-legacy-wcs-bridge") < runner.index("acceptance_wcs_cuda")
    assert "vos_upload after_wcs" in runner
    for mode in ("spectral", "sphere", "sparse", "spin", "geometry", "cuhpx"):
        assert f"\n  {mode})" in runner or f"\n  {mode} |" in runner
    assert "sphere_spectral_nside128.json" in runner
    assert "bench_sphere_sparse.py" in runner
    assert "bench_cuhpx_sht_spike.py" in runner
    assert "--no-throughput-gate" in runner
    assert runner.count("--no-throughput-gate") >= 4
    assert "--n-queries 200000" in runner
    assert "--n-queries 500000" in runner
    for mode in (
        "wcs-paper2",
        "wcs-tpv",
        "wcs-spectral",
        "wcs-tabular",
        "sht-profile",
        "moc",
        "catalog",
        "catalog-scale",
        "catalog-scale-1e6",
        "catalog-scale-1e7",
        "observed",
        "geom-cuda",
        "world-cuda",
    ):
        assert f"\n  {mode})" in runner
    assert "--moc-only" in runner
    assert "bench_catalog_match_exact.py" in runner
    assert "bench_catalog_match_perf.py" in runner
    assert "catalog_match_perf_1e6.json" in runner
    assert "catalog-match-1e7 | catalog-match-1e8)" in runner
    assert "\n  catalog-match-real)" in runner
    assert "\n  catalog-match-cuda)" in runner
    assert "catalog_match_perf_cuda.json" in runner
    assert "\n  catalog-match-gpu-peers)" in runner
    assert "install_gpu_nn_peers.sh" in runner
    assert "catalog_match_gpu_peers.json" in runner
    assert (ROOT / "scripts" / "install_gpu_nn_peers.sh").exists()
    assert "bench_catalog_match_real.py" in runner
    assert "bench_catalog_crossmatch.py" not in runner
    assert "test_crossmatch_engine.py" in runner
    assert "bench_catalog_compat.py" in runner
    assert "catalog_compat.json" in runner
    assert "run_catalog_scale_contract.py" in runner
    assert "catalog_scale_200k_dense.json" in runner
    assert "catalog_scale_1e7_sparse_best.json" in runner
    assert "vos_upload after_catalog_compat" in runner
    assert "vos_upload after_catalog_match_exact" in runner
    assert "vos_run" in runner
    assert "timeout --foreground" in runner
    assert 'vcp "${artifact_root}/."' not in runner
    assert "wcs_paper2_fused0.json" in runner
    assert "test_wcs_observed_frames.py" in runner
    assert "test_catalogs.py" in runner
    assert "bench_wcs_spectral_temporal.py" in runner
    assert "bench_wcs_tabular.py" in runner
    assert "bench_wcs_small_n_paths.py" in runner
    assert "--device cuda" in runner
    assert "wcs_paper2_cuda.json" in runner
    assert "observed_place_cuda.json" in runner
    assert "catalog_cuda.json" in runner


def test_canfar_torchsky_tests_cuda_mode() -> None:
    runner = (ROOT / "scripts" / "run_canfar_wcs_sphere.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "launch_canfar_wcs_sphere.sh").read_text(
        encoding="utf-8"
    )
    assert "\n  torchsky-tests-cuda)" in runner
    body = runner[runner.index("\n  torchsky-tests-cuda)") :]
    body = body[: body.index("\n    ;;")]
    # Subject installed into the CUDA env the other GPU modes use, CUDA asserted
    # before any test runs, torchsky's test-feature dependencies installed.
    assert "install_env cuda-bench" in body
    assert body.index("torch.cuda.is_available()") < body.index("pytest tests")
    assert '"pytest-xdist>=3.0"' in body and '"pyarrow>=25"' in body
    # Main suite as torchsky's `pixi run test`, map tests run separately.
    assert "pytest tests --ignore-glob='tests/test_map_*.py'" in body
    assert "KMP_DUPLICATE_LIB_OK=TRUE" in body
    assert "pytest -n 4" in body and "tests/test_map_*.py" in body
    # Per-test outcomes and skip reasons are artifacts; failures fail the job
    # only after both suites and the summary ran.
    assert "torchsky_tests_main_cuda.xml" in body
    assert "torchsky_tests_map_cuda.xml" in body
    assert "-rfEs" in body
    assert "summarize_junit.py" in body
    assert "torchsky_tests_cuda_summary.json" in body
    assert "device_probe.json" in body
    assert body.index("summarize_junit.py") < body.index('exit "${tests_rc}"')
    assert (ROOT / "scripts" / "summarize_junit.py").exists()
    assert 'launch_one torchsky-tests-cuda 1 "torchsky-tests-cuda-${STAMP}"' in launcher
    assert "| torchsky-tests-cuda |" in launcher


def test_summarize_junit_counts_and_device_skips(tmp_path) -> None:
    import json
    import subprocess
    import sys

    xml = tmp_path / "r.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="6">
<testcase classname="tests.test_a" name="test_ok" time="0.1"/>
<testcase classname="tests.test_a" name="test_bad[cuda]" time="0.2">
  <failure message="AssertionError: x">def f():
E   AssertionError: tensors differ</failure></testcase>
<testcase classname="tests.test_b" name="test_dev[mps]" time="0.0">
  <skipped type="pytest.skip" message="mps unavailable">tests/test_b.py:9: mps unavailable</skipped></testcase>
<testcase classname="tests.test_b" name="test_f64[mps]" time="0.0">
  <skipped type="pytest.skip" message="needs float64; MPS has none">x</skipped></testcase>
<testcase classname="tests.test_c" name="test_xf" time="0.0">
  <skipped type="pytest.xfail" message="known">x</skipped></testcase>
<testcase classname="tests.test_c" name="test_err" time="0.0">
  <error message="fixture failed">E   RuntimeError: boom</error></testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )
    out = tmp_path / "s.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_junit.py"),
            f"main={xml}",
            f"absent={tmp_path / 'nope.xml'}",
            "--meta",
            "device=cuda",
            "--json-out",
            str(out),
        ],
        check=True,
    )
    summary = json.loads(out.read_text())
    assert summary["meta"] == {"device": "cuda"}
    assert summary["suites"]["absent"]["missing"] is True
    main = summary["suites"]["main"]
    assert main["counts"] == {
        "total": 6,
        "passed": 1,
        "failed": 1,
        "error": 1,
        "skipped": 2,
        "xfailed": 1,
    }
    assert main["device_absent_skips"] == 1
    reasons = {g["reason"]: g["device_absent"] for g in main["skip_groups"]}
    assert reasons == {"mps unavailable": True, "needs float64; MPS has none": False}
    bad = {f["name"]: f for f in main["failures"]}
    assert bad["test_bad[cuda]"]["error"] == "AssertionError: tensors differ"
    assert bad["test_bad[cuda]"]["file"] == "tests/test_a.py"
    assert bad["test_err"]["outcome"] == "error"


def test_canfar_wcs_sphere_launcher_payload_has_no_shell_vars() -> None:
    launcher = (ROOT / "scripts" / "launch_canfar_wcs_sphere.sh").read_text(
        encoding="utf-8"
    )
    assert "canfar" in launcher
    assert "tr ' ' '\\t'" in launcher
    assert "vos:sfabbro/torchsky-runs" in launcher
    assert "refusing canfar payload" in launcher
    assert "gh repo clone ${harness_repo}" in launcher
    assert "TORCHSKY_HARNESS_REPO:-astroai/torchsky-harness" in launcher
    assert launcher.count("gh auth setup-git") == 2
    assert "git push before launching CANFAR" in launcher
    assert "optimization-cpu" in launcher
    assert "optimization-cuda" in launcher
    assert "real-data-gates" in launcher
    assert "images.canfar.net/astroai/notebook:latest" in launcher
    assert "RUNNER_VOS_URI" not in launcher
    assert "suite" in launcher
    assert "cuhpx" in launcher
    assert "sht" in launcher
    assert "world" in launcher
    assert "world-cuda" in launcher
    assert "catalog-scale" in launcher
    assert "catalog-scale-1e7" in launcher
    assert 'launch_one catalog-match-gpu-peers 1 "catalog-match-gpu-peers-${STAMP}"' in launcher
    assert "wcs-spectral" in launcher
    assert "wcs-tabular" in launcher
    assert launcher.count('--env "TORCHSKY_SPIN_THREADS=${CPU}"') >= 2
    assert launcher.count("PATH=/opt/conda/bin:") == 2
    assert '--env "TORCHSKY_MAP_COMPAT_ENV=${map_compat_env}"' in launcher
    assert "map-compat-cuda" in launcher
    fetch = (ROOT / "scripts" / "fetch_canfar_wcs_sphere_run.sh").read_text(
        encoding="utf-8"
    )
    assert "vos:sfabbro/torchsky-runs" in fetch
    assert "vcp" in fetch


def test_real_data_fleet_is_safe_by_default() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "launch_canfar_real_data_fleet.sh"
    ).read_text()
    assert "SUBMIT=0" in script
    assert "TORCHSKY_CANFAR_DRY_RUN=$((1 - SUBMIT))" in script
    assert "git -C \"${repository}\" status --porcelain" in script
    assert "real-data-gates catalog-scale-real" in script
    assert "MODES+=(observed optimization gpu-validate depth)" in script
    assert "status --porcelain" in script
    assert "git ls-remote" in script
    assert "TORCHSKY_SUBJECT_REPO=$(" in script
    assert "export CONTAINER_IMAGE" not in script
    assert 'dirname "${BASH_SOURCE[0]}"' in script
    assert "HARNESS_REPO=astroai/torchsky-harness" in script


def test_catalog_compat_fits_tables_write_numeric_columns(tmp_path) -> None:
    left, right = _make_real_fits_tables()
    assert set(left.colnames) == {"RA", "DEC"}
    assert set(right.colnames) == {"RA", "DEC"}
    path = tmp_path / "left_catalog.fits"
    fits_table.write(
        str(path),
        {"RA": np.asarray(left["RA"]), "DEC": np.asarray(left["DEC"])},
        overwrite=True,
    )
    assert path.stat().st_size > 0
    ra, dec = _load_torchsky_catalog_from_fits(path)
    assert ra.dtype == torch.float64
    assert dec.dtype == torch.float64
    assert ra.numel() == len(left)
