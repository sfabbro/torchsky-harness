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


def test_canfar_wcs_sphere_launcher_payload_has_no_shell_vars() -> None:
    launcher = (ROOT / "scripts" / "launch_canfar_wcs_sphere.sh").read_text(
        encoding="utf-8"
    )
    assert "canfar" in launcher
    assert "tr ' ' '\\t'" in launcher
    assert "vos:sfabbro/torchsky-runs" in launcher
    assert "refusing canfar payload" in launcher
    assert "gh repo clone astroai/torchsky-harness" in launcher
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
