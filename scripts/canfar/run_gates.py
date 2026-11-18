"""Run harness gates on a CANFAR session without Pixi and record results.

Gate commands are read from ``pixi.toml`` so the Pixi tasks stay the single
definition: ``pixi run -e <env> python ...`` becomes ``<this python> ...``.
Chained (``&&``) tasks are expanded segment by segment. Each gate writes
``<out>/gates/<name>.log`` and a row in ``<out>/gates.jsonl``.

    python run_gates.py --suite performance --mode gpu --out /arc/.../RUN
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older CANFAR images
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]

# Bounded benchmark gates (JSON output, committed baselines or hard thresholds).
PERFORMANCE_TASKS = (
    "bench-coords-astropy-gate",
    "bench-domain-fuzz-gate",
    "bench-native-contracts-gate",
    "bench-wcs-release-gate",
    "bench-wcs-real-headers-gate",
    "bench-healpix-release-gate",
    "bench-healpix-advanced-release-gate",
    "bench-sphere-matrix-release-gate",
    "bench-spin-matrix-release-gate",
    "bench-sphere-real-map-gate",
    "bench-catalog-release-gate",
    "bench-catalog-match-exact-release-gate",
    "bench-docs-reference-gate",
    "bench-map-real-cutout-gate",
    "bench-map-cube-gate",
    "bench-map-ifu-gate",
    "bench-map-ifu-spectral-axis-gate",
    "bench-map-healpix-multiband-gate",
    "bench-native-response-gate",
    "bench-native-likelihood-gate",
    "bench-native-nuisance-gate",
    "bench-map-ground-truth-gate",
    "bench-map-multifield-gate",
    "bench-map-field-fit-gate",
    "bench-map-compat-gate",
    # Torchsky vs every comparator in the map (record, never gate): wcs, healpix, catalogs, moc, sparse, sht, map.
    "bench-comparator-perf-gate",
)

# Gated comparator-parity test files (see tests/conftest.py RELEASE_GATE_FILES).
PARITY_TESTS = (
    "tests/test_cosmology_astropy_parity.py",
    "tests/test_extinction_parity.py",
    "tests/test_spectrophotometry_parity.py",
    "tests/test_redshift_units_parity.py",
    "tests/test_transforms_correctness.py",
    "tests/test_wcs_parity.py",
    "tests/test_wcs_upstream_astropy.py",
    "tests/test_wcs_pyast_parity.py",
    "tests/test_healpix_parity.py",
    "tests/test_pixwin_parity.py",
    "tests/test_healsparse_upstream.py",
    "tests/test_sphere_moc.py",
    "tests/test_sphere_polygons.py",
    "tests/test_sphere_exact_regions.py",
    "tests/test_sphere_compat.py",
    "tests/test_catalog_compat.py",
    "tests/test_synalm_parity.py",
    "tests/test_map_conservative_external_parity.py",
    "tests/test_map_drizzle_external_parity.py",
    "tests/test_map_reproject_fft_parity.py",
    "tests/test_map_dfreproject_parity.py",
    "tests/test_map_astrodrizzle_parity.py",
    "tests/test_map_swarp_optional_parity.py",
    "tests/test_map_external_tools_smoke.py",
    # Comparator packages added to close the claims-vs-gates audit (docs/coverage-matrix.md).
    "tests/test_erfa_observed_place_parity.py",
    "tests/test_gwcs_live_parity.py",
    "tests/test_wcs_optional_extras_parity.py",
    "tests/test_multiorder_interchange_parity.py",
    "tests/test_specutils_parity.py",
    "tests/test_torch_harmonics_parity.py",
    "tests/test_sphere_s2fft.py",
    # External binaries (improc image): required when TORCHSKY_HARNESS_REQUIRE_TOOLS=1.
    "tests/test_external_tools_required.py",
    "tests/test_wcstools_parity.py",
    "tests/test_montage_projection_parity.py",
    "tests/test_stilts_parity.py",
)

CORRECTNESS_MARKERS = "not performance and not benchmark and not slow and not integration"


def _pixi_tasks() -> dict[str, str]:
    payload = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    tasks: dict[str, str] = {}
    for name, value in payload.get("tasks", {}).items():
        command = value if isinstance(value, str) else value.get("cmd") if isinstance(value, dict) else None
        if command:
            tasks[name] = command
    return tasks


# CLI pipelines that need an external catalogue release, memberships and a
# checkpoint; torchsky's own examples/test_examples.py excludes them too.
EXAMPLES_NOT_SMOKE = frozenset({"catalogue_photoz.py"})


def _segments(command: str, tasks: dict[str, str], depth: int = 0) -> list[str]:
    out: list[str] = []
    for part in (p.strip() for p in command.split("&&")):
        match = re.fullmatch(r"pixi run (?!-e)(\S+)", part)
        if match and depth < 4 and match.group(1) in tasks:
            out.extend(_segments(tasks[match.group(1)], tasks, depth + 1))
            continue
        part = re.sub(r"^pixi run (?:-e \S+ )?(?:-- )?", "", part)
        part = re.sub(r"^python\b", shlex.quote(sys.executable), part)
        out.append(part)
    return out


def _run(name: str, suite: str, command: str, out: Path, timeout: int, env: dict[str, str]) -> dict:
    log = out / "gates" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"$ {command}\n")
            stream.flush()
            completed = subprocess.run(
                command, shell=True, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
    row = {
        "gate": name,
        "suite": suite,
        "command": command,
        "returncode": returncode,
        # Exit 77 (automake convention) marks an opt-in example skipped for missing data.
        "status": "pass" if returncode == 0 else ("skip" if (suite == "examples" and returncode == 77) else "fail"),
        "seconds": round(time.time() - started, 2),
        "log": str(log),
    }
    with (out / "gates.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(f"[{row['status']}] {suite}/{name} {row['seconds']}s", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", choices=["correctness", "parity", "performance", "examples"], required=True)
    parser.add_argument("--examples-dir", type=Path, default=None, help="torchsky examples/ directory (examples suite)")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600, help="per-gate timeout in seconds")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PYTHONNOUSERSITE="1", TORCHSKY_HARNESS_REQUIRE_GATES="1")
    env.pop("PYTHONPATH", None)
    env["TORCHSKY_HARNESS_REQUIRE_MAP_COMPAT"] = env.get("TORCHSKY_HARNESS_REQUIRE_MAP_COMPAT", "0")
    env["TORCHSKY_HARNESS_REQUIRE_TOOLS"] = env.get("TORCHSKY_HARNESS_REQUIRE_TOOLS", "0")
    # Without an explicit device torchsky follows its inputs and, for file-path
    # observations, defaults to CUDA when present. CPU-semantics gates build their
    # truth/oracle tensors on the host, so pin them to CPU in gpu mode; the CUDA
    # variants and cuda-smoke select CUDA explicitly (an explicit device wins).
    cpu_env = dict(env, TORCHSKY_DEVICE="cpu")
    py = shlex.quote(sys.executable)
    results: list[dict] = []

    if args.suite == "correctness":
        workers = " -n 4" if _has_xdist() else ""
        cmd = f'{py} -m pytest tests -q{workers} -p no:cacheprovider -m "{CORRECTNESS_MARKERS}" -rfs'
        correctness_env = dict(env, TORCHSKY_HARNESS_REQUIRE_GATES="0")
        cpu_correctness_env = dict(correctness_env, TORCHSKY_DEVICE="cpu")
        results.append(_run("harness-nonslow", "correctness", cmd, args.out, args.timeout * 2, cpu_correctness_env))
        if args.mode == "gpu":
            cuda_cmd = f"{py} -m pytest tests/test_cuda_smoke.py -q -p no:cacheprovider -rfs"
            results.append(_run("cuda-smoke", "correctness", cuda_cmd, args.out, args.timeout, correctness_env))
            cuda_correctness_env = dict(correctness_env, TORCHSKY_DEVICE="cuda")
            results.append(_run("harness-nonslow--cuda", "correctness", cmd, args.out, args.timeout * 2, cuda_correctness_env))
    elif args.suite == "parity":
        for path in PARITY_TESTS:
            if not (ROOT / path).is_file():
                results.append({"gate": path, "suite": "parity", "status": "fail", "returncode": 127, "seconds": 0.0, "log": "", "command": "missing test file"})
                continue
            name = Path(path).stem
            cmd = f"{py} -m pytest {path} -q -p no:cacheprovider -rfs"
            results.append(_run(name, "parity", cmd, args.out, args.timeout, cpu_env))
            if args.mode == "gpu":
                # Same comparator checks with CUDA as the default device, so a GPU
                # campaign really exercises the CUDA paths (tests must be device-aware).
                cuda_env = dict(env, TORCHSKY_DEVICE="cuda")
                results.append(_run(f"{name}--cuda", "parity", cmd, args.out, args.timeout, cuda_env))
    elif args.suite == "examples":
        if args.examples_dir is None or not args.examples_dir.is_dir():
            print("--examples-dir is required for the examples suite", file=sys.stderr)
            return 2
        workdir = args.out / "examples-work"
        workdir.mkdir(parents=True, exist_ok=True)
        example_env = dict(env, MPLBACKEND="Agg", TORCHSKY_HARNESS_REQUIRE_GATES="0")
        for script in sorted(args.examples_dir.glob("*.py")):
            if script.name.startswith("test_") or script.name in EXAMPLES_NOT_SMOKE:
                continue
            # run from a scratch directory so examples cannot write into the source tree
            cmd = f"cd {shlex.quote(str(workdir))} && {py} -s {shlex.quote(str(script))}"
            results.append(_run(script.stem, "examples", cmd, args.out, args.timeout, example_env))
    else:
        tasks = _pixi_tasks()
        for task in PERFORMANCE_TASKS:
            if task not in tasks:
                results.append({"gate": task, "suite": "performance", "status": "fail", "returncode": 127, "seconds": 0.0, "log": "", "command": "task missing from pixi.toml"})
                continue
            segments = _segments(tasks[task], tasks)
            devices = [("cpu", segments)]
            if args.mode == "gpu" and any("--device cpu" in s for s in segments):
                devices.append(("cuda", [s.replace("--device cpu", "--device cuda").replace(".json", "_cuda.json") for s in segments]))
            for label, segs in devices:
                name = task if label == "cpu" else f"{task}--cuda"
                results.append(
                    _run(name, "performance", " && ".join(segs), args.out, args.timeout, cpu_env if label == "cpu" else env)
                )

    failed = [r for r in results if r["status"] == "fail"]
    print(f"{args.suite}: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


def _has_xdist() -> bool:
    try:
        import xdist  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
