#!/usr/bin/env bash
# In-container CANFAR runner for torchsky WCS + sphere tests, healpy parity, and benches.
# Writes ${TORCHSKY_ARC_ROOT:-/arc/projects/ots}/torchsky-runs/<run_id>/ and copies the same tree to
# vos:sfabbro/torchsky-runs/<run_id>/.
set -euo pipefail

mode=${1:?usage: run_canfar_wcs_sphere.sh MODE RUN_ID}
run_id=${2:?usage: run_canfar_wcs_sphere.sh MODE RUN_ID}
commit=${TORCHSKY_HARNESS_COMMIT:?set TORCHSKY_HARNESS_COMMIT to an exact pushed SHA}
container_image=${CONTAINER_IMAGE:-images.canfar.net/astroai/notebook:latest}
artifact_root=${TORCHSKY_ARC_ROOT:-/arc/projects/ots}/torchsky-runs/${run_id}
source_root=${TORCHSKY_SOURCE_ROOT:-/scratch/${run_id}-src}
vos_dest=${TORCHSKY_VOS_DEST:-vos:sfabbro/torchsky-runs/${run_id}}

case "${container_image}" in
  images.canfar.net/astroai/* | astroai/*) ;;
  *)
    echo "CONTAINER_IMAGE must name an AstroAI image, got: ${container_image}" >&2
    exit 2
    ;;
esac

case "${mode}" in
  cpu | cuda | optimization-cpu | optimization-cuda | real-data-gates | spectral | sphere | sparse | healsparse | spin | geometry | cuhpx | \
  wcs-paper2 | wcs-tpv | wcs-spectral | wcs-tabular | sht-profile | moc | catalog | \
  catalog-scale | catalog-scale-1e6 | catalog-scale-1e7 | catalog-scale-real | \
  catalog-match-1e7 | catalog-match-1e8 | catalog-match-real | \
  observed | geom-cuda | world-cuda | gpu-validate) ;;
  *)
    echo "unknown mode: ${mode} (expected cpu, cuda, optimization-cpu, optimization-cuda, real-data-gates, spectral, sphere, sparse, healsparse, spin, geometry, cuhpx, wcs-paper2, wcs-tpv, wcs-spectral, wcs-tabular, sht-profile, moc, catalog, catalog-scale, catalog-scale-1e6, catalog-scale-1e7, catalog-scale-real, catalog-match-1e7, catalog-match-1e8, catalog-match-real, observed, geom-cuda, or world-cuda)" >&2
    exit 2
    ;;
esac

# Threads for catalog benchmarks: every engine gets the same count.
catalog_threads="${CATALOG_THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}"

mkdir -p "${artifact_root}"
echo "started_utc=$(date -u +%FT%TZ)" > "${artifact_root}/HEARTBEAT"
exec > >(tee "${artifact_root}/session.log") 2>&1

echo "run_id=${run_id}"
echo "mode=${mode}"
echo "requested_commit=${commit}"
echo "image=${container_image}"
echo "vos_dest=${vos_dest}"
echo "started_utc=$(date -u +%FT%TZ)"
nvidia-smi || true

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR=/scratch/pip-cache
export TMPDIR=/scratch/tmp
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

# vos/CADC CLIs ship with AstroAI images but are often off PATH.
for d in /opt/astroai/venv/cadc/bin /opt/astroai/bin /opt/conda/bin /usr/local/bin; do
  if [[ -x "${d}/vcp" || -x "${d}/gh" ]]; then
    export PATH="${d}:${PATH}"
  fi
done

gist_upload() {
  local label=$1
  if ! command -v gh >/dev/null || ! gh auth status >/dev/null 2>&1; then
    echo "gh not authenticated; skip gist fallback" >&2
    return 1
  fi
  local files=()
  local f
  for f in "${artifact_root}/HEARTBEAT" "${artifact_root}/session.log" "${artifact_root}"/*.json; do
    if [[ -f "${f}" && -s "${f}" ]]; then
      files+=("${f}")
    fi
  done
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "gist_upload: no non-empty artifact files" >&2
    return 1
  fi
  local url
  url="$(gh gist create --desc "torchsky ${run_id} ${label}" "${files[@]}")"
  echo "${url}" | tee "${artifact_root}/GIST_URL"
  echo "gist_url=${url}"
}

vos_run() {
  local sec=$1
  shift
  if command -v timeout >/dev/null; then
    timeout --foreground --signal=TERM --kill-after=15 "${sec}" "$@"
  else
    "$@"
  fi
}

vos_upload() {
  local label=$1
  echo "vos_upload label=${label} utc=$(date -u +%FT%TZ)"
  if ! command -v vcp >/dev/null; then
    echo "vcp not on PATH; cannot copy ${artifact_root} to ${vos_dest}" >&2
    return 1
  fi
  # ponytail: vault directory PUT stores 0-byte nodes; vls of those nodes
  # hung catalog jobs for 15h after JSON was already on /arc. Tarball only,
  # 3 min ceiling, then the bench continues.
  tar -C "$(dirname "${artifact_root}")" -czf "/scratch/${run_id}.tar.gz" "$(basename "${artifact_root}")" || return 1
  vos_run 180 vcp "/scratch/${run_id}.tar.gz" "vos:sfabbro/torchsky-runs/${run_id}.tar.gz" || return 1
  echo "vos_tarball=vos:sfabbro/torchsky-runs/${run_id}.tar.gz"
}

on_exit() {
  local rc=$?
  set +e
  echo "exit_utc=$(date -u +%FT%TZ) rc=${rc}" | tee -a "${artifact_root}/HEARTBEAT"
  vos_upload "exit_rc${rc}" || echo "VOS upload failed rc=${rc}" >&2
  gist_upload "exit_rc${rc}" || echo "gist fallback failed rc=${rc}" >&2
  exit "${rc}"
}
trap on_exit EXIT

vos_upload heartbeat || echo "heartbeat VOS upload failed; continuing job" >&2

clone_source() {
  if [[ -d "${source_root}/.git" ]] \
    && [[ "$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || true)" == "${commit}" ]]; then
    echo "reusing TORCHSKY_SOURCE_ROOT=${source_root}"
    return
  fi
  test ! -e "${source_root}"
  local repo="${TORCHSKY_HARNESS_REPO:-astroai/torchsky-harness}"
  local clone_url="https://github.com/${repo}.git"
  if [[ -n "${GH_TOKEN:-}" ]]; then
    clone_url="https://x-access-token:${GH_TOKEN}@github.com/${repo}.git"
  fi
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo clone "${repo}" "${source_root}" -- --filter=blob:none
  else
    git clone --filter=blob:none "${clone_url}" "${source_root}"
  fi
  git -C "${source_root}" checkout --detach "${commit}"
}

clone_source
test "$(git -C "${source_root}" rev-parse HEAD)" = "${commit}"
git -C "${source_root}" status --short

export PIXI_HOME=/scratch/pixi
export PIXI_CACHE_DIR=/scratch/pixi-cache
export PATH="${PIXI_HOME}/bin:${PATH}"
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
pixi --version

cd "${source_root}"
source "${source_root}/scripts/subject_install.sh"
echo "git_head=$(git rev-parse HEAD)"
pixi info || true

install_env() {
  local env_name=$1
  torchsky_install_subject "${env_name}" "${source_root}"
  pixi run --manifest-path "${source_root}/pixi.toml" -e "${env_name}" -- \
    python -m torchsky_harness.cli subject > "${artifact_root}/SUBJECT_${env_name}.json"
}

record_versions() {
  local env_name=$1
  pixi run --manifest-path "${source_root}/pixi.toml" -e "${env_name}" -- \
    python "${source_root}/scripts/print_benchmark_comparator_versions.py" \
    --json-out "${artifact_root}/VERSIONS_${env_name}.json"
}

case "${mode}" in
  real-data-gates)
    install_env bench-all
    install_env sphere-bench
    record_versions bench-all
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python -m torchsky_harness.cli run bench_map_real_cutout -- \
      --json-out "${artifact_root}/map_real_cutout.json" \
      --max-relative-error 1e-5 --max-p99-relative-error 1e-6 || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python -m torchsky_harness.cli run bench_sphere_real_map -- \
      --lmax 64 --fwhm-deg 1.0 \
      --json-out "${artifact_root}/sphere_real_map.json" \
      --max-relative-error 1e-4 --max-ud-grade-absolute-error 1e-9 || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python -m torchsky_harness.cli run bench_wcs_real_headers -- \
      --grid 17 --json-out "${artifact_root}/wcs_real_headers.json" \
      --max-angular-error-arcsec 1e-6 --max-inverse-pixel-error 1e-5 || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python -m torchsky_harness.scripts.run_local_sphere_bench \
      "${source_root}/benchmarks/bench_map_compat.py" \
      --size 32 --n-obs 2 --runs 1 \
      --cases real_astropy_sip_fixture,real_astropy_sip_shifted_wcs,real_astropy_sip_mef,real_astropy_sip_mef_shifted_wcs \
      --torchsky-kernels bilinear --reproject-kernels bilinear \
      --drizzle-kernels point --swarp-kernels bilinear \
      --json-out "${artifact_root}/map_compat_real.json" \
      --baseline-json "${source_root}/benchmarks/baselines/map_compat_real_smoke.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "real-data gate failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  optimization-cpu)
    install_env test
    install_env sphere-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_map_image_plane.py" \
      --profile target-preparation-quick --size 512 --runs 5 --warmups 2 \
      --tile-sizes 128,256 --exposure-counts 1,4,24 --cache-capacities 0,1,4 \
      --device cpu --json-out "${artifact_root}/optimization_map_target_preparation_cpu.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/benchmarks/bench_solver_pcg.py" \
      --profile full --device cpu --runs 7 \
      --json-out "${artifact_root}/optimization_solver_pcg_cpu.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/benchmarks/bench_wcs_coupled_calibration.py" \
      --profile full --runs 5 \
      --json-out "${artifact_root}/optimization_calibration_graph_cpu.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_core.py" \
      --moc-only --sampling-profile full --device cpu --runs 7 \
      --json-out "${artifact_root}/optimization_random_moc_cpu.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --nside 512 --coverage-frac 0.1 --n-queries 200000 --sampling-profile full \
      --device cpu --runs 7 --json-out "${artifact_root}/optimization_random_sparse_cpu.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/benchmarks/bench_field_triplane_sampling.py" \
      --profile full --devices cpu --warmup 3 --runs 10 \
      --json-out "${artifact_root}/optimization_triplane_cpu.json"
    ;;
  optimization-cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_map_image_plane.py" \
      --profile target-preparation-quick --size 1024 --runs 7 --warmups 3 \
      --tile-sizes 128,256,512 --exposure-counts 1,4,24 --cache-capacities 0,1,4 \
      --device cuda --json-out "${artifact_root}/optimization_map_target_preparation_cuda.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_solver_pcg.py" \
      --profile full --device cuda --runs 10 \
      --json-out "${artifact_root}/optimization_solver_pcg_cuda.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_core.py" \
      --moc-only --sampling-profile full --device cuda --runs 10 \
      --json-out "${artifact_root}/optimization_random_moc_cuda.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --nside 512 --coverage-frac 0.1 --n-queries 200000 --sampling-profile full \
      --device cuda --runs 10 --json-out "${artifact_root}/optimization_random_sparse_cuda.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_field_triplane_sampling.py" \
      --profile full --devices cuda --warmup 3 --runs 10 \
      --json-out "${artifact_root}/optimization_triplane_cuda.json"
    ;;
  cpu)
    # ponytail: do not call wcs-sphere-acceptance-full here. That task's WCS
    # path auto-spawns pixi -e bench-legacy (pyast). The last CPU job
    # died there after finishing native WCS, and copy_tree never ran.
    install_env bench-all
    install_env sphere-bench
    record_versions bench-all
    cpu_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --wcs-only --no-gpu --no-legacy-wcs-bridge \
      --run-id acceptance_wcs_full \
      --output-dir "${artifact_root}" || cpu_rc=$?
    vos_upload after_wcs || echo "after_wcs VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only --no-gpu \
      --run-id acceptance_sphere_full \
      --output-dir "${artifact_root}" || cpu_rc=$?
    vos_upload after_sphere || echo "after_sphere VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      pytest \
      "${TORCHSKY_SUBJECT_ROOT}"/tests/test_wcs_*.py \
      "${TORCHSKY_SUBJECT_ROOT}"/tests/test_sphere_*.py \
      "${source_root}/tests/test_healsparse_upstream.py" \
      "${TORCHSKY_SUBJECT_ROOT}"/tests/test_healpix_*.py \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_spectral.py" \
      -q --tb=line \
      --junitxml="${artifact_root}/cpu_pytest.xml" || cpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set default --runs 2 --no-throughput-gate \
      --json-out "${artifact_root}/acceptance_spin_matrix.json" || cpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 32 --lmax 32 --runs 7 \
      --json-out "${artifact_root}/sphere_spectral.json" || cpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 64 --lmax 63 --runs 5 \
      --json-out "${artifact_root}/sphere_spectral_nside64.json" || cpu_rc=$?
    if [[ "${cpu_rc}" -ne 0 ]]; then
      echo "cpu job failures rc=${cpu_rc}"
      exit "${cpu_rc}"
    fi
    ;;
  cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install healpy || true
    # ponytail: pytest -k cuda is a smoke check, not the Linux evidence.
    # Keep benches running after a cuda-named test fails so VOS still gets
    # WCS/sphere JSON. Fail the job after those benches using cuda_rc.
    cuda_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      "${source_root}/tests/test_cuda_smoke.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_healpix_parity.py" \
      -q --tb=line -k cuda \
      --junitxml="${artifact_root}/cuda_pytest.xml" || cuda_rc=$?
    record_versions cuda-bench || cuda_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --wcs-only --wcs-device cuda --no-legacy-wcs-bridge \
      --run-id acceptance_wcs_cuda \
      --output-dir "${artifact_root}" || cuda_rc=$?
    vos_upload after_wcs || echo "after_wcs VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only \
      --run-id acceptance_sphere_cuda \
      --output-dir "${artifact_root}" || cuda_rc=$?
    if [[ "${cuda_rc}" -ne 0 ]]; then
      echo "cuda job failures rc=${cuda_rc}"
      exit "${cuda_rc}"
    fi
    ;;
  spectral)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 32 --lmax 32 --runs 7 \
      --json-out "${artifact_root}/sphere_spectral_nside32.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 64 --lmax 63 --runs 5 \
      --json-out "${artifact_root}/sphere_spectral_nside64.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 128 --lmax 128 --runs 3 \
      --json-out "${artifact_root}/sphere_spectral_nside128.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "spectral job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  sphere)
    install_env bench-all
    install_env sphere-bench
    record_versions sphere-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only --no-gpu \
      --run-id acceptance_sphere_full \
      --output-dir "${artifact_root}"
    ;;
  sparse | healsparse)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --nside 256 --coverage-frac 0.1 --n-queries 200000 --runs 9 \
      --json-out "${artifact_root}/sphere_sparse_nside256.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --nside 512 --coverage-frac 0.1 --n-queries 500000 --runs 7 \
      --json-out "${artifact_root}/sphere_sparse_nside512.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "sparse job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  spin)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set default --runs 5 --no-throughput-gate \
      --json-out "${artifact_root}/acceptance_spin_matrix.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set extended --runs 3 --no-throughput-gate \
      --json-out "${artifact_root}/acceptance_spin_matrix_extended.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "spin job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  geometry)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_matrix.py" \
      --device cpu --nside 1024 --n-points 200000 --runs 5 --strict-missing \
      --no-throughput-gate \
      --output-dir "${artifact_root}/sphere_matrix" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_healpix_advanced.py" \
      --device cpu --nside 1024 --n-points 200000 --runs 5 \
      --json-out "${artifact_root}/healpix_advanced.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "geometry job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  cuhpx)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install "git+https://github.com/NVlabs/cuHPX.git" || true
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_cuhpx_sht_spike.py" \
      --nside 32 --runs 3 \
      --json-out "${artifact_root}/cuhpx_sht_spike.json"
    record_versions cuda-bench || true
    ;;
  wcs-paper2)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases PCO,XPH,TSC,CSC,QSC,NCP \
      --n-points 100000 --runs 3 --device cpu \
      --json-out "${artifact_root}/wcs_paper2.json"
    TORCHSKY_WCS_FUSED=1 pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases CSC --n-points 20000 --runs 1 --device cpu \
      --json-out "${artifact_root}/wcs_paper2_fused1.json"
    TORCHSKY_WCS_FUSED=0 pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases CSC --n-points 20000 --runs 1 --device cpu \
      --json-out "${artifact_root}/wcs_paper2_fused0.json"
    ;;
  wcs-tpv)
    install_env bench-all
    record_versions bench-all
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases TPV --n-points 200000 --runs 5 --device cpu \
      --json-out "${artifact_root}/wcs_tpv.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_tpv_inverse.py" \
      --json-out "${artifact_root}/wcs_tpv_inverse.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_small_n_paths.py" \
      --cases TAN,SIN,ARC,TPV --n-points 256 --runs 11 \
      --json-out "${artifact_root}/wcs_small_n_paths.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "wcs-tpv job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  wcs-spectral)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_spectral_temporal.py" \
      --n-points 100000 --runs 3 \
      --json-out "${artifact_root}/wcs_spectral_temporal.json"
    ;;
  wcs-tabular)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_tabular.py" \
      --grid-size 8 --n-points 10000 --runs 5 \
      --json-out "${artifact_root}/wcs_tabular.json"
    ;;
  sht-profile)
    install_env sphere-bench
    record_versions sphere-bench
    export TORCHSKY_SHT_PROFILE=1
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set default --runs 5 --no-throughput-gate \
      --json-out "${artifact_root}/sht_profile_spin.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 128 --lmax 128 --runs 5 \
      --json-out "${artifact_root}/sht_profile_nside128.json"
    ;;
  moc)
    install_env sphere-bench
    record_versions sphere-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_core.py" \
      --moc-only --n-points 200000 --runs 5 --moc-max-order 8 \
      --json-out "${artifact_root}/moc_core.json"
    ;;
  catalog)
    install_env bench-all
    record_versions bench-all
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_compat.py" \
      --n-left 10000 --n-right-noise 80000 --runs 3 \
      --json-out "${artifact_root}/catalog_compat.json" || job_rc=$?
    vos_upload after_catalog_compat || echo "after_catalog_compat VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_match_exact.py" --scale 4 \
      --json-out "${artifact_root}/catalog_match_exact.json" || job_rc=$?
    vos_upload after_catalog_match_exact || echo "after_catalog_match_exact VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_match_perf.py" \
      --sizes 10000,100000,1000000 --threads "1,${catalog_threads}" --runs 3 \
      --json-out "${artifact_root}/catalog_match_perf_1e6.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "catalog job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  catalog-scale)
    install_env test
    record_versions test
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      pytest "${TORCHSKY_SUBJECT_ROOT}/tests/test_catalog_covering.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_catalog_no_dataframe.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_catalogs.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_crossmatch_engine.py" -q --tb=line \
      --junitxml="${artifact_root}/catalog_scale_pytest.xml" || job_rc=$?
    vos_upload after_catalog_scale_pytest || echo "after_catalog_scale_pytest VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/scripts/run_catalog_scale_contract.py" \
      --left-rows 200000 --layout dense --find all --engine-fast \
      --json-out "${artifact_root}/catalog_scale_200k_dense.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "catalog-scale job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  catalog-scale-1e6)
    install_env test
    record_versions test
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/scripts/run_catalog_scale_contract.py" \
      --left-rows 1000000 --layout sparse --find best --engine-fast \
      --json-out "${artifact_root}/catalog_scale_1e6_sparse_best.json" || job_rc=$?
    vos_upload after_catalog_scale_1e6_sparse || echo "after_catalog_scale_1e6_sparse VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/scripts/run_catalog_scale_contract.py" \
      --left-rows 1000000 --layout dense --find best --engine-fast \
      --json-out "${artifact_root}/catalog_scale_1e6_dense_best.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "catalog-scale-1e6 job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  catalog-scale-1e7)
    install_env test
    record_versions test
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/scripts/run_catalog_scale_contract.py" \
      --left-rows 10000000 --layout sparse --find best --engine-fast \
      --json-out "${artifact_root}/catalog_scale_1e7_sparse_best.json"
    ;;
  catalog-match-real)
    install_env test
    record_versions test
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/benchmarks/bench_catalog_match_real.py" \
      --hats-root /arc/projects/hats \
      --json-out "${artifact_root}/catalog_match_real.json"
    ;;
  catalog-match-1e7 | catalog-match-1e8)
    install_env bench-all
    record_versions bench-all
    if [[ "${mode}" == "catalog-match-1e7" ]]; then sizes="10000000"; else sizes="100000000"; fi
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_match_perf.py" \
      --sizes "${sizes}" --layouts sparse,dense --modes best,all --threads "${catalog_threads}" \
      --runs 1 --astropy-max-n 0 \
      --json-out "${artifact_root}/catalog_match_perf_${mode#catalog-match-}.json"
    ;;
  catalog-scale-real)
    install_env test
    record_versions test
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/scripts/run_catalog_scale_real.py" \
      --json-out "${artifact_root}/catalog_scale_real.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python - "${artifact_root}/catalog_scale_real.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("skipped"):
    raise SystemExit(f"real catalog data unavailable: {payload.get('reason', 'unknown')}")
PY
    ;;
  observed)
    install_env test
    record_versions test
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      pytest "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_catalog_state.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_frames.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_observed_frames.py" \
      -q --tb=line \
      --junitxml="${artifact_root}/observed_pytest.xml" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/benchmarks/bench_observed_place.py" \
      --json-out "${artifact_root}/observed_place.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "observed job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  geom-cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_polygons.py" \
      --device cuda --n-points 200000 --runs 3 \
      --json-out "${artifact_root}/geom_cuda.json"
    record_versions cuda-bench || true
    ;;
  gpu-validate)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy || true
    gpu_rc=0
    # Device-named package tests (CPU->CUDA portability of core kernels).
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_core.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_extended_projections.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_projection_matrix.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_invariants_randomized.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_science_invariants.py" \
      -q --tb=line -k cuda \
      --junitxml="${artifact_root}/gpu_validate_pytest.xml" || gpu_rc=$?
    # Harness comparison on GPU, including the reference-offset cases that
    # exposed the center-native inverse defect on CPU. Record metrics ungated:
    # CUDA float64 transcendental kernels differ from CPU, so the CPU
    # thresholds are not reused. The check below applies GPU-scale bounds and
    # prints the per-case maxima for the evidence bundle.
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m torchsky_harness.cli run bench_wcs -- \
      --cases TAN,SIN,ARC,AIT,MER,MER_OFF,CYP_OFF,AIT_OFF,HPX_OFF,SFL_OFF,XPH_OFF,BON_OFF,PAR_OFF,TSC,CSC,QSC,TPV \
      --n-points 50000 --runs 2 --device cuda \
      --json-out "${artifact_root}/wcs_gpu_validate.json" || gpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python - "${artifact_root}/wcs_gpu_validate.json" <<'PY' || gpu_rc=$?
import json
import sys

rows = json.load(open(sys.argv[1]))
bad = [row for row in rows if str(row.get("status", "OK")) != "OK"]
bad_cases = [row.get("case") for row in bad]
max_angular = max(float(row.get("max_angular_error_arcsec") or 0.0) for row in rows)
max_roundtrip = max(
    float(row.get("max_roundtrip_sky_error_arcsec_pole_excluded") or 0.0)
    for row in rows
    if row.get("case") not in {"TSC", "CSC", "QSC"}
)
print(
    f"gpu-check rows={len(rows)} max_angular={max_angular:.3e} asec "
    f"max_roundtrip={max_roundtrip:.3e} asec bad={bad_cases}"
)
sys.exit(0 if not bad and max_angular < 1e-2 and max_roundtrip < 1e-2 else 1)
PY
    record_versions cuda-bench || gpu_rc=$?
    if [[ "${gpu_rc}" -ne 0 ]]; then
      echo "gpu-validate job failures rc=${gpu_rc}"
      exit "${gpu_rc}"
    fi
    ;;
  world-cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install healpy || true
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_core.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_extended_projections.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_observed_frames.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_tabular.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_spectral.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_temporal.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_catalogs.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_sphere_polygons.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_sphere_sparse.py" \
      "${TORCHSKY_SUBJECT_ROOT}/tests/test_wcs_gwcs_sampling.py" \
      -q --tb=line -k cuda \
      --junitxml="${artifact_root}/world_cuda_pytest.xml" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases PCO,XPH,TSC,CSC,QSC,NCP \
      --n-points 100000 --runs 3 --device cuda \
      --json-out "${artifact_root}/wcs_paper2_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases TPV --n-points 200000 --runs 5 --device cuda \
      --json-out "${artifact_root}/wcs_tpv_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_tpv_inverse.py" \
      --n-points 200000 --runs 3 --device cuda \
      --json-out "${artifact_root}/wcs_tpv_inverse_cuda.json" \
      --markdown-out "${artifact_root}/wcs_tpv_inverse_cuda.md" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_small_n_paths.py" \
      --cases TAN,SIN,ARC,TPV --n-points 256 --runs 11 --device cuda \
      --json-out "${artifact_root}/wcs_small_n_paths_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_spectral_temporal.py" \
      --n-points 100000 --runs 3 --device cuda \
      --json-out "${artifact_root}/wcs_spectral_temporal_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_tabular.py" \
      --grid-size 8 --n-points 10000 --runs 5 --device cuda \
      --json-out "${artifact_root}/wcs_tabular_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_catalog_match_perf.py" \
      --sizes 100000,1000000 --layouts sparse,dense --modes best,all,knn2 --runs 3 \
      --device cuda --astropy-max-n 0 \
      --json-out "${artifact_root}/catalog_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_observed_place.py" \
      --device cuda --n-points 4096 --runs 7 \
      --json-out "${artifact_root}/observed_place_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_polygons.py" \
      --device cuda --n-points 200000 --runs 3 \
      --json-out "${artifact_root}/geom_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --device cuda --nside 256 --coverage-frac 0.1 --n-queries 200000 --runs 5 \
      --json-out "${artifact_root}/sphere_sparse_cuda.json" || job_rc=$?
    record_versions cuda-bench || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "world-cuda job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
esac

echo "completed_utc=$(date -u +%FT%TZ)"
echo "artifacts=${artifact_root}"
