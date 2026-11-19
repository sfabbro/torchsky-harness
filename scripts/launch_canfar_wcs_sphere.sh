#!/usr/bin/env bash
# Launch CANFAR headless CPU and/or CUDA jobs for torchsky WCS + sphere.
# Usage: launch_canfar_wcs_sphere.sh [cpu|cuda|all|optimization|optimization-cpu|optimization-cuda|real-data-gates|suite|sht|spectral|sparse|healsparse|spin|cuhpx|wcs-paper2|wcs-tpv|wcs-spectral|wcs-tabular|sht-profile|moc|catalog|catalog-scale|catalog-scale-1e6|catalog-scale-1e7|catalog-scale-real|catalog-match-1e7|catalog-match-1e8|catalog-match-real|catalog-match-cuda|catalog-match-gpu-peers|observed|geom-cuda|world-cuda|world|depth]
#
# Jobs clone TORCHSKY_COMMIT from GitHub (must already be pushed), run
# scripts/run_canfar_wcs_sphere.sh from that clone, write
# ${TORCHSKY_ARC_ROOT:-/arc/projects/ots}/torchsky-runs/<run_id>/, and copy artifacts to
# vos:sfabbro/torchsky-runs/<run_id>/ (plus a gist fallback).
# suite: parallel spectral, sphere, sparse, spin, geometry, and cuHPX jobs.
# sht: spectral + sparse + spin only.
# world: paper2/tpv/spectral/tabular WCS, moc, catalog, observed-place, world-cuda.
# world-cuda: CUDA pytest + world-layer benches with tensors on the GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-all}"
case "${MODE}" in
  cpu | cuda | all | optimization | optimization-cpu | optimization-cuda | real-data-gates | suite | many | spectral | sparse | healsparse | spin | sht | cuhpx | \
  wcs-paper2 | wcs-tpv | wcs-spectral | wcs-tabular | sht-profile | moc | catalog | \
  catalog-scale | catalog-scale-1e6 | catalog-scale-1e7 | catalog-scale-real | \
  catalog-match-1e7 | catalog-match-1e8 | catalog-match-real | catalog-match-cuda | catalog-match-gpu-peers | \
  observed | geom-cuda | world-cuda | gpu-validate | world | depth) ;;
  *)
    echo "usage: $0 [cpu|cuda|all|optimization|optimization-cpu|optimization-cuda|real-data-gates|suite|sht|spectral|sparse|healsparse|spin|cuhpx|wcs-paper2|wcs-tpv|wcs-spectral|wcs-tabular|sht-profile|moc|catalog|catalog-scale|catalog-scale-1e6|catalog-scale-1e7|catalog-scale-real|catalog-match-1e7|catalog-match-1e8|catalog-match-real|catalog-match-cuda|catalog-match-gpu-peers|observed|geom-cuda|world-cuda|world|depth]" >&2
    exit 2
    ;;
esac

COMMIT="${TORCHSKY_HARNESS_COMMIT:-$(git rev-parse HEAD)}"
SUBJECT_COMMIT="${TORCHSKY_COMMIT:?set exact torchsky subject SHA}"
if [[ ! "${SUBJECT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "TORCHSKY_COMMIT must be a full 40-char subject SHA, got ${SUBJECT_COMMIT}" >&2
  exit 2
fi
if ! git cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  echo "TORCHSKY_COMMIT=${COMMIT} is not a local commit" >&2
  exit 2
fi
COMMIT="$(git rev-parse "${COMMIT}")"
if ! git fetch -q origin; then
  echo "git fetch origin failed; cannot check that ${COMMIT} is pushed" >&2
  exit 2
fi
if ! git branch -r --contains "${COMMIT}" | grep -q 'origin/'; then
  echo "TORCHSKY_COMMIT=${COMMIT} is not on origin; git push before launching CANFAR" >&2
  exit 2
fi
IMAGE="${CONTAINER_IMAGE:-images.canfar.net/astroai/notebook:latest}"
if [[ "${MODE}" == "depth" ]]; then
  IMAGE="${CONTAINER_IMAGE:-images.canfar.net/astroai/improc:latest}"
fi
CPU="${TORCHSKY_CANFAR_CPU:-8}"
MEMORY="${TORCHSKY_CANFAR_MEMORY:-32}"
# TORCHSKY_SPIN_THREADS matches --cpu. SpinMPool used to follow host
# hardware_concurrency and oversubscribe Skaha CFS quota.
STAMP="$(date -u +%Y%m%d-%H%M%S)"

if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found" >&2
  exit 1
fi
if [[ "${TORCHSKY_CANFAR_DRY_RUN:-0}" != "1" ]]; then
  canfar server use "${TORCHSKY_CANFAR_SERVER:-canfar}"
fi

launch_one() {
  local job_mode=$1
  local gpu_flag=$2
  local run_id="${3:-wcs-sphere-${job_mode}-${STAMP}}"
  local vos_dest="vos:sfabbro/torchsky-runs/${run_id}"
  local name
  name="$(printf 'torchsky-%s-%s' "${job_mode}" "${STAMP}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
  local local_out="${ROOT_DIR}/benchmarks_results/canfar_${run_id}"
  mkdir -p "${local_out}"

  # ponytail: Skaha splits argv on spaces; tabs keep bash -c as one script.
  # Do not put $ or & in the payload (CANFAR expands $name; & splits URLs).
  # Clone from GitHub. Do not vcp the runner: vault PUT has been storing 0-byte
  # nodes, and uncommitted local edits never reach the container anyway.
  local src="/scratch/${run_id}-src"
  # Fork commits are not in astroai/torchsky-harness: clone where COMMIT lives.
  local harness_repo="${TORCHSKY_HARNESS_REPO:-astroai/torchsky-harness}"
  local remote_plain="PATH=/opt/conda/bin:/opt/astroai/venv/cadc/bin:/opt/astroai/bin:/usr/local/bin:/usr/bin:/bin; gh auth setup-git; gh repo clone ${harness_repo} ${src} -- --filter=blob:none; git -C ${src} checkout --detach ${COMMIT}; bash ${src}/scripts/run_canfar_wcs_sphere.sh ${job_mode} ${run_id}"
  local remote_cmd
  remote_cmd="$(printf '%s' "${remote_plain}" | tr ' ' '\t')"
  case "${remote_cmd}" in
    *'$'* | *'&'* | *' '*)
      echo "refusing canfar payload with \$, &, or spaces: ${remote_plain}" >&2
      exit 2
      ;;
  esac

  local create_args=(
    create headless "${IMAGE}"
    --name "${name}"
    --cpu "${CPU}"
    --memory "${MEMORY}"
    --env "TORCHSKY_HARNESS_COMMIT=${COMMIT}"
    --env "TORCHSKY_COMMIT=${SUBJECT_COMMIT}"
    --env "TORCHSKY_SUBJECT_ROOT=/scratch/${run_id}-subject"
    --env "TORCHSKY_SUBJECT_REPO=${TORCHSKY_SUBJECT_REPO:-astroai/torchsky}"
    --env "TORCHSKY_HARNESS_REPO=${harness_repo}"
    --env "TORCHSKY_VOS_DEST=${vos_dest}"
    --env "CONTAINER_IMAGE=${IMAGE}"
    --env "PYTHONUNBUFFERED=1"
    --env "TORCHSKY_SOURCE_ROOT=${src}"
    --env "TORCHSKY_SPIN_THREADS=${CPU}"
    --env "TORCHSKY_SHT_PROFILE=1"
  )
  if [[ -n "${gpu_flag}" ]]; then
    create_args+=(--gpu "${gpu_flag}")
  fi
  # Optional catalog-match-gpu-peers selection (comma lists, no spaces).
  if [[ -n "${TORCHSKY_CATALOG_GPU_PEERS:-}" ]]; then
    create_args+=(--env "TORCHSKY_CATALOG_GPU_PEERS=${TORCHSKY_CATALOG_GPU_PEERS}")
  fi
  if [[ -n "${TORCHSKY_CATALOG_SIZES:-}" ]]; then
    create_args+=(--env "TORCHSKY_CATALOG_SIZES=${TORCHSKY_CATALOG_SIZES}")
  fi
  # Only forward a GitHub token when explicitly requested. The AstroAI
  # notebook image already has session GitHub credentials; do not scrape
  # `gh auth token` into CANFAR argv or logs.
  if [[ -n "${TORCHSKY_GH_TOKEN:-}" ]]; then
    create_args+=(--env "GH_TOKEN=${TORCHSKY_GH_TOKEN}")
  fi

  echo "launching mode=${job_mode} name=${name} commit=${COMMIT} vos=${vos_dest}"
  local create_log="${local_out}/create.log"
  if [[ "${TORCHSKY_CANFAR_DRY_RUN:-0}" == "1" ]]; then
    echo "dry-run: canfar create headless ${IMAGE} --name ${name} --cpu ${CPU} --memory ${MEMORY} gpu=${gpu_flag:-omit}"
    echo "${run_id}" > "${local_out}/run_id.txt"
    echo "${vos_dest}" > "${local_out}/vos_dest.txt"
    echo "${COMMIT}" > "${local_out}/commit.txt"
    return 0
  fi
  set +o pipefail
  canfar "${create_args[@]}" -- bash -c "${remote_cmd}" 2>&1 | tee "${create_log}"
  local create_rc=${PIPESTATUS[0]}
  set -o pipefail
  if [[ "${create_rc}" -ne 0 ]]; then
    echo "canfar create failed (rc=${create_rc}); see ${create_log}" >&2
    exit 1
  fi

  local session_id
  session_id="$(
    python3 - "${create_log}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r"ID:\s*([^)\s]+)", text)
print(m.group(1).strip() if m else "")
PY
  )"
  if [[ -z "${session_id}" ]]; then
    echo "could not parse session ID from ${create_log}" >&2
    exit 1
  fi
  echo "${session_id}" > "${local_out}/session_id.txt"
  echo "${run_id}" > "${local_out}/run_id.txt"
  echo "${vos_dest}" > "${local_out}/vos_dest.txt"
  echo "${COMMIT}" > "${local_out}/commit.txt"
  echo "session_id=${session_id} run_id=${run_id} vos=${vos_dest}"
}

if [[ "${MODE}" == "cpu" || "${MODE}" == "all" ]]; then
  launch_one cpu ""
fi
if [[ "${MODE}" == "cuda" || "${MODE}" == "all" ]]; then
  launch_one cuda 1
fi
if [[ "${MODE}" == "optimization-cpu" || "${MODE}" == "optimization" ]]; then
  launch_one optimization-cpu "" "optimization-cpu-${STAMP}"
fi
if [[ "${MODE}" == "optimization-cuda" || "${MODE}" == "optimization" ]]; then
  launch_one optimization-cuda 1 "optimization-cuda-${STAMP}"
fi
if [[ "${MODE}" == "real-data-gates" ]]; then
  launch_one real-data-gates "" "real-data-gates-${STAMP}"
fi
if [[ "${MODE}" == "suite" || "${MODE}" == "many" ]]; then
  launch_one spectral "" "sphere-spectral-${STAMP}"
  launch_one sphere "" "sphere-only-${STAMP}"
  launch_one sparse "" "sphere-sparse-${STAMP}"
  launch_one spin "" "sphere-spin-${STAMP}"
  launch_one geometry "" "sphere-geometry-${STAMP}"
  launch_one cuhpx 1 "cuhpx-sht-${STAMP}"
fi
if [[ "${MODE}" == "spectral" || "${MODE}" == "sht" ]]; then
  launch_one spectral "" "sphere-spectral-${STAMP}"
fi
if [[ "${MODE}" == "sparse" || "${MODE}" == "sht" || "${MODE}" == "healsparse" ]]; then
  launch_one sparse "" "sphere-sparse-${STAMP}"
fi
if [[ "${MODE}" == "spin" || "${MODE}" == "sht" ]]; then
  launch_one spin "" "sphere-spin-${STAMP}"
fi
if [[ "${MODE}" == "cuhpx" ]]; then
  launch_one cuhpx 1 "cuhpx-sht-${STAMP}"
fi
if [[ "${MODE}" == "wcs-paper2" ]]; then
  launch_one wcs-paper2 "" "wcs-paper2-${STAMP}"
fi
if [[ "${MODE}" == "wcs-tpv" ]]; then
  launch_one wcs-tpv "" "wcs-tpv-${STAMP}"
fi
if [[ "${MODE}" == "wcs-spectral" ]]; then
  launch_one wcs-spectral "" "wcs-spectral-${STAMP}"
fi
if [[ "${MODE}" == "wcs-tabular" ]]; then
  launch_one wcs-tabular "" "wcs-tabular-${STAMP}"
fi
if [[ "${MODE}" == "sht-profile" ]]; then
  launch_one sht-profile "" "sht-profile-${STAMP}"
fi
if [[ "${MODE}" == "moc" ]]; then
  launch_one moc "" "moc-${STAMP}"
fi
if [[ "${MODE}" == "catalog" ]]; then
  launch_one catalog "" "catalog-${STAMP}"
fi
if [[ "${MODE}" == "catalog-scale" ]]; then
  launch_one catalog-scale "" "catalog-scale-${STAMP}"
fi
if [[ "${MODE}" == "catalog-scale-1e6" ]]; then
  launch_one catalog-scale-1e6 "" "catalog-scale-1e6-${STAMP}"
fi
if [[ "${MODE}" == "catalog-scale-1e7" ]]; then
  launch_one catalog-scale-1e7 "" "catalog-scale-1e7-${STAMP}"
fi
if [[ "${MODE}" == "catalog-scale-real" ]]; then
  launch_one catalog-scale-real "" "catalog-scale-real-${STAMP}"
fi
if [[ "${MODE}" == "catalog-match-cuda" ]]; then
  launch_one catalog-match-cuda 1 "catalog-match-cuda-${STAMP}"
fi
if [[ "${MODE}" == "catalog-match-gpu-peers" ]]; then
  launch_one catalog-match-gpu-peers 1 "catalog-match-gpu-peers-${STAMP}"
fi
if [[ "${MODE}" == "catalog-match-1e7" || "${MODE}" == "catalog-match-1e8" || "${MODE}" == "catalog-match-real" ]]; then
  launch_one "${MODE}" "" "${MODE}-${STAMP}"
fi
if [[ "${MODE}" == "observed" ]]; then
  launch_one observed "" "observed-${STAMP}"
fi
if [[ "${MODE}" == "gpu-validate" ]]; then
  launch_one gpu-validate 1 "gpu-validate-${STAMP}"
fi
if [[ "${MODE}" == "geom-cuda" ]]; then
  launch_one geom-cuda 1 "geom-cuda-${STAMP}"
fi
if [[ "${MODE}" == "world-cuda" ]]; then
  launch_one world-cuda 1 "world-cuda-${STAMP}"
fi
if [[ "${MODE}" == "world" ]]; then
  launch_one wcs-paper2 "" "wcs-paper2-${STAMP}"
  launch_one wcs-tpv "" "wcs-tpv-${STAMP}"
  launch_one wcs-spectral "" "wcs-spectral-${STAMP}"
  launch_one wcs-tabular "" "wcs-tabular-${STAMP}"
  launch_one moc "" "moc-${STAMP}"
  launch_one catalog "" "catalog-${STAMP}"
  launch_one observed "" "observed-${STAMP}"
  launch_one world-cuda 1 "world-cuda-${STAMP}"
fi

# depth: autonomous deep-field reconstruction chain (run_canfar_adass_depth.sh
# full). Env FIELD_MODELS / PSF_MODEL / BACKGROUND_MODEL / MODULES /
# CONTAINER_IMAGE are forwarded; exact-SHA checkout and image enforcement stay
# in the runner.
launch_depth() {
  local run_id="depth-${STAMP}"
  local depth_mode="${DEPTH_MODE:-full}"
  local vos_dest="vos:sfabbro/torchsky-runs/${run_id}"
  local map_compat_env="${TORCHSKY_MAP_COMPAT_ENV:-map-compat-cuda}"
  if [[ "${TORCHSKY_DEVICE:-auto}" == "cpu" && -z "${TORCHSKY_MAP_COMPAT_ENV:-}" ]]; then
    map_compat_env=map-compat
  fi
  local name
  name="$(printf 'torchsky-depth-%s' "${STAMP}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
  local local_out="${ROOT_DIR}/benchmarks_results/canfar_${run_id}"
  mkdir -p "${local_out}"

  local src="/scratch/${run_id}-src"
  # Fork commits are not in astroai/torchsky-harness: clone where COMMIT lives.
  local harness_repo="${TORCHSKY_HARNESS_REPO:-astroai/torchsky-harness}"
  local remote_plain="PATH=/opt/conda/bin:/opt/astroai/venv/cadc/bin:/opt/astroai/bin:/usr/local/bin:/usr/bin:/bin; gh auth setup-git; gh repo clone ${harness_repo} ${src} -- --filter=blob:none; git -C ${src} checkout --detach ${COMMIT}; bash ${src}/scripts/run_canfar_adass_depth.sh ${depth_mode} ${run_id}"
  local remote_cmd
  remote_cmd="$(printf '%s' "${remote_plain}" | tr ' ' '\t')"
  case "${remote_cmd}" in
    *'$'* | *'&'* | *' '*)
      echo "refusing canfar payload with \$, &, or spaces: ${remote_plain}" >&2
      exit 2
      ;;
  esac

  local create_args=(
    create headless "${IMAGE}"
    --name "${name}"
    --cpu "${CPU}"
    --memory "${MEMORY}"
    --env "TORCHSKY_HARNESS_COMMIT=${COMMIT}"
    --env "TORCHSKY_COMMIT=${SUBJECT_COMMIT}"
    --env "TORCHSKY_SUBJECT_ROOT=/scratch/${run_id}-subject"
    --env "TORCHSKY_SUBJECT_REPO=${TORCHSKY_SUBJECT_REPO:-astroai/torchsky}"
    --env "TORCHSKY_HARNESS_REPO=${harness_repo}"
    --env "TORCHSKY_VOS_DEST=${vos_dest}"
    --env "CONTAINER_IMAGE=${IMAGE}"
    --env "PYTHONUNBUFFERED=1"
    --env "TORCHSKY_SPIN_THREADS=${CPU}"
    --env "TORCHSKY_MAP_COMPAT_ENV=${map_compat_env}"
    --env "FIELD_MODELS=${FIELD_MODELS:-spectral_power_foam}"
    --env "FOAM_BANDS=${FOAM_BANDS:-F090W,F150W}"
    --env "FOAM_CROP_SIZE=${FOAM_CROP_SIZE:-32}"
    --env "FOAM_STEPS=${FOAM_STEPS:-2}"
    --env "FOAM_GRID_SIDE=${FOAM_GRID_SIDE:-4}"
    --env "FOAM_BASIS_COUNT=${FOAM_BASIS_COUNT:-3}"
    --env "FOAM_INITIALIZE_FROM_COADD=${FOAM_INITIALIZE_FROM_COADD:-0}"
    --env "FOAM_AFFINE_SPATIAL_BASIS=${FOAM_AFFINE_SPATIAL_BASIS:-0}"
    --env "FOAM_WHITEN_SPECTRAL_BASIS=${FOAM_WHITEN_SPECTRAL_BASIS:-0}"
    --env "FOAM_TARGET_HEADER_PATH=${FOAM_TARGET_HEADER_PATH:-}"
    --env "GRID_FIT_TARGET_FOOTPRINT=${GRID_FIT_TARGET_FOOTPRINT:-0}"
    --env "GRID_PSF_MARGIN_SIGMA=${GRID_PSF_MARGIN_SIGMA:-4.0}"
    --env "MODEL_KWARGS=${MODEL_KWARGS:-}"
    --env "SCENE_INITIALIZATION=${SCENE_INITIALIZATION:-common_product}"
    --env "SCENE_INITIALIZATION_STEPS=${SCENE_INITIALIZATION_STEPS:-32}"
    --env "SCENE_INITIALIZATION_LR=${SCENE_INITIALIZATION_LR:-0.05}"
    --env "PASSBAND_OUTPUT_DIR=${PASSBAND_OUTPUT_DIR:-${TORCHSKY_ARC_ROOT:-/arc/projects/ots}/jwst_smacs0723/data/calibration/svo}"
    --env "PASSBAND_BANDS=${PASSBAND_BANDS:-F090W,F150W,F200W,F277W,F356W,F444W}"
    --env "PASSBAND_CACHE_DIR=${PASSBAND_CACHE_DIR:-}"
    --env "PSF_MODEL=${PSF_MODEL:-fitted_gaussian}"
    --env "CALIBRATION_EVIDENCE_MANIFEST=${CALIBRATION_EVIDENCE_MANIFEST:-}"
    --env "CALIBRATION_FIT_REPORT=${CALIBRATION_FIT_REPORT:-}"
    --env "MEASURED_PASSBAND_DIR=${MEASURED_PASSBAND_DIR:-}"
    --env "INDEPENDENT_FROZEN_CALIBRATION=${INDEPENDENT_FROZEN_CALIBRATION:-0}"
    --env "CALIBRATION_FAMILY=${CALIBRATION_FAMILY:-stationary}"
    --env "CALIBRATION_FIT_DEVICE=${CALIBRATION_FIT_DEVICE:-cpu}"
    --env "BACKGROUND_MODEL=${BACKGROUND_MODEL:-none}"
    --env "COMPARATOR_TOOL=${COMPARATOR_TOOL:-swarp}"
    --env "MODULES=${MODULES:-b}"
    --env "MOSAIC_BANDS=${MOSAIC_BANDS:-}"
    --env "MULTIBAND_CANARY_SEED=${MULTIBAND_CANARY_SEED:-101}"
    --env "MULTIBAND_CANARY_SIZE=${MULTIBAND_CANARY_SIZE:-96}"
    --env "MULTIBAND_CANARY_BANDS=${MULTIBAND_CANARY_BANDS:-6}"
    --env "MULTIBAND_CANARY_SOURCES=${MULTIBAND_CANARY_SOURCES:-24}"
    --env "MULTIBAND_CANARY_EXPOSURES=${MULTIBAND_CANARY_EXPOSURES:-4}"
    --env "MULTIBAND_CANARY_FOAM_GRID_SIDE=${MULTIBAND_CANARY_FOAM_GRID_SIDE:-}"
    --env "TORCHSKY_DATA_ROOT=${TORCHSKY_DATA_ROOT:-}"
    --env "UPDEC_REFERENCE_ROOT=${UPDEC_REFERENCE_ROOT:-}"
    --env "MOSAIC_INPUT_ROOT=${MOSAIC_INPUT_ROOT:-}"
    --env "MOSAIC_EXPECTED_TOTAL_EXPOSURES=${MOSAIC_EXPECTED_TOTAL_EXPOSURES:-135}"
    --env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}"
    --env "TORCHSKY_DEVICE=${TORCHSKY_DEVICE:-}"
  )
  # MIG slices assert inside the torch caching allocator (torch 2.10);
  # only take GPU slots when the operator will actually run on CUDA.
  # TORCHSKY_CANFAR_GPUS: number of GPU slots (the foam fit itself is
  # single-device; extra slots only expose more devices to the session).
  local needs_gpu=0
  if [[ "${depth_mode}" == "calibration-fit" ]]; then
    [[ "${CALIBRATION_FIT_DEVICE:-cpu}" == "cuda" ]] && needs_gpu=1
  elif [[ "${depth_mode}" != "passband-bootstrap" && "${TORCHSKY_DEVICE:-auto}" != "cpu" ]]; then
    needs_gpu=1
  fi
  if [[ "${needs_gpu}" == 1 ]]; then
    create_args+=(--gpu "${TORCHSKY_CANFAR_GPUS:-1}")
  fi
  if [[ -n "${TORCHSKY_GH_TOKEN:-}" ]]; then
    create_args+=(--env "GH_TOKEN=${TORCHSKY_GH_TOKEN}")
  fi

  echo "launching mode=depth name=${name} commit=${COMMIT} vos=${vos_dest} field_models=${FIELD_MODELS:-spectral_power_foam} psf=${PSF_MODEL:-fitted_gaussian} modules=${MODULES:-b}"
  local create_log="${local_out}/create.log"
  if [[ "${TORCHSKY_CANFAR_DRY_RUN:-0}" == "1" ]]; then
    echo "dry-run: canfar create headless ${IMAGE} --name ${name} --gpu 1 depth-env"
    echo "${run_id}" > "${local_out}/run_id.txt"
    return 0
  fi
  set +o pipefail
  canfar "${create_args[@]}" -- bash -c "${remote_cmd}" 2>&1 | tee "${create_log}"
  local create_rc=${PIPESTATUS[0]}
  set -o pipefail
  if [[ "${create_rc}" -ne 0 ]]; then
    echo "canfar create failed (rc=${create_rc}); see ${create_log}" >&2
    exit 1
  fi
  echo "${run_id}" > "${local_out}/run_id.txt"
}

if [[ "${MODE}" == "depth" ]]; then
  launch_depth
fi
