#!/usr/bin/env bash
# CPU job (no GPU): stage3_blot baseline (calwebb_image3 + blot) -- see stage3_blot.py.
# Launch (8 cores, 64 GB), e.g.:
#   canfar create headless images.canfar.net/astroai/base:latest -n s3blotsmacs -c 8 -m 64 -e FIELD=SMACS -e PROCS=3 -- bash /arc/projects/ots/adass2026/scripts/run_stage3_blot.sh
# env: FIELD (SMACS|LMC)  PROCS (parallel tasks, default 3)  BANDS  TILES  SPLIT (default dither)  TAG (log name)  ARGS (extra stage3_blot.py flags)
#      JWST_SPEC (pip spec, default 'jwst')  OUTROOT PRED_ROOT
set -uo pipefail
A=/arc/projects/ots/adass2026
TAG=${TAG:-${FIELD:-x}_$(date -u +%m%dT%H%M)}
LOG=$A/logs/stage3_blot_$TAG.log
mkdir -p "$A/logs" "$A/crds_cache"
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) FIELD=${FIELD:-} BANDS=${BANDS:-} TILES=${TILES:-}"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP:-2} OPENBLAS_NUM_THREADS=${OMP:-2} MKL_NUM_THREADS=${OMP:-2}
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu CRDS_PATH=$A/crds_cache
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" /scratch/stage3_blot
python3 --version
python3 -m venv /scratch/venvS && /scratch/venvS/bin/pip install -q "${JWST_SPEC:-jwst}" || { echo "pip failed"; exit 4; }
OUTROOT=${OUTROOT:-$A/baselines/stage3_blot}
mkdir -p "$OUTROOT/provenance"
/scratch/venvS/bin/pip freeze > "$OUTROOT/provenance/pip-freeze_$TAG.txt"
cp "$A/scripts/stage3_blot.py" "$A/scripts/run_stage3_blot.sh" "$OUTROOT/provenance/" 2>/dev/null
printf '%s\n' "FIELD=${FIELD:-}" "BANDS=${BANDS:-}" "TILES=${TILES:-}" "SPLIT=${SPLIT:-dither}" "ARGS=${ARGS:-}" "CRDS_PATH=$CRDS_PATH" > "$OUTROOT/provenance/launch_$TAG.env"
FL="--field ${FIELD:?} --split ${SPLIT:-dither} --procs ${PROCS:-3} --outroot $OUTROOT"
[ -n "${BANDS:-}" ] && FL="$FL --bands $BANDS"
[ -n "${TILES:-}" ] && FL="$FL --tiles $TILES"
[ -n "${PRED_ROOT:-}" ] && FL="$FL --pred-root $PRED_ROOT"
echo "stage3_blot.py $FL ${ARGS:-}"
# shellcheck disable=SC2086
/scratch/venvS/bin/python -u "$A/scripts/stage3_blot.py" $FL ${ARGS:-}
rc=$?
echo "exit=$rc done $(date -u +%FT%TZ)"
exit $rc
