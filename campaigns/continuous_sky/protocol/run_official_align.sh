#!/usr/bin/env bash
# CPU job: official JWST pipeline alignment (tweakreg, skymatch, outlier_detection, resample) of all exposures of a protocol field.
# Launch (example, LMC, 4 tiles in parallel; each tile is single-threaded):
#   canfar create headless images.canfar.net/astroai/base:latest -n alignlmc -c 8 -m 64 -e FIELD=LMC -e WORKERS=4 -- bash /arc/projects/ots/adass2026/scripts/run_official_align.sh
# env:
#   FIELD    SMACS | LMC               (required)
#   WORKERS  parallel units (LMC tiles)  default 4
#   UNITS    comma list of LMC tile ids  default all
#   EXTRA    extra official_align.py flags (e.g. "--preset default --stage tweakreg --tag defaultcheck")
#   TAG      log-name suffix
#   MAKE_SPLITS  1 = (re)run make_splits.py for FIELD first (default: only if splits_<FIELD>.json is missing)
set -uo pipefail
A=/arc/projects/ots/adass2026
P=$A/protocol/v1
FIELD=${FIELD:?FIELD required}
TAG=${TAG:-}
mkdir -p "$P/logs" "$P/code"
exec > >(tee -a "$P/logs/align_${FIELD}${TAG:+_$TAG}.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) field=$FIELD"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu CRDS_PATH=$A/crds_cache
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$CRDS_PATH" /scratch/align_work
python3 -m venv /scratch/venvJ && /scratch/venvJ/bin/pip install -q jwst scipy 2>&1 | grep -v notice
/scratch/venvJ/bin/pip freeze > "$P/code/pip-freeze-align.txt"
cp "$A/scripts/official_align.py" "$A/scripts/make_splits.py" "$A/scripts/run_official_align.sh" "$P/code/" 2>/dev/null
if [ ! -f "$P/splits_${FIELD}.json" ] || [ -n "${MAKE_SPLITS:-}" ]; then
  /scratch/venvJ/bin/python -u "$A/scripts/make_splits.py" --fields "$FIELD" --out "$P" || { echo "make_splits failed"; exit 3; }
fi
# shellcheck disable=SC2086
/scratch/venvJ/bin/python -u "$A/scripts/official_align.py" --field "$FIELD" --out "$P" --workers "${WORKERS:-4}" ${UNITS:+--units $UNITS} ${EXTRA:-}
rc=$?
echo "exit=$rc done $(date -u +%FT%TZ)"
exit $rc
