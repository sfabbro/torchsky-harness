#!/usr/bin/env bash
# CANFAR batch job for the pixel store (U0). One job = one (MISSION, shard) extraction, compacted into the arc store; or the
# finalize step (FINALIZE=1). Environment:
#   STORE     arc store directory, e.g. /arc/projects/ots/adass2026/pixelstore            (required)
#   MISSION   jwst | hst | spitzer | spherex | galex                                      (extract)
#   INPUTS    space-separated globs and/or manifest CSVs (quoted by this script, never expanded by the shell)   (extract)
#   DATA_DIR  space-separated directories searched for manifest file names                (optional)
#   SHARD     k/N, 0 <= k < N (default 0/1)
#   FINALIZE  1 -> only run `pixelstore.py finalize` on STORE (after all extract jobs finished)
#   SCRIPTS   default /arc/projects/ots/adass2026/scripts (pixelstore.py, hst_io.py, smacs_four_v31.py)
#   SCRATCH   default /scratch ; DRYRUN=1 prints the commands only
# Logs: $STORE/logs/pixelstore-<mission>-<k>of<N>.log  (finalize: pixelstore-finalize.log)
set -euo pipefail
SCRIPTS=${SCRIPTS:-/arc/projects/ots/adass2026/scripts}
SCRATCH=${SCRATCH:-/scratch}
: "${STORE:?set STORE}"
SHARD=${SHARD:-0/1}
K=${SHARD%/*}; N=${SHARD#*/}
mkdir -p "$STORE/logs"
if [ "${FINALIZE:-0}" = "1" ]; then TAG=finalize; else
  : "${MISSION:?set MISSION}"; : "${INPUTS:?set INPUTS}"; TAG="$MISSION-${K}of${N}"; fi
LOG="$STORE/logs/pixelstore-$TAG.log"
run() { if [ "${DRYRUN:-0}" = "1" ]; then echo "DRYRUN: $*"; else "$@"; fi; }
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u +%FT%TZ) pixelstore job $TAG on $(hostname) ==="

VENV="$SCRATCH/pixelstore-venv"
if [ "${DRYRUN:-0}" != "1" ]; then
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet numpy scipy astropy pyarrow shapely
  python3 -c "import torch, astropy, pyarrow; print('torch', torch.__version__, 'astropy', astropy.__version__, 'pyarrow', pyarrow.__version__)"
fi
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"

if [ "${FINALIZE:-0}" = "1" ]; then
  run python3 "$SCRIPTS/pixelstore.py" finalize --store "$STORE"
  run python3 "$SCRIPTS/pixelstore.py" summary --store "$STORE" --json "$STORE/logs/summary.json"
  exit 0
fi

set -f                                   # no pathname expansion: globs are passed to pixelstore.py as-is
ARGS=(extract --store "$STORE" --mission "$MISSION" --shard "$SHARD" --local-dir "$SCRATCH/pixelstore-local")
for i in $INPUTS; do ARGS+=(--inputs "$i"); done
for d in ${DATA_DIR:-}; do ARGS+=(--data-dir "$d"); done
set +f
run python3 "$SCRIPTS/pixelstore.py" "${ARGS[@]}"
echo "=== done $(date -u +%FT%TZ) ==="
