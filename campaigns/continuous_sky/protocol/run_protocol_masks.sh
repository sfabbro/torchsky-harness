#!/usr/bin/env bash
# CPU job: noise factors + held-out masks (make_heldout_masks.py), then the "zero" reference method and its scores.
# Needs the output of run_official_align.sh (aligned WCS + reference mosaics) for FIELD.
#   canfar create headless images.canfar.net/astroai/base:latest -n protomasks -c 8 -m 64 -e FIELD=SMACS -- bash /arc/projects/ots/adass2026/scripts/run_protocol_masks.sh
# env: FIELD (required), STAGE (noise|masks|all|score; score = only the zero method + scoring, default all), WORKERS (default 8), ZERO (1 = also write/score the zero method, default 1), ARGS (extra masks flags)
set -uo pipefail
A=/arc/projects/ots/adass2026; P=$A/protocol/v1
FIELD=${FIELD:?FIELD required}
mkdir -p $P/logs $P/code $P/heldout/$FIELD
exec > >(tee -a $P/logs/masks_${FIELD}.log) 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) field=$FIELD"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu CRDS_PATH=$A/crds_cache
mkdir -p $PIP_CACHE_DIR $TMPDIR
python3 -m venv /scratch/venvM && /scratch/venvM/bin/pip install -q jwst scipy 2>&1 | grep -v notice
/scratch/venvM/bin/pip freeze > $P/code/pip-freeze-masks.txt
cp $A/scripts/make_heldout_masks.py $A/scripts/score.py $A/scripts/make_zero_pred.py $A/scripts/run_protocol_masks.sh $A/scripts/test_protocol_score.py $P/code/ 2>/dev/null
PY=/scratch/venvM/bin/python
if [ "${STAGE:-all}" != "score" ]; then
  $PY -u $A/scripts/make_heldout_masks.py --field $FIELD --root $P --stage ${STAGE:-all} --workers ${WORKERS:-8} ${ARGS:-} || { echo "masks failed"; exit 3; }
fi
if [ "${ZERO:-1}" = "1" ] && [ "${STAGE:-all}" != "noise" ]; then
  $PY -u $A/scripts/make_zero_pred.py --field $FIELD --root $P || exit 4
  for S in $($PY -c "import json;print(' '.join(json.load(open('$P/splits_$FIELD.json'))['splits']))"); do
    $PY -u $A/scripts/score.py score --field $FIELD --method zero --split $S --root $P
  done
fi
echo "exit=$? done $(date -u +%FT%TZ)"
