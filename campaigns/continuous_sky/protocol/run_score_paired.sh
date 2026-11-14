#!/usr/bin/env bash
# Paired scores and residual power by angular scale (protocol/v1/code/score_paired.py, the companion of score.py).
# Modelled on run_score.sh (venv + pip install, tee to a log) and run_py_cpu.sh (matplotlib, Agg backend).
# Environment:  FIELD (SMACS or LMC)   SPLIT (default dither)   CMD (paired, byscale or both; default both)
#               METHODS (space-separated, e.g. "fieldop_fix_200 fieldop stage3_blot")   REF (reference method, needed for paired)
#               ARGS (extra arguments passed to both commands, e.g. "--nboot 1000")
# score_paired.py must sit next to score.py in $A/protocol/v1/code/ (it imports score.py).  Output: $A/protocol/v1/scores_paired/<FIELD>/.
# Does only reads of the protocol folder; writes only scores_paired/ and the log.
set -uo pipefail
A=/arc/projects/ots/adass2026
LOG=$A/logs/score_paired_${FIELD:-f}_${CMD:-both}.log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 MPLBACKEND=Agg
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvC && /scratch/venvC/bin/pip install -q numpy astropy scipy matplotlib || exit 4
S=$A/protocol/v1/code/score_paired.py
PY=/scratch/venvC/bin/python
rc=0
if [ "${CMD:-both}" != "byscale" ]; then
  $PY -u $S paired --field "${FIELD:?}" --split "${SPLIT:-dither}" --methods ${METHODS:?} --reference "${REF:?}" ${ARGS:-}
  rc=$?; echo "paired exit=$rc"
fi
if [ "${CMD:-both}" != "paired" ]; then
  $PY -u $S byscale --field "${FIELD:?}" --split "${SPLIT:-dither}" --methods ${METHODS:?} ${ARGS:-}
  rc2=$?; echo "byscale exit=$rc2"; [ $rc2 -ne 0 ] && rc=$rc2
fi
echo "exit=$rc done $(date -u +%FT%TZ)"
exit $rc
