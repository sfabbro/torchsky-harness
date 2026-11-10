#!/usr/bin/env bash
# Score a method with the protocol's scorer (protocol/v1/code/score.py). env: FIELD METHOD SPLIT (default dither)
set -uo pipefail
A=/arc/projects/ots/adass2026
LOG=$A/logs/score_${METHOD:-m}_${FIELD:-f}.log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvC && /scratch/venvC/bin/pip install -q numpy astropy scipy || exit 4
S=$A/protocol/v1/code/score.py
/scratch/venvC/bin/python -u $S score --field "${FIELD:?}" --method "${METHOD:?}" --split "${SPLIT:-dither}" ${ARGS:-}
/scratch/venvC/bin/python -u $S compare --field "$FIELD" --split "${SPLIT:-dither}"
echo "exit=$? done $(date -u +%FT%TZ)"
