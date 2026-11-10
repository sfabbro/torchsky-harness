#!/usr/bin/env bash
# CPU job: protocol v1 splits for SMACS and LMC (headers only).  canfar create headless images.canfar.net/astroai/base:latest -n protosplits -c 4 -m 16 -- bash /arc/projects/ots/adass2026/scripts/run_splits.sh
set -uo pipefail
A=/arc/projects/ots/adass2026; P=$A/protocol/v1; mkdir -p $P/logs $P/code
exec > >(tee -a $P/logs/splits.log) 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp; mkdir -p $PIP_CACHE_DIR $TMPDIR
python3 -m venv /scratch/venvS && /scratch/venvS/bin/pip install -q numpy astropy 2>&1 | grep -v notice
cp $A/scripts/make_splits.py $P/code/
/scratch/venvS/bin/python -u $A/scripts/make_splits.py --out $P --workers 8 ${ARGS:-}
echo "exit=$? done $(date -u +%FT%TZ)"
