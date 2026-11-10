#!/usr/bin/env bash
# env: FETCH (empty = inventory only, else 1), FETCH_PROGRAMS, FETCH_FILTERS
set -uo pipefail
A=/arc/projects/ots/adass2026; D=/arc/projects/ots/torchsky-data/aggregate
mkdir -p $D
exec > >(tee -a "$D/inventory_session.log") 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvI && /scratch/venvI/bin/pip install -q numpy astropy astroquery
EXTRA=""
if [ -n "${FETCH:-}" ]; then EXTRA="--fetch $D --fetch-programs ${FETCH_PROGRAMS:-1476,4447} --fetch-filters ${FETCH_FILTERS:-F090W,F150W,F200W,F277W,F356W,F444W}"; fi
/scratch/venvI/bin/python -u $A/scripts/aggregate_inventory.py --out $D $EXTRA
echo "exit=$? done $(date -u +%FT%TZ)"
