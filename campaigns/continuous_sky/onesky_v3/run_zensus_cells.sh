#!/usr/bin/env bash
# CPU job: count spectroscopic redshifts per HEALPix cell of zensus (read only). Everything is written under onesky_v3/zensus_cells.
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=$A/onesky_v3/zensus_cells; mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ)"
mkdir -p /scratch/job && cp "$A/scripts/onesky_v3/zensus_cells.py" /scratch/job/job.py
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venv
/scratch/venv/bin/pip install -q numpy pyarrow healpy astropy
/scratch/venv/bin/python -u /scratch/job/job.py --hats "${HATS:-/arc/projects/hats/zensus/20261019/zensus_20261019}" --out "$OUT" --workers 8
echo "done rc=$? $(date -u +%FT%TZ)"
