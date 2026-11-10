#!/usr/bin/env bash
# E2 physics PSF libraries (warm starts). env: OUTV (subdir name, default v1)
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=$A/psflib/${OUTV:-v1}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 STPSF_PATH=$A/stpsf-data/stpsf-data
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$A/stpsf-data"
if [ ! -d "$STPSF_PATH" ]; then
  curl -sSL -o /scratch/stpsf-data.tar.gz https://stsci.box.com/shared/static/kqfolg2bfzqc4mjkgmujo06d3iaymahv.gz && tar -xzf /scratch/stpsf-data.tar.gz -C $A/stpsf-data
fi
ls $A/stpsf-data
python3 -m venv /scratch/venvP
/scratch/venvP/bin/pip install -q numpy scipy astropy astroquery stpsf
/scratch/venvP/bin/pip freeze > "$OUT/pip-freeze.txt"
cp $A/scripts/gen_psf_lib.py "$OUT/"
/scratch/venvP/bin/python -u $A/scripts/gen_psf_lib.py "$OUT" /arc/projects/ots/torchsky-data/jwst_smacs0723/cal
echo "exit=$? done $(date -u +%FT%TZ)"
