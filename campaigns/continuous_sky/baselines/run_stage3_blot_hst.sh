#!/usr/bin/env bash
# CPU job: stage3_blot for the SMACS HST WFC3/IR exposures (AstroDrizzle + blot), see stage3_blot_hst.py.
#   canfar create headless images.canfar.net/astroai/base:latest -n s3blothst -c 4 -m 32 -- bash /arc/projects/ots/adass2026/scripts/run_stage3_blot_hst.sh
# env: BANDS (default all) SPLIT ARGS TAG
set -uo pipefail
A=/arc/projects/ots/adass2026
TAG=${TAG:-hst}
LOG=$A/logs/stage3_blot_$TAG.log
mkdir -p "$A/logs" "$A/baselines/stage3_blot/provenance"
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 MPLBACKEND=Agg
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" /scratch/stage3_blot_hst
python3 --version
python3 -m venv /scratch/venvH && /scratch/venvH/bin/pip install -q drizzlepac || { echo "pip failed"; exit 4; }
P=$A/baselines/stage3_blot/provenance
/scratch/venvH/bin/pip freeze > "$P/pip-freeze_$TAG.txt"
cp "$A/scripts/stage3_blot_hst.py" "$A/scripts/run_stage3_blot_hst.sh" "$P/" 2>/dev/null
FL="--split ${SPLIT:-dither}"
[ -n "${BANDS:-}" ] && FL="$FL --bands $BANDS"
echo "stage3_blot_hst.py $FL ${ARGS:-}"
# shellcheck disable=SC2086
/scratch/venvH/bin/python -W ignore -u "$A/scripts/stage3_blot_hst.py" $FL ${ARGS:-}
rc=$?
echo "exit=$rc done $(date -u +%FT%TZ)"
exit $rc
