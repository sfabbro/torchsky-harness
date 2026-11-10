#!/usr/bin/env bash
# Generic CPU job: env SCRIPT (in /arc/projects/ots/adass2026/scripts), ARGS, LOG (path)
set -uo pipefail
A=/arc/projects/ots/adass2026
exec > >(tee -a "${LOG:-$A/logs/cpu_job.log}") 2>&1
echo "start $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 MPLBACKEND=Agg
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvP && /scratch/venvP/bin/pip install -q numpy scipy astropy matplotlib
/scratch/venvP/bin/python -u $A/scripts/$SCRIPT $ARGS
echo "exit=$? done $(date -u +%FT%TZ)"
