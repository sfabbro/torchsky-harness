#!/usr/bin/env bash
# Diagnostic: run stage3_blot.py with several tweakreg settings in parallel (one process each). env: FIELD BANDS TILES, VARIANTS="name|ARGS;name2|ARGS2"
# outputs go to $A/baselines/stage3_blot_variants/<name>/ (own outroot and pred root); never touches the protocol pred directory.
set -uo pipefail
A=/arc/projects/ots/adass2026
LOG=$A/logs/stage3_variants_${TAG:-v}.log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -u +%FT%TZ) nproc=$(nproc)"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu CRDS_PATH=$A/crds_cache
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvS && /scratch/venvS/bin/pip install -q "${JWST_SPEC:-jwst}" || { echo "pip failed"; exit 4; }
IFS=';' read -ra VS <<< "$VARIANTS"
pids=()
for v in "${VS[@]}"; do
  name=${v%%|*}; args=${v#*|}
  O=$A/baselines/stage3_blot_variants/$name
  mkdir -p "$O"
  FL="--field ${FIELD:?} --split dither --procs 1 --outroot $O --pred-root $O/pred --scratch /scratch/s3_$name"
  [ -n "${BANDS:-}" ] && FL="$FL --bands $BANDS"
  [ -n "${TILES:-}" ] && FL="$FL --tiles $TILES"
  echo "variant $name: $FL $args"
  /scratch/venvS/bin/python -u "$A/scripts/stage3_blot.py" $FL $args > "$O/run.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "exit=$rc done $(date -u +%FT%TZ)"
exit $rc
