#!/usr/bin/env bash
# GPU timing of the item rasteriser. env: SCRIPT (package tarball in scripts/), RUN (output folder name under onesky_v2/gpu_bench/).
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=$A/onesky_v2/gpu_bench/${RUN:?}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ)"; nvidia-smi -L || true
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvG
/scratch/venvG/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
/scratch/venvG/bin/pip install -q numpy
mkdir -p /scratch/pkg && tar -xzf "$A/scripts/${SCRIPT:?}" -C /scratch/pkg
/scratch/venvG/bin/python -u $A/scripts/onesky/${BENCH:-gpu_bench_items.py} /scratch/pkg "$OUT/bench.json"
echo "exit=$? done $(date -u +%FT%TZ)"
