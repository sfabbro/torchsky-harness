#!/usr/bin/env bash
# SMACS forward-model run with provenance. env: RUN (name), SCRIPT, ARGS, TSKY_SRC (versioned torchsky tarball)
# Everything (provenance, metrics, cubes, residuals, best/final/resume checkpoints, log) is in
# /arc/projects/ots/adass2026/smacs_four/$RUN and mirrored to vos:sfabbro/adass2026/smacs_four/$RUN on exit.
set -uo pipefail
A=/arc/projects/ots/adass2026
RUN=${RUN:-run_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$A/smacs_four/$RUN; mkdir -p "$OUT/provenance"
exec > >(tee -a "$OUT/session.log") 2>&1
publish() {
  echo "publishing $(date -u +%FT%TZ)"
  V=/scratch/venvG/bin; [ -x $V/vcp ] || V=$(dirname "$(command -v vcp 2>/dev/null || echo /usr/bin/vcp)")
  $V/vmkdir -p vos:sfabbro/adass2026/smacs_four/$RUN 2>/dev/null
  (cd "$OUT" && find . -type d | while read d; do $V/vmkdir -p "vos:sfabbro/adass2026/smacs_four/$RUN/${d#./}" 2>/dev/null; done)
  (cd "$OUT" && find . -type f | while read f; do $V/vcp "$f" "vos:sfabbro/adass2026/smacs_four/$RUN/${f#./}" || echo "publish failed $f"; done)
  echo "published to vos:sfabbro/adass2026/smacs_four/$RUN"
}
trap publish EXIT
echo "start $(date -u +%FT%TZ) host=$(hostname) run=$RUN"; nvidia-smi -L || true
P=$OUT/provenance
cp "$A/scripts/${SCRIPT}" "$P/"; cp "$0" "$P/" 2>/dev/null || cp $A/scripts/smacs_run.sh "$P/"
printf '%s\n' "SCRIPT=${SCRIPT}" "TSKY_SRC=${TSKY_SRC}" "ARGS=${ARGS}" > "$P/launch.env"
sha256sum "$A/scripts/${SCRIPT}" "$A/scripts/${TSKY_SRC}" > "$P/sha256.txt"
nvidia-smi > "$P/nvidia-smi.txt" 2>&1 || true
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvC
/scratch/venvC/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
/scratch/venvC/bin/pip install -q scikit-build-core nanobind numpy cmake ninja
mkdir -p /scratch/torchsky-src /scratch/wheels && tar -xzf "$A/scripts/${TSKY_SRC}" -C /scratch/torchsky-src
/scratch/venvC/bin/pip wheel -q --no-build-isolation --no-deps -w /scratch/wheels /scratch/torchsky-src || exit 4
python3 -m venv /scratch/venvG
/scratch/venvG/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
/scratch/venvG/bin/pip install -q numpy scipy astropy astroquery matplotlib "torchfits==1.3.1" vos
/scratch/venvG/bin/pip install -q --no-deps /scratch/wheels/torchsky-*.whl
/scratch/venvG/bin/pip freeze > "$P/pip-freeze.txt"
/scratch/venvG/bin/python -c "import torch, torchfits, torchsky; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchfits', torchfits.__version__, 'torchsky', torchsky.__version__)" | tee "$P/versions.txt"
/scratch/venvG/bin/python -u "$A/scripts/${SCRIPT}" --out "$OUT" --stripes 1 --svo-cache $A/svo_cache ${ARGS}
echo "fit exit=$?"
echo "done $(date -u +%FT%TZ)"
