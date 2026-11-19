#!/usr/bin/env bash
# General field operator (fieldop package) run with provenance, protocol-v1 predictions and scoring.
# env: RUN (name), SCRIPT (versioned package tarball in scripts/, default fieldop.tar.gz), TSKY_SRC (versioned torchsky tarball), ARGS (fieldop flags),
#      FIELD (default SMACS), SPLIT (default dither), METHOD (default fieldop), SCORE (1 = score after predicting, default 1)
# Launch (1 GPU), e.g.:
#   canfar create headless images.canfar.net/astroai/base:latest -n fieldopsmoke -c 8 -m 64 -g 1 -e RUN=fieldop_smoke_s0 -e SCRIPT=fieldop-<sha8>.tar.gz \
#     -e TSKY_SRC=torchsky-src-9749c80.tar.gz -e "ARGS=--budget-min 25" -- bash /arc/projects/ots/adass2026/scripts/run_fieldop.sh
# Everything (provenance, log, metrics.json, state) is in /arc/projects/ots/adass2026/fieldop/$RUN; predictions go to
# protocol/v1/pred/$FIELD/$METHOD/$SPLIT/ and scores to protocol/v1/scores/$FIELD/$METHOD/$SPLIT.json; the run folder is mirrored to
# vos:sfabbro/adass2026/fieldop/$RUN on exit.
set -uo pipefail
A=/arc/projects/ots/adass2026
SCRIPT=${SCRIPT:-fieldop.tar.gz}
RUN=${RUN:-fieldop_$(date -u +%Y%m%dT%H%M%SZ)}
FIELD=${FIELD:-SMACS}; SPLIT=${SPLIT:-dither}; METHOD=${METHOD:-fieldop}
OUT=$A/fieldop/$RUN; mkdir -p "$OUT/provenance"
exec > >(tee -a "$OUT/session.log") 2>&1
publish() {
  echo "publishing $(date -u +%FT%TZ)"
  V=/scratch/venvG/bin; [ -x $V/vcp ] || V=$(dirname "$(command -v vcp 2>/dev/null || echo /usr/bin/vcp)")
  $V/vmkdir -p vos:sfabbro/adass2026/fieldop/$RUN 2>/dev/null
  (cd "$OUT" && find . -type d | while read d; do $V/vmkdir -p "vos:sfabbro/adass2026/fieldop/$RUN/${d#./}" 2>/dev/null; done)
  (cd "$OUT" && find . -type f | while read f; do $V/vcp "$f" "vos:sfabbro/adass2026/fieldop/$RUN/${f#./}" || echo "publish failed $f"; done)
  echo "published to vos:sfabbro/adass2026/fieldop/$RUN"
}
trap publish EXIT
echo "start $(date -u +%FT%TZ) host=$(hostname) run=$RUN field=$FIELD split=$SPLIT method=$METHOD"; nvidia-smi -L || true
P=$OUT/provenance
cp "$A/scripts/${SCRIPT}" "$P/"; cp "$0" "$P/" 2>/dev/null || cp "$A/scripts/run_fieldop.sh" "$P/"
printf '%s\n' "SCRIPT=${SCRIPT}" "TSKY_SRC=${TSKY_SRC}" "ARGS=${ARGS:-}" "FIELD=$FIELD" "SPLIT=$SPLIT" "METHOD=$METHOD" > "$P/launch.env"
sha256sum "$A/scripts/${SCRIPT}" "$A/scripts/${TSKY_SRC}" "$A/protocol/v1/code/score.py" "$A/protocol/v1/splits_${FIELD}.json" > "$P/sha256.txt"
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
mkdir -p /scratch/fieldop_pkg && tar -xzf "$A/scripts/${SCRIPT}" -C /scratch/fieldop_pkg
echo "fit start $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
(cd /scratch/fieldop_pkg && /scratch/venvG/bin/python -u -m fieldop --field "$FIELD" --split "$SPLIT" --method-name "$METHOD" --out "$OUT" \
   --protocol-root "$A/protocol/v1" --svo-cache "$A/svo_cache" ${ARGS:-})
rc=$?
echo "fit exit=$rc $(date -u +%FT%TZ)"
if [ "$rc" = 0 ] && [ "${SCORE:-1}" = 1 ]; then
  /scratch/venvG/bin/python -u "$A/protocol/v1/code/score.py" score --field "$FIELD" --method "$METHOD" --split "$SPLIT" 2>&1 | tee "$OUT/score.txt"
  cp "$A/protocol/v1/scores/$FIELD/$METHOD/$SPLIT.json" "$OUT/score_${SPLIT}.json" 2>/dev/null
  cp "$A/protocol/v1/scores/$FIELD/comparison_${SPLIT}.md" "$OUT/" 2>/dev/null
fi
echo "done $(date -u +%FT%TZ)"
exit $rc
