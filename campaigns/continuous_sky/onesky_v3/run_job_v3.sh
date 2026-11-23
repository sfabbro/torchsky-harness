#!/usr/bin/env bash
# A one-sky job on real exposures: the torchsky library path (run_library_fit.py, the default) or a comparator (PY=<script>), with
# provenance, protocol-v1 predictions and scoring. The scripts are COPIED to /scratch at the start and run from there; files on /arc that a
# running job uses must never be overwritten (bash reads its script as it goes): upload a new version under a new name instead.
# Modelled on scripts/run_fieldop.sh. env: RUN (name, also the method name unless METHOD is set), TSKY_SRC (torchsky source tarball in
#      scripts/onesky_v3/, project root at the top of the archive), ARGS (run_library_fit.py flags), FIELD (default SMACS), SPLIT (default
#      dither), SCORE (1 = score after predicting, default 1)
# Launch (1 GPU):
# NOTE: the launcher drops an env value that contains "=", so ARGS must not hold one (write --internal key:value).
#   canfar create headless images.canfar.net/astroai/base:latest -n <name> -c 8 -m 64 -g 1 -e RUN=lib_base_s0 -e TSKY_SRC=torchsky-wt-<id>.tar.gz \
#     -e "ARGS=--budget-min 200" -- bash /arc/projects/ots/adass2026/scripts/onesky_v3/run_library.sh
# Everything is in /arc/projects/ots/adass2026/onesky_v3/$RUN; predictions go to protocol/v1/pred/$FIELD/$METHOD/$SPLIT/ and scores to
# protocol/v1/scores/$FIELD/$METHOD/$SPLIT.json. Nothing is installed outside /scratch.
set -uo pipefail
A=/arc/projects/ots/adass2026
D=$A/scripts/onesky_v3
RUN=${RUN:?}; FIELD=${FIELD:-SMACS}; SPLIT=${SPLIT:-dither}; METHOD=${METHOD:-$RUN}; PY=${PY:-run_library_fit_v5.py}
OUT=$A/onesky_v3/$RUN; mkdir -p "$OUT/provenance"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) run=$RUN field=$FIELD split=$SPLIT method=$METHOD"; nvidia-smi -L || true
P=$OUT/provenance
mkdir -p /scratch/job && cp "$D/$PY" /scratch/job/job.py && cp "$D/$PY" "$0" "$P/"
printf '%s\n' "TSKY_SRC=${TSKY_SRC}" "ARGS=${ARGS:-}" "FIELD=$FIELD" "SPLIT=$SPLIT" "METHOD=$METHOD" "PY=$PY" > "$P/launch.env"
sha256sum "$D/${TSKY_SRC}" "$D/$PY" "$A/protocol/v1/code/score.py" "$A/protocol/v1/splits_${FIELD}.json" > "$P/sha256.txt"
nvidia-smi > "$P/nvidia-smi.txt" 2>&1 || true
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvC
/scratch/venvC/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
/scratch/venvC/bin/pip install -q scikit-build-core nanobind numpy cmake ninja
mkdir -p /scratch/torchsky-src /scratch/wheels && tar -xzf "$D/${TSKY_SRC}" -C /scratch/torchsky-src
/scratch/venvC/bin/pip wheel -q --no-build-isolation --no-deps -w /scratch/wheels /scratch/torchsky-src || exit 4
python3 -m venv /scratch/venvG
/scratch/venvG/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
/scratch/venvG/bin/pip install -q "numpy>=2.0" scipy astropy "torchfits>=1.3.3"
/scratch/venvG/bin/pip install -q --no-deps /scratch/wheels/torchsky-*.whl
/scratch/venvG/bin/pip freeze > "$P/pip-freeze.txt"
/scratch/venvG/bin/python -c "import torch, torchfits, torchsky; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchfits', torchfits.__version__, 'torchsky', torchsky.__version__)" | tee "$P/versions.txt"
echo "fit start $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
(cd /scratch && /scratch/venvG/bin/python -u /scratch/job/job.py --field "$FIELD" --split "$SPLIT" --method-name "$METHOD" --out "$OUT" \
   --protocol-root "$A/protocol/v1" --svo-cache "$A/svo_cache" ${ARGS:-})
rc=$?
echo "fit exit=$rc $(date -u +%FT%TZ)"
if [ "$rc" = 0 ] && [ "${SCORE:-1}" = 1 ]; then
  /scratch/venvG/bin/python -u "$A/protocol/v1/code/score.py" score --field "$FIELD" --method "$METHOD" --split "$SPLIT" 2>&1 | tee "$OUT/score.txt"
  cp "$A/protocol/v1/scores/$FIELD/$METHOD/$SPLIT.json" "$OUT/score_${SPLIT}.json" 2>/dev/null
  # variants written by the same run under <METHOD>_<kind><d> (the latent compression test): each is scored like a method
  for dir in "$A/protocol/v1/pred/$FIELD/${METHOD}"_*; do
    [ -d "$dir/$SPLIT" ] || continue
    m=$(basename "$dir")
    /scratch/venvG/bin/python -u "$A/protocol/v1/code/score.py" score --field "$FIELD" --method "$m" --split "$SPLIT" > "$OUT/score_$m.txt" 2>&1
    cp "$A/protocol/v1/scores/$FIELD/$m/$SPLIT.json" "$OUT/score_${SPLIT}_$m.json" 2>/dev/null && echo "scored variant $m"
  done
fi
echo "done rc=$rc $(date -u +%FT%TZ)"
exit $rc
