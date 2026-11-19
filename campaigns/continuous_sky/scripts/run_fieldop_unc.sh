#!/usr/bin/env bash
# Sky uncertainty map (python -m fieldop.uncertainty) on a saved fieldop fit, and optionally the half-difference of two fits.
# env: RUN (finished fit folder under fieldop/), SCRIPT (package tarball in scripts/), TSKY_SRC (torchsky tarball), BANDS (space separated; a comma list or 'all' as one word = one solve for all its bands),
#      UNC_ARGS (extra --unc-* options), TAG (suffix of the output folder, optional),
#      CMD=map (default) | diff : diff needs RUN_B and the maps of both runs already written with the same TAG and bands.
# Output: /arc/projects/ots/adass2026/fieldop/$RUN/unc[_TAG]/{unc.log,sky_<band>_uncertainty.fits,halfdiff_<band>.fits}
set -uo pipefail
A=/arc/projects/ots/adass2026
SRC=$A/fieldop/$RUN; U=$SRC/unc${TAG:+_$TAG}; mkdir -p "$U"
exec > >(tee -a "$U/unc.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) run=$RUN cmd=${CMD:-map} bands=${BANDS:-} args=${UNC_ARGS:-} script=$SCRIPT"; nvidia-smi -L || true
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
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
mkdir -p /scratch/fieldop_pkg && tar -xzf "$A/scripts/${SCRIPT}" -C /scratch/fieldop_pkg
cd /scratch/fieldop_pkg
rc=0
if [ "${CMD:-map}" = map ]; then
  for b in $BANDS; do
    echo "== map $b $(date -u +%FT%TZ)"
    case "$b" in *,*|all) o='{band}';; *) o=$b;; esac   # several bands in one word (comma list or all): one solve, one file per band
    # shellcheck disable=SC2086
    /scratch/venvG/bin/python -u -m fieldop.uncertainty --unc-config "$SRC/metrics.json" --device cuda --unc-band "$b" \
        --unc-out "$U/sky_${o}_uncertainty.fits" ${UNC_ARGS:-} || rc=$?
    echo "== map $b exit=$rc $(date -u +%FT%TZ)"
  done
else
  UB=$A/fieldop/$RUN_B/unc${TAG:+_$TAG}
  for b in $BANDS; do
    echo "== half-difference $b"
    /scratch/venvG/bin/python -u -m fieldop.uncertainty --unc-cmd diff-files --unc-file-a "$U/sky_${b}_uncertainty.fits" \
        --unc-file-b "$UB/sky_${b}_uncertainty.fits" --unc-out "$U/halfdiff_${b}.fits" ${UNC_ARGS:-} || rc=$?
  done
fi
echo "done rc=$rc $(date -u +%FT%TZ)"
exit $rc
