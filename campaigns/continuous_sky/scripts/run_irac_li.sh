#!/usr/bin/env bash
# Learned IRAC instrument track (6 Nov 2026): frozen NIRCam LW sky -> learned vs structured IRAC operator, all frames.
set -uo pipefail
S=/arc/projects/ots/adass2026/scripts; U=/arc/projects/ots/torchsky-data/smacs0723_union
OUT=/arc/projects/ots/adass2026/learned_instrument/${RUN:-irac_li_v1}; mkdir -p $OUT; exec > >(tee -a $OUT/session.log) 2>&1
echo "start $(date -u +%FT%TZ) $(hostname)"; nvidia-smi -L || true
python3 -m venv /scratch/v && . /scratch/v/bin/activate && pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126 && pip install -q numpy scipy astropy matplotlib
cp $S/learned_instrument.py $S/jwst_lw_sky.py $S/run_irac_li.sh $OUT/ 2>/dev/null
SKY=$OUT/sky; mkdir -p $SKY
if [ ! -s $SKY/jwst_lw_sky.fits ]; then
  python3 $S/jwst_lw_sky.py --out $SKY --input /arc/projects/ots/torchsky-data/jwst_smacs0723/cal --input $U/jwst/2736_nircam --input $U/jwst/4043_nircam --input $U/jwst/6882_nircam --input $U/jwst/12774_nircam --scale 0.12 --method splat || exit 3
fi
python3 $S/learned_instrument.py --sky $SKY/jwst_lw_sky.fits --irac-root $U/spitzer --out $OUT --model both --device cuda --budget-min ${BUDGET:-150} ${EXTRA:-}
echo "exit=$? done $(date -u +%FT%TZ)"
