#!/bin/bash
# sequential synthetic runs of tests/synth_gain.py (each < 4 min), 200 steps
cd /home/claude/work/fieldop_gain
run() { name=$1; shift; timeout 260 python3 -W ignore tests/synth_gain.py --steps 200 --out tests/gain_results/$name.json "$@" > tests/gain_results/$name.log 2>&1; echo "$(date +%T) $name rc=$?"; }
run r0_noflat_off   --inject-rms 0    --pixel-gain off
run r1_flat_off     --inject-rms 0.03 --pixel-gain off
run r2_flat_on_1e-3 --inject-rms 0.03 --pixel-gain detector --lr-gain 1e-3
run r3_flat_on_3e-4 --inject-rms 0.03 --pixel-gain detector --lr-gain 3e-4
run r4_flat_on_3e-3 --inject-rms 0.03 --pixel-gain detector --lr-gain 3e-3
run r5_noflat_on_1e-3 --inject-rms 0  --pixel-gain detector --lr-gain 1e-3
echo ALLDONE
