#!/usr/bin/env bash
# Quick bit-identity check (30 fit steps instead of 150, about 2 minutes) of this package against the saved reference state
# tests/oracle/results/ref/reference_minimal_items30_state.pt; run from the package root. The package is copied first (so that it can be edited
# while the run goes on); temporary files go to $TMPDIR (default /tmp).
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/claude/torchsky/src
T=${TMPDIR:-/tmp}
S=$(mktemp -d $T/snap.XXXXXX)
cp -r fieldop $S/fieldop
python -W ignore tests/synth_small.py --pkg $S --sky-form items --seed 0 --steps 30 --gauge-ref-inst 0 --minimal 1 --truth-eval 0 --save-state $S/new30.pt --out $S/new30.json 2>&1 | tail -4
python tests/oracle/compare_states.py tests/oracle/results/ref/reference_minimal_items30_state.pt $S/new30.pt | tail -12
