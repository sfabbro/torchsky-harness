#!/usr/bin/env bash
# The oracle runs of MINIMAL_PRESET_2026-11-18.md, from the package root. One core each (the scripts set one thread); about 10 minutes each on a shared machine.
#   bash tests/oracle/run_oracles.sh hash    # (a) the 150-step synthetic item run with --minimal, twice
#   bash tests/oracle/run_oracles.sh truth   # (b) the truth problem, 8 exposures, seeds 0 1 2: response frozen at the truth (the harness's standard), then learned
#   bash tests/oracle/run_oracles.sh golden  # (c) the golden-step files, then their check
#   bash tests/oracle/run_oracles.sh default # proof that the default 150-step item run is unchanged (compare with default_items150_before.json)
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
R=tests/oracle/results
case "${1:-}" in
  hash)
    for run in 1 2; do
      python -W ignore tests/synth_small.py --pkg . --sky-form items --seed 0 --steps 150 --gauge-ref-inst 0 --minimal 1 --truth-eval 0 --out $R/minimal_items150_run$run.json
    done ;;
  truth)
    for s in 0 1 2; do
      python -W ignore tests/truth/run_truth.py fit --form items --n-train 8 --seed $s --steps 300 --minimal 1 --out $R/truth_minimal_frozen_s$s.json
    done
    for s in 0 1 2; do
      python -W ignore tests/truth/run_truth.py fit --form items --n-train 8 --seed $s --steps 300 --minimal 1 --freeze-kernel 0 --blur-floor 1.0 --out $R/truth_minimal_learned_s$s.json
    done ;;
  golden)
    python -W ignore tests/oracle/golden_step.py
    python -W ignore tests/oracle/golden_step.py --check ;;
  default)
    python -W ignore tests/synth_small.py --pkg . --sky-form items --seed 0 --steps 150 --gauge-ref-inst 0 --kernel-norm-radius 2.0 --truth-eval 0 --out $R/default_items150_after.json ;;
  *) echo "usage: $0 hash|truth|golden|default"; exit 2 ;;
esac
