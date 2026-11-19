#!/usr/bin/env bash
# The oracle runs of the pruning (PRUNE_2026-11-19.md), from the package root. One core each (the scripts set one thread).
#   bash tests/oracle/run_oracles.sh hash     # rule A: the 150-step synthetic item run, then compare_states.py against the reference run's final state (about 4 minutes)
#   bash tests/oracle/run_oracles.sh quick    # the same with 30 steps (about 2 minutes)
#   bash tests/oracle/run_oracles.sh golden   # rule B: reload the three golden-step files and compare the loss and every surviving gradient (seconds)
#   bash tests/oracle/run_oracles.sh truth    # the truth problem, 8 exposures, seeds 0 1 2: response frozen at the truth, then learned (long; the numbers in results/ are the unpruned package's)
# The reference states (ref/reference_minimal_items{30,150}_state.pt) were made once by the unpruned package with its --minimal 1 run; they are not remade here.
# In tests/synth_small.py the flag `--minimal 1` is only the protocol of that reference run: no validation exposure (the pruned package has none), and the script's own
# tau_mode / pattern_grid / stripes options dropped, so the package's defaults hold.
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${PYTHONPATH:-/home/claude/torchsky/src}
R=tests/oracle/results
T=${TMPDIR:-/tmp}
case "${1:-}" in
  hash|quick)
    n=150; [ "$1" = quick ] && n=30
    python -W ignore tests/synth_small.py --pkg . --sky-form items --seed 0 --steps $n --gauge-ref-inst 0 --minimal 1 --truth-eval 0 --save-state $T/new_state_$n.pt --out $T/new_items$n.json
    python tests/oracle/compare_states.py $R/ref/reference_minimal_items${n}_state.pt $T/new_state_$n.pt ;;
  truth)
    for s in 0 1 2; do
      python -W ignore tests/truth/run_truth.py fit --form items --n-train 8 --seed $s --steps 300 --out $T/truth_frozen_s$s.json
    done
    for s in 0 1 2; do
      python -W ignore tests/truth/run_truth.py fit --form items --n-train 8 --seed $s --steps 300 --freeze-kernel 0 --blur-floor 1.0 --out $T/truth_learned_s$s.json
    done ;;
  golden)
    python -W ignore tests/oracle/golden_step.py --check ;;
  *) echo "usage: $0 hash|quick|golden|truth"; exit 2 ;;
esac
