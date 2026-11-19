#!/bin/bash
# Feature isolation on the reproducer: each configuration with the new ('filter') preconditioner, then the old ('fourier') one at the same mu.
# Usage: bash tests/solver_repro/run_isolation.sh <stream: 0|1>
cd "$(dirname "$0")/../.."
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src
R="python tests/solver_repro/run_repro.py --draw 0 --checks 0 --maxiter 100"
declare -A C
C[six_plain]="--gap 0 --bright 0 --floor 0 0 --huber 0"
C[six_gap]="--gap 1 --bright 0 --floor 0 0 --huber 0"
C[six_gapc]="--gap 1 --gap-centre-filter F090W --bright 0 --floor 0 0 --huber 0"
C[six_bright]="--gap 0 --bright 3000 --huber 0"
C[six_bright_huber]="--gap 0 --bright 3000 --huber 1"
C[one_all]="--filters one"
C[onechan_all]="--two-channels 0"
C[items_all]="--sky-form items"
C[all]=""
if [ "$1" = "0" ]; then L="six_plain six_gapc six_bright_huber one_all all"; else L="six_gap six_bright onechan_all items_all"; fi
for n in $L; do
  $R --name iso_${n}_filter --precond filter ${C[$n]} > tests/solver_repro/logs/iso_${n}_filter.log 2>&1
  $R --name iso_${n}_old --precond fourier --mu-from iso_${n}_filter ${C[$n]} > tests/solver_repro/logs/iso_${n}_old.log 2>&1
done
