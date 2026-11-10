#!/usr/bin/env bash
# Continue an opjoint training from another run's final checkpoint into a NEW run directory (so the first run's evaluation outputs stay intact).
# env: SRC_RUN (finished run whose ckpt.pt to copy), RUN (new name), ARGS (same as SRC_RUN with a larger --budget-min), SCRIPT, TSKY_SRC.
# Approximation: opjoint's learning-rate schedule is cosine in progress = elapsed / budget-min; raising the budget lowers the progress, so the learning rate
# jumps back up (a warm restart, SGDR-like) instead of continuing from its end value. Coarse-to-fine and sky warm-up stay finished (progress >= their fractions).
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=$A/smacs_four/$RUN; mkdir -p "$OUT"
if [ ! -f "$OUT/ckpt.pt" ]; then cp "$A/smacs_four/$SRC_RUN/ckpt.pt" "$OUT/ckpt.pt" && echo "copied ckpt.pt from $SRC_RUN" >> "$OUT/continue.log"; fi
exec bash $A/scripts/run_opjoint.sh
