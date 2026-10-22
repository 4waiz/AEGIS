#!/usr/bin/env bash
# Round 7 (organizers' reply of 2026-10-22): the final evaluation averages 10 forward passes per batch instead
# of the released notebook's 2. Measure AEGIS and the baseline on the public test split both ways, in one
# session (one power mode), to see how much the per-pass averages move.
set -u
cd "$(dirname "$0")/.."
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(pwd)/artifacts}"
out=experiments/repetitions_check
mkdir -p "$out"
for reps in 2 10; do
  python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/aegis_final" --repetitions "$reps" --out "$out/aegis_r$reps.json" > "$out/aegis_r$reps.log" 2>&1; echo "done aegis r$reps"
  python -W ignore scripts/measure.py NLBSE/nlbse27-code-comment-classification-baseline --batching official --repetitions "$reps" --out "$out/baseline_r$reps.json" > "$out/baseline_r$reps.log" 2>&1; echo "done baseline r$reps"
done
echo ROUND7_DONE
