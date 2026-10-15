#!/usr/bin/env bash
# Round 4: robustness of the lead student to the *unmodified* official notebook procedure
# (raw input, global padding to 512, threshold 0.5), on internal validation.
set -u
export AEGIS_ARTIFACTS="C:/Users/awaiz/aegis_artifacts"
cd "$(dirname "$0")/.."
until grep -q ROUND3_DONE experiments/queue_round3.log 2>/dev/null; do sleep 30; done
python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/kd_teacher_L3/best" --split val --normalize raw --batching official --threshold 0.5 --out experiments/kd_teacher_L3/robust_val_vanilla_notebook.json > experiments/kd_teacher_L3/robust_val_vanilla_notebook.log 2>&1
python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/kd_teacher_L3/best" --split val --normalize raw --batching sorted --out experiments/kd_teacher_L3/robust_val_raw_sorted.json > experiments/kd_teacher_L3/robust_val_raw_sorted.log 2>&1
echo ROUND4_DONE
