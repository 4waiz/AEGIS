#!/usr/bin/env bash
set -u
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(cd "$(dirname "$0")/.." && pwd)/artifacts}"
cd "$(dirname "$0")/.."
until grep -q ROUND4B_DONE experiments/queue_round4b.log 2>/dev/null; do sleep 30; done
python -W ignore scripts/run_queue.py configs/kd_teacher_L3_mixed.yaml
for e in kd_teacher_L3_mixed kd_teacher_L3_raw; do
  python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/$e/best" --split val --normalize raw --batching official --threshold 0.5 --out experiments/$e/robust_val_vanilla_notebook.json > experiments/$e/robust_val_vanilla_notebook.log 2>&1
done
echo ROUND4C_DONE
