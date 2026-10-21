#!/usr/bin/env bash
# Round 5: one-time public-test scoring of the final model and the reported comparisons (D12).
set -u
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(cd "$(dirname "$0")/.." && pwd)/artifacts}"
cd "$(dirname "$0")/.."
until grep -q ROUND4_DONE experiments/queue_round4.log 2>/dev/null; do sleep 30; done
for e in kd_teacher_L3 student_unix_L3 teacher_unixcoder; do
  python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/$e/best" --split test --out experiments/$e/measure_test.json > experiments/$e/measure_test.log 2>&1
  echo "done $e"
done
# Final model through the unmodified official procedure (raw input, global padding, threshold 0.5)
python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/kd_teacher_L3/best" --split test --normalize raw --batching official --threshold 0.5 --out experiments/kd_teacher_L3/robust_test_vanilla_notebook.json > experiments/kd_teacher_L3/robust_test_vanilla_notebook.log 2>&1
echo ROUND5_DONE
