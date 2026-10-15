#!/usr/bin/env bash
# Round 5b: clean, sequential public-test measurements (idle GPU) for every reported row (D15).
set -u
export AEGIS_ARTIFACTS="C:/Users/awaiz/aegis_artifacts"
cd "$(dirname "$0")/.."
M="python -W ignore scripts/measure.py"
B=NLBSE/nlbse27-code-comment-classification-baseline
$M $B --split test --batching official --threshold 0.5 --out experiments/baseline/measure_test_official.json > experiments/baseline/measure_test_official.log 2>&1; echo "done baseline official"
$M $B --split test --batching sorted --threshold 0.5 --out experiments/baseline/measure_test_sorted.json > experiments/baseline/measure_test_sorted.log 2>&1; echo "done baseline sorted"
for e in teacher_unixcoder student_unix_L3 kd_teacher_L3; do
  $M "$AEGIS_ARTIFACTS/$e/best" --split test --out experiments/$e/measure_test.json > experiments/$e/measure_test.log 2>&1; echo "done $e"
done
# Final model: exported checkpoint with folded thresholds
$M "$AEGIS_ARTIFACTS/aegis_final" --split test --out experiments/kd_teacher_L3_mixed/measure_test.json > experiments/kd_teacher_L3_mixed/measure_test.log 2>&1; echo "done final (AEGIS pipeline)"
$M "$AEGIS_ARTIFACTS/aegis_final" --split test --normalize raw --batching official --threshold 0.5 --out experiments/kd_teacher_L3_mixed/robust_test_vanilla_notebook.json > experiments/kd_teacher_L3_mixed/robust_test_vanilla_notebook.log 2>&1; echo "done final (vanilla notebook)"
$M "$AEGIS_ARTIFACTS/kd_teacher_L3/best" --split test --normalize raw --batching official --threshold 0.5 --out experiments/kd_teacher_L3/robust_test_vanilla_notebook.json > experiments/kd_teacher_L3/robust_test_vanilla_notebook.log 2>&1; echo "done kd_teacher_L3 (vanilla notebook)"
echo ROUND5B_DONE
