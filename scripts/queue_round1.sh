#!/usr/bin/env bash
# Round 1: wait for the teacher, then measure + student sweep (sequential GPU use).
set -u
export AEGIS_ARTIFACTS="C:/Users/awaiz/aegis_artifacts"
cd "$(dirname "$0")/.."
while [ ! -f experiments/teacher_unixcoder/metrics.json ]; do sleep 30; done
python -W ignore scripts/measure.py "$AEGIS_ARTIFACTS/teacher_unixcoder/best" --split val --out experiments/teacher_unixcoder/measure_val.json > experiments/teacher_unixcoder/measure_val.log 2>&1
python -W ignore scripts/measure.py NLBSE/nlbse27-code-comment-classification-baseline --split test --batching official --threshold 0.5 --out experiments/baseline/measure_test_official.json > experiments/baseline/measure_test_official.log 2>&1
python -W ignore scripts/measure.py NLBSE/nlbse27-code-comment-classification-baseline --split test --batching sorted --threshold 0.5 --out experiments/baseline/measure_test_sorted.json > experiments/baseline/measure_test_sorted.log 2>&1
python -W ignore scripts/run_queue.py configs/student_unix_L2.yaml configs/student_unix_L4.yaml configs/student_unix_L1.yaml configs/student_unix_L6.yaml configs/student_unix_L3.yaml configs/student_scratch_4x256.yaml
echo ROUND1_DONE
