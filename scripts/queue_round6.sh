#!/usr/bin/env bash
# Round 6 (reviewer audit): distillation-vs-initialization control, seed repeats of the final configuration,
# and the official notebook replica on the exported final model.
set -u
cd "$(dirname "$0")/.."
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(pwd)/artifacts}"
python -W ignore scripts/official_notebook_replica.py "$AEGIS_ARTIFACTS/aegis_final" experiments/kd_teacher_L3_mixed/official_replica_test.json > experiments/kd_teacher_L3_mixed/official_replica_test.log 2>&1; echo "done replica"
python -W ignore scripts/run_queue.py configs/student_tinit_L3.yaml configs/student_tinit_L3_s43.yaml configs/student_tinit_L3_s44.yaml configs/kd_teacher_L3_mixed_s43.yaml configs/kd_teacher_L3_mixed_s44.yaml
echo ROUND6_DONE
