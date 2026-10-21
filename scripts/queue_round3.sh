#!/usr/bin/env bash
# Round 3: seed repeats, longer schedule, normalization ablation (after round 2).
set -u
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(cd "$(dirname "$0")/.." && pwd)/artifacts}"
cd "$(dirname "$0")/.."
until grep -q ROUND2_DONE experiments/queue_round2.log 2>/dev/null; do sleep 30; done
python -W ignore scripts/run_queue.py configs/kd_teacher_L3_s43.yaml configs/student_unix_L3_s43.yaml configs/kd_teacher_L3_s44.yaml configs/student_unix_L3_s44.yaml configs/kd_teacher_L3_e6.yaml configs/kd_teacher_L3_raw.yaml
echo ROUND3_DONE
