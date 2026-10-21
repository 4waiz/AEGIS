#!/usr/bin/env bash
set -u
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(cd "$(dirname "$0")/.." && pwd)/artifacts}"
cd "$(dirname "$0")/.."
until grep -q ROUND4_DONE experiments/queue_round4.log 2>/dev/null; do sleep 30; done
python -W ignore scripts/run_queue.py configs/kd_teacher_L3_collapse.yaml configs/kd_teacher_L3_raw_s43.yaml
echo ROUND4B_DONE
