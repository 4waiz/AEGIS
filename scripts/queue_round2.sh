#!/usr/bin/env bash
# Round 2: after round 1, dump teacher logits on the fit split, then multitask distillation runs.
set -u
export AEGIS_ARTIFACTS="${AEGIS_ARTIFACTS:-$(cd "$(dirname "$0")/.." && pwd)/artifacts}"
cd "$(dirname "$0")/.."
until grep -q ROUND1_DONE experiments/queue_round1.log 2>/dev/null; do sleep 30; done
python -W ignore scripts/train.py --dump-teacher "$AEGIS_ARTIFACTS/teacher_unixcoder/best" configs/teacher_unixcoder.yaml "$AEGIS_ARTIFACTS/teacher_unixcoder/teacher_logits_fit.npz" > experiments/teacher_unixcoder/dump_logits.log 2>&1
python -W ignore scripts/run_queue.py configs/kd_teacher_L2.yaml configs/kd_pre_L2.yaml configs/kd_teacher_L3.yaml configs/kd_teacher_L4.yaml
echo ROUND2_DONE
