# Reproducing the AEGIS results

All commands run from the repository root. Times are wall-clock on one RTX 5080 Laptop GPU in its
low-power ("Silent") mode, as observed for this work; a desktop GPU or Turbo mode is about 2-3x faster.

## 0. Setup (5 min)

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.txt
python -m pip install -e .
export AEGIS_ARTIFACTS=$PWD/artifacts      # where checkpoints go (git-ignored)
python -m pytest -q                         # 10 tests
```

## 1. Quick check: score the released model (2 min)

```bash
python scripts/measure.py 4waiz/aegis-nlbse27 --split test --out experiments/reproduce_hub/measure_test.json
```

Expected (public test, official measurement loop): weakness F1 0.9323, MAT F1 0.9307, mean F1 0.9315,
206,123 GFLOPs. F1 and GFLOPs are deterministic; runtime and score depend on the GPU.

## 2. Data and audit (10 min)

```bash
python -c "from aegis.data.dataset import download; download()"   # pinned revision feca19b
python scripts/dataset_audit.py      # reports/dataset_stats.json, data_quality.md, figures/dataset_overview.*
python scripts/token_budget.py       # reports/token_budget.csv
```

## 3. Official baseline (15 min)

```bash
python scripts/official_notebook_replica.py NLBSE/nlbse27-code-comment-classification-baseline experiments/baseline/official_replica.json
python scripts/measure.py NLBSE/nlbse27-code-comment-classification-baseline --split test --batching official --threshold 0.5 --out experiments/baseline/measure_test_official.json
python scripts/measure.py NLBSE/nlbse27-code-comment-classification-baseline --split test --batching sorted --threshold 0.5 --out experiments/baseline/measure_test_sorted.json
```

## 4. Teacher (55 min)

```bash
python scripts/train.py configs/teacher_unixcoder.yaml
python scripts/train.py --dump-teacher $AEGIS_ARTIFACTS/teacher_unixcoder/best configs/teacher_unixcoder.yaml $AEGIS_ARTIFACTS/teacher_unixcoder/teacher_logits_fit.npz
```

## 5. Final AEGIS model (20 min) and export

```bash
python scripts/run_queue.py configs/kd_teacher_L3_mixed.yaml          # trains + scores on internal validation
python -m aegis.optimization.export $AEGIS_ARTIFACTS/kd_teacher_L3_mixed/best $AEGIS_ARTIFACTS/aegis_final
python scripts/measure.py $AEGIS_ARTIFACTS/aegis_final --split test --out experiments/kd_teacher_L3_mixed/measure_test.json
python scripts/measure.py $AEGIS_ARTIFACTS/aegis_final --split test --normalize raw --batching official --threshold 0.5 --out experiments/kd_teacher_L3_mixed/robust_test_vanilla_notebook.json
```

Training is seeded (seed 42), but GPU kernels are not bitwise deterministic, so retrained models can
differ slightly (the three seeds of the single-line variant agree within 0.0003 mean F1 on internal validation).

## 6. Ablations (about 5 h for all)

Each config in `configs/` corresponds to one directory in `experiments/` (config, training log, metrics,
official-loop measurement). Run any subset with `python scripts/run_queue.py configs/<a>.yaml configs/<b>.yaml ...`.
The order used for the paper is in `scripts/queue_round1.sh` ... `queue_round5b.sh`. The distillation configs
(`kd_*.yaml`) need the teacher and its logits from step 4.

| Paper item | Configs |
|---|---|
| Depth sweep (Table II, Fig. 1) | `student_unix_L{1,2,3,4,6}.yaml`, `teacher_unixcoder.yaml` |
| Seeds | `student_unix_L3_s{43,44}.yaml`, `kd_teacher_L3_s{43,44}.yaml` |
| Distillation | `kd_teacher_L{2,3,4}.yaml`, `kd_pre_L2.yaml` |
| Variants | `kd_teacher_L3_e6.yaml`, `kd_teacher_L3_raw.yaml`, `kd_teacher_L3_raw_s43.yaml`, `kd_teacher_L3_collapse.yaml`, `kd_teacher_L3_mixed.yaml` |
| From scratch | `student_scratch_4x256.yaml` |
| Lexical reference | `python scripts/linear_reference.py` |

## 7. Tables, figure, numbers, paper (2 min)

```bash
python scripts/leaderboard.py && python scripts/ablation_table.py && python scripts/paper_numbers.py && python scripts/make_figures.py
python scripts/error_analysis.py kd_teacher_L3_mixed --test experiments/kd_teacher_L3_mixed/measure_test.probs.npy --threshold 0.5
cd paper && tectonic AEGIS.tex       # or ../tools/tectonic/tectonic.exe AEGIS.tex
```

Every number in the paper is a macro in `paper/numbers.tex`, generated from the files above.

## 8. Colab T4 (official hardware)

Open `notebooks/AEGIS_T4_evaluation.ipynb` in Colab with a T4 runtime and run all cells. It scores the official
baseline and AEGIS with the official measurement loop in the same session.
