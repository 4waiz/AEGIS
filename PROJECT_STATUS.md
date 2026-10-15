# AEGIS project status

Last updated: 2026-10-15. Deadline: 2026-11-13 AoE (29 days).

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0. Competition intelligence | Done; organizer questions drafted, **not sent** | research/competition_intelligence.md, research/organizer_questions.md |
| 1. Dataset forensics | Done | research/dataset_audit.md, reports/dataset_stats.json, reports/token_budget.csv |
| 2. Baseline reproduction | Done (checkpoint reproduction with the official loop) | experiments/baseline/ |
| 3-6. Architecture, contribution, experiments, score optimization | Done for this cycle | experiments/, reports/leaderboard.md, DECISIONS.md D8-D15 |
| 7. Ablations | Done (depth, distillation with 3 seeds, thresholds, input format, from-scratch) | reports/ablations.md |
| 8. Error analysis | Done (internal val and public test) | reports/error_analysis_kd_teacher_L3*.md |
| 9. Software package | Working: training, measurement, export, inference, tests | src/aegis/, scripts/, tests/ |
| 10. Demo | Not started (optional) | |
| 11. Paper | Full draft; **not compiled yet (no LaTeX toolchain)** | paper/AEGIS.tex, paper/numbers.tex |
| 12. Reviewer audit | Not started | |
| 13. Submission package | Not started; needs T4 run, HF model upload, public repo at submission | |

## Final model (D14)
`kd_teacher_L3_mixed`, exported as `$AEGIS_ARTIFACTS/aegis_final`:
- 3-layer UniXcoder student, initialized from the fine-tuned 12-layer teacher.
- Multitask distillation, trained on a seeded 50/50 mix of raw and single-line inputs.
- Per-task thresholds folded into the output bias, so it works in any 0.5-threshold pipeline.
- Standard `RobertaForSequenceClassification`.

## Public-test results (official measurement loop; all rows except the first measured in one session, RTX 5080 Laptop, Turbo mode)

| Model | Weakness F1 | MAT F1 | Mean F1 | Runtime (s) | GFLOPs | Score |
|---|---|---|---|---|---|---|
| CodeBERT baseline, organizers' printed values (Colab T4) | 0.9390 | 0.8814 | 0.9102 | 198.93 | 2,666,510 | 0.7286 |
| CodeBERT baseline, official loop | 0.9391 | 0.8814 | 0.9102 | 71.02 | 2,666,498 | 0.7542 |
| + length-sorted dynamic padding (same predictions) | 0.9391 | 0.8814 | 0.9102 | 30.51 | 1,171,888 | 0.8619 |
| UniXcoder teacher, 12 layers | 0.9485 | 0.9279 | 0.9382 | 21.53 | 824,370 | 0.9037 |
| 3-layer student, no distillation | 0.9295 | 0.9273 | 0.9284 | 5.99 | 206,123 | 0.9421 |
| 3-layer + KD, single-line only | 0.9359 | 0.9295 | 0.9327 | 5.95 | 206,123 | 0.9447 |
| **AEGIS final** | **0.9323** | **0.9307** | **0.9315** | **6.20** | **206,123** | **0.9439** |
| AEGIS final, unmodified official notebook | 0.9348 | 0.9182 | 0.9265 | 19.03 | 666,661 | 0.9077 |

Runtimes depend on the GPU: this laptop in Turbo mode runs the baseline 2.8x faster than the organizers' T4.
GFLOPs are hardware-independent. A T4 measurement of AEGIS requires running notebooks/AEGIS_T4_evaluation.ipynb
on Colab after the model is hosted.

## Environment
- Windows 11, Python 3.12.10, torch 2.11.0+cu128, transformers 4.57.6.
- ASUS ROG Strix G18, RTX 5080 Laptop (16 GB). Silent mode caps the GPU at ~55 W; Turbo mode draws ~130 W
  (66 TFLOPS fp16 matmul measured). Experiments up to round 4c ran in Silent mode; the public-test table ran in Turbo.
- No LaTeX toolchain installed.
