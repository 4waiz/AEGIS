# AEGIS project status

Last updated: 2026-10-15. Deadline: 2026-11-13 AoE (29 days).

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0. Competition intelligence | Done (organizer questions drafted, not sent) | research/competition_intelligence.md, research/organizer_questions.md |
| 1. Dataset forensics | Core audit done; figures generated | reports/dataset_stats.json, reports/data_quality.md, reports/label_distribution.csv, figures/dataset_overview.pdf |
| 2. Baseline reproduction | Checkpoint reproduction done (official loop, local GPU) | experiments/baseline/official_replica_rtx5080.json |
| 3. Architecture research | Teacher done; student depth sweep + distillation queued (scripts/queue_round1.sh, queue_round2.sh) | experiments/ |
| 4-13 | Not started | |

## Verified results so far (public test split, official measurement loop)

| Model | Weakness F1 | MAT F1 | Mean F1 | Runtime (s) | GFLOPs | Score | Hardware | Comparable? |
|---|---|---|---|---|---|---|---|---|
| Official CodeBERT baseline (organizers) | 0.9390 | 0.8814 | 0.9102 | 198.93 | 2,666,510 | 0.7286 (displayed 0.71) | Colab T4 | reference |
| Official baseline, AEGIS replica | 0.9391 | 0.8814 | 0.9102 | 193.90 | 2,666,498 | 0.7296 | RTX 5080 Laptop @ ~54 W | yes (same loop; different GPU) |
| Official baseline, AEGIS module, official batching | 0.9391 | 0.8814 | 0.9102 | 189.69 | 2,666,498 | 0.7304 | RTX 5080 Laptop @ ~54 W | yes (validates the module) |
| Official baseline checkpoint + length-sorted dynamic padding (inference-only change) | 0.9391 | 0.8814 | 0.9102 | 80.84 | 1,171,888 | 0.8518 | RTX 5080 Laptop @ ~54 W | yes |
| TF-IDF + LR (lexical reference, not a submission) | 0.8311 | 0.8782 | 0.8546 | n/a | n/a | n/a | CPU | F1 only |

## Internal validation (21,243 functions; model selection only)

| Model | Weakness F1 | MAT F1 | Mean F1 (tuned thr.) | Runtime (s) | GFLOPs | Train time |
|---|---|---|---|---|---|---|
| Teacher: UniXcoder-base 12 layers, single-line input, 3 epochs | 0.9479 | 0.9313 | 0.9396 | 38.43 | 580,442 | 55 min |

## Environment
- Windows 11, Python 3.12.10, torch 2.11.0+cu128, transformers 4.57.6.
- GPU: RTX 5080 Laptop (16 GB, sm_120). **Power-limited (~54 W of 175 W, "PD Silent" scheme)**:
  runtime numbers are only comparable within the same power mode; each run records the power state.
- CPU: Intel Core Ultra 9 275HX (24 cores), 31 GB RAM, ~217 GB free disk.
- No LaTeX toolchain installed (needed for the paper; options in NEXT_STEPS.md).
