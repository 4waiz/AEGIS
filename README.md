# AEGIS

**An Efficiency-Guided Inference System** for the
[NLBSE'27 Multitask Code Classification tool competition](https://nlbse2027.github.io/tools/):
one model that reads a C/C++ function and jointly predicts
`[weakness/vulnerability, MAT (self-admitted technical debt)]`.

> Status: research in progress (see [PROJECT_STATUS.md](PROJECT_STATUS.md)). All numbers in this
> repository are produced by the scripts here; nothing is copied from elsewhere unless cited.

## Why efficiency matters here

The official ranking score is

```
score = 0.6 * meanF1 + 0.2 * max((1000 - runtime_s) / 1000, 0) + 0.2 * max((3e6 - GFLOPs) / 3e6, 0)
```

measured on a Colab T4 with `torch.profiler` over the whole test split. One point of mean F1 (0.01)
is worth 0.006 score, as much as 30 s of runtime or 90,000 GFLOPs.

## Results (public test split, official measurement loop)

| Model | Weakness F1 | MAT F1 | Mean F1 | Runtime (s)* | GFLOPs | Score* |
|---|---|---|---|---|---|---|
| CodeBERT baseline (official procedure) | 0.939 | 0.881 | 0.910 | 71.0 | 2,666,498 | 0.754 |
| **AEGIS** (3-layer distilled UniXcoder student) | 0.932 | 0.931 | **0.931** | **6.2** | **206,123** | **0.944** |

\*Measured on one RTX 5080 Laptop GPU in the same session (the organizers' T4 is slower; the official
baseline score on T4 is 0.729). Full table, ablations and error analysis: [PROJECT_STATUS.md](PROJECT_STATUS.md),
[reports/ablations.md](reports/ablations.md), [DECISIONS.md](DECISIONS.md).

## Setup

```bash
python -m pip install -e .[dev]
python -c "from aegis.data.dataset import download; download()"   # pinned dataset revision
python -m pytest -q
```

Large checkpoints are written to `$AEGIS_ARTIFACTS` (default `./artifacts`, git-ignored).

## Reproducing results

| Step | Command |
|---|---|
| Dataset audit (reports/, figures/) | `python scripts/dataset_audit.py` |
| Official baseline, official notebook loop | `python scripts/official_notebook_replica.py NLBSE/nlbse27-code-comment-classification-baseline experiments/baseline/official_replica.json` |
| Train a model | `python scripts/train.py configs/<config>.yaml` |
| Score a checkpoint with the official protocol | `python scripts/measure.py $AEGIS_ARTIFACTS/<name>/best --split test` |
| Lexical reference (TF-IDF + LR) | `python scripts/linear_reference.py` |
| Export with thresholds folded into the output bias | `python -m aegis.optimization.export $AEGIS_ARTIFACTS/<name>/best $AEGIS_ARTIFACTS/aegis_final` |
| Regenerate tables, figure and paper numbers | `python scripts/leaderboard.py && python scripts/ablation_table.py && python scripts/paper_numbers.py && python scripts/make_figures.py` |
| Error analysis | `python scripts/error_analysis.py <experiment> [--test probs.npy --threshold 0.5]` |

## Data usage rules followed

- Only the `Function` text is used as model input (competition rule; `Projectname`/`Filepath`
  are used for analysis only).
- Model selection and threshold tuning use an internal validation split carved from the official
  training split (`aegis.data.dataset.internal_split`). The public test split is used for final
  reporting only. The hidden test split is never accessed.

## Repository layout

```
src/aegis/      data loading, preprocessing, models, training, official evaluation
scripts/        command-line entry points
configs/        experiment configurations (YAML)
experiments/    per-run configs, logs and metrics (small files only)
reports/        dataset audit outputs, leaderboard
research/       competition intelligence, related work, organizer questions
figures/        generated figures
paper/          LaTeX sources of the competition paper
```

## License

Code: MIT. The competition dataset and baseline are distributed by the NLBSE'27 organizers
(repository under CC0); pretrained encoders keep their own licenses.
