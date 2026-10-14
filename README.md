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
