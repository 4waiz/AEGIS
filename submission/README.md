# AEGIS: NLBSE'27 submission package

Everything needed to submit the AEGIS entry to the NLBSE'27 multitask code classification tool competition.
Deadline: **2026-11-13 AoE**.

## Contents

| Item | Location | Status |
|---|---|---|
| Paper (2-4 pages, ICSE'27 IEEEtran, not anonymized) | `submission/AEGIS.pdf` (built from `paper/AEGIS.tex`) | compiled, 4 pages |
| Trained model | https://huggingface.co/4waiz/aegis-nlbse27 (revision `2011b2f`, hashes in `submission/model_manifest.json`) | hosted, **public** since 2026-10-21 |
| Source code (training, tuning, evaluation) | https://github.com/4waiz/AEGIS | complete, **private** |
| Reproduction guide | `submission/REPRODUCE.md` | written and checked |
| Official-loop evaluation on a Colab T4 | `notebooks/AEGIS_T4_evaluation.ipynb`; results in `experiments/colab_t4/` | done 2026-10-21: AEGIS 16.4 s, score 0.942 (baseline 192.7 s, 0.730) |
| Submission-form answers | `submission/form_answers.md` | drafted; author details needed |
| Questions for the organizers | `research/organizer_questions.md` | drafted, not sent |

## Checklist before pressing "Submit"

- [ ] Author affiliation and e-mail filled in `paper/AEGIS.tex` and in `form_answers.md`.
- [ ] AI-use disclosure in the paper's acknowledgment confirmed by the author (IEEE policy).
- [ ] Confirm with the organizers which form to use (the linked form is still the NLBSE'26 one).
- [ ] GitHub repository made public (`gh repo edit 4waiz/AEGIS --visibility public --accept-visibility-change-consequences`).
- [x] Hugging Face model made public (2026-10-21).
- [x] T4 numbers added to the paper (Table I, abstract).
- [x] `submission/AEGIS.pdf` re-built after the last edit (2026-10-21).

## Competition requirements and where they are met

| Requirement (nlbse2027.github.io/tools/, instructions notebook) | Where |
|---|---|
| One multitask model predicting `[weakness, MAT]` | 3-layer `RobertaForSequenceClassification`, 2 sigmoid outputs (paper Sec. III) |
| Only the `Function` text as input; no external data | `src/aegis/data/dataset.py` docstring; DECISIONS.md D2 |
| Pretrained model fine-tuned on the provided training data | UniXcoder-base, fine-tuned and distilled on the training split only |
| Architecture and preprocessing described | paper Sec. III |
| Tuning procedure on the training set | paper Sec. IV (internal 10% validation split of the training data) |
| Results on the validation (public test) split | paper Table I |
| Link to documented code that reproduces the results | README.md, submission/REPRODUCE.md |
| ICSE'27 format, 2-4 pages, need not be double-blind | IEEEtran 10pt conference, 4 pages, author named |
| Cite the competition, MADE-WIC, VulSATD (ICPC'25), CodeBERT | paper references |
