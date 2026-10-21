# Internal reviewer audit (2026-10-21)

Three independent reviewers (fresh agents, read-only, no shared context with the author's session) reviewed the
paper and repository at commit `8091e4c`:

- **A:** ML methodology;
- **B:** software-engineering relevance;
- **C:** reproducibility and competition compliance.

Verdicts were weak accept (A), weak accept (B), and "compliant on every substantive rule, with open
reproducibility gaps" (C). Reviewer A recomputed every number in the paper from the experiment files; all of
them matched.

Each finding is listed below with what was done about it. **Fixed** means resolved in the code, data or paper,
with evidence. **Disclosed** means stated in the paper or documentation. **Open** means it needs the author's
action.

## Major findings

| # | Reviewer | Finding | Resolution |
|---|---|---|---|
| 1 | B | The weakness label is mostly the CWE heuristic W, not vulnerabilities. | **Fixed.** MADE-WIC Table 2 confirms OSPR (Linux, Chromium, Firefox) has no original vulnerability labels. The project-to-source mapping is consistent: about 35% of each project is retained. OSPR is 80% of the data, so its weakness label *is* W. Changes: title now says "Weakness"; the paper states the provenance; the error analysis reports F1 by label source (OSPR 0.954, Big-Vul 0.866, Devign 0.768; `scripts/error_analysis.py` `source_dataset`). |
| 2 | B | No related work; organizers' VulSATD paper not cited; dynamic padding presented as novel. | **Fixed.** Added a related-work paragraph citing VulSATD (required by the competition page), TraCC and Lightweight KD (metadata verified via Crossref), EarlyBIRD and fairseq. Padding is now presented as a fix to the evaluation pipeline. |
| 3 | B, A | On MAT the model only matches a keyword rule (0.931 vs. 0.935), and the paper did not say so. | **Fixed.** The paper reports that AEGIS agrees with the tag rule on 30,649 of 30,655 functions (+4 FP), and that the MAT gain over the baseline comes from avoiding truncation. All 39 tagged positives the baseline misses have their tag beyond token 512 (`reports/test_statistics.json`). |
| 4 | A | The KD effect is confounded with teacher initialization. | **Fixed by experiment:** 3-seed control with teacher initialization and no KD (`configs/student_tinit_L3*.yaml`). The paper reports the result and the effect is attributed accordingly. Welch p-values were replaced by seed ranges plus paired-bootstrap CIs on test. |
| 5 | A | A test-scored model that beats AEGIS (single-line KD, 0.9447) was omitted. | **Fixed.** Added to Table I. The mixed-input cost is quantified: weakness −0.004 [−0.006, −0.001], exchanged for robustness to the unmodified notebook. |
| 6 | A | The abstract's headline score assumes our inference code. | **Fixed.** The abstract gives both scores (0.944 / 0.908). |
| 7 | A, C | No T4 measurement. | **Open (needs the author).** Reaching the private model from Colab needs a public model or the author's token. The notebook is ready, revisions are pinned, and it was smoke-tested. The paper states that no T4 numbers are reported. |
| 8 | A | The 3-layer optimum holds only at public-test size. | **Disclosed.** Totalled over the hidden split without rescaled constants, 2 layers would lead by 0.001 (0.9275 vs. 0.9264). Organizer question 2 asks about this. |
| 9 | C | User-specific paths in configs (blocker). | **Fixed** (`bc7847f`): `${AEGIS_ARTIFACTS}` placeholders, `load_config` expansion, test. |
| 10 | C | Incomplete reproduction steps. | **Fixed:** `submission/REPRODUCE.md` gives an ordered recipe with timings, including the teacher-logit dump and export. |
| 11 | C | Checkpoints missing; no hash. | **Fixed.** The artifacts folder had been deleted (Recycle Bin) and was restored. `submission/model_manifest.json` records SHA-256 of every file of the released model. |
| 12 | C | No pinned versions or revisions. | **Fixed:** `requirements-lock.txt`; `PINNED_REVISIONS` (UniXcoder, CodeBERT, baseline, final model) used in every `from_pretrained`; revisions pinned in the notebook. |

## Minor findings

| # | Reviewer | Finding | Resolution |
|---|---|---|---|
| 13 | A | The threshold-tuning gain was measured in-sample. | **Fixed.** On test, tuned thresholds change student mean F1 by at most 0.0004 (teacher −0.0019). The paper now presents thresholds as robustness to a 0.5 cut-off, not as an accuracy gain. |
| 14 | A | No uncertainty on test results. | **Fixed:** paired-bootstrap 95% CIs (B=2000) in the text (`scripts/test_statistics.py`). |
| 15 | A | Validation numbers are optimistic. | **Disclosed** in the setup section. |
| 16 | A | Over-strong causal claims ("pretraining matters more than width", short functions). | **Fixed.** The from-scratch claim was removed from the text (it remains in Table II as an observation). Error rates (FNR) replace F1 for prevalence-confounded comparisons. |
| 17 | A | Novelty overstated; 13× mixes padding and model. | **Fixed:** 13× is decomposed into 2.3× (padding) × 5.7× (model/input). The cost model is described as counting what the profiler counts. |
| 18 | B, C | The "public test used once" wording vs. the D15 incident. | **Fixed:** the paper discloses the discarded accidental run. A correction note is in DECISIONS.md (D16). |
| 19 | B | Rebalanced prevalence and token filter not mentioned. | **Fixed:** stated in Sec. II and Limitations. |
| 20 | B | Method missing: 90% fit split, mixed input, weight decay, early stopping. | **Fixed** in Secs. III–IV. |
| 21 | B | Leading-comment claim unsourced. | **Fixed:** footnote with the organizers' repository; hedged wording; "61 of 62". |
| 22 | B | "AEGIS" not expanded; "MAT is solved" overstated. | **Fixed.** |
| 23 | C | "Unmodified notebook" row was emulated. | **Fixed:** the verbatim replica (`scripts/official_notebook_replica.py`) on the exported model gives identical TP/FP/FN and GFLOPs (`experiments/kd_teacher_L3_mixed/official_replica_test.json`). |
| 24 | C | fp32 vs. fp16 sigmoid deviation. | **Disclosed** in `official.py`. At most 3 public-test functions are within the affected margin. |
| 25 | C | D14 numbers not traceable. | **Fixed** in D16: 0.9289 is the mixed model in the vanilla setting with its folded thresholds (from the saved probabilities). The worst-case-loss ranking is unchanged with the committed 0.5-threshold numbers. |
| 26 | C | Collapse ablation and second raw seed omitted. | **Fixed:** both in Table II. |
| 27 | C | Attention FLOPs not quantified. | **Fixed:** +11% (baseline) and +5.5% (AEGIS) in Sec. II. |
| 28 | C | Split provenance. | **Fixed:** `data/splits/internal_val_row_ids.txt` committed and checked by a test; the dataset revision is recorded in every `env.json`. |
| 29 | C | Encoding bugs (cp1252). | **Fixed:** explicit UTF-8 in 38 text I/O calls. |
| 30 | C | Portability (replica path, CUDA hard-coded in the logit dump). | **Fixed.** |
| 31 | C | Missing tests (folding, notebook parity, split determinism). | **Fixed:** `tests/test_export_and_splits.py`. Total: 13 tests. |
| 32 | C | Model card placeholder; model license not in README. | **Fixed.** |
| 33 | B | `nlbse2027` bib entry for an unpublished overview. | **Kept:** it is the citation the organizers ask for on the competition page. |

## Still open (author action)

1. Affiliation and e-mail in the paper and the form; confirm the AI-use disclosure.
2. T4 run: make the Hugging Face model public (or log in on Colab), then run `notebooks/AEGIS_T4_evaluation.ipynb`.
3. Send `research/organizer_questions.md`. Question 0 matters most: the linked submission form is the NLBSE'26 one.
4. At submission: make the GitHub repository and the model public, then submit the form.
