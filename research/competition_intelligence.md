# NLBSE'27 Multitask Code Classification: competition intelligence

Compiled 2026-10-15. **V** = verified by reading the cited source; **U** = unverified / inferred.
All official sources were read directly (repository cloned, notebooks downloaded and parsed,
Hugging Face API queried), not via search snippets.

## 1. Authoritative sources

| Source | URL | Status |
|---|---|---|
| Competition rules page | https://nlbse2027.github.io/tools/ (`/tool/` in the brief returns 404) | V, read 2026-10-15 |
| Workshop CFP | https://conf.researchr.org/home/icse-2027/nlbse-2027 | V |
| Official repository | https://github.com/nlbse2027/code-comment-classification (HEAD `310c2b0`, 2026-09-03, CC0) | V, cloned |
| Instructions notebook | Colab `1b_EiXx5woyCrDntwbljfi-BwS9Dwvhoa` | V, downloaded via Drive export |
| Baseline/evaluation notebook | Colab `1q1fGPpU7uxaK1mUVHa3b2xJOtEwhWwLG` = repo `CodeBERT_baseline.ipynb` | V (code identical except a model-id typo `commnet` in the Colab copy) |
| Dataset | `NLBSE/nlbse27-code-comment-classification`, revision `feca19b9…` (2026-08-19) | V |
| Baseline checkpoint | `NLBSE/nlbse27-code-comment-classification-baseline`, revision `b8ffd89b…` (2026-08-19), no model card | V |
| Submission form | https://docs.google.com/forms/d/1EinwODkpYaEzejk3TCiLxFgPmkwv0J5d1jOkFMePj7Q (linked from the rules page) | V (link only; not opened/submitted) |

## 2. Rules (verified wording)

| Question | Answer | Source |
|---|---|---|
| Model type | "a **single** ML/DL model (e.g., SVM or BERT) considering **all the categories** … we do not allow using distinct models for subsets of categories." | instructions notebook (V) |
| Pretrained encoders | Allowed: "we allow using classification models pre-trained on other datasets but they must be fine-tuned on the provided training dataset." | instructions notebook (V) |
| External data | Not allowed: "No external information sources can be used to train/fine-tune … **only the functions** can be used for classification." | instructions notebook (V) |
| Metadata columns | `Projectname`/`Filepath` exist in the data, but the rule above restricts inputs to functions. AEGIS uses `Function` only. | (V + decision D2) |
| Teacher models / distillation | Not addressed. A teacher fine-tuned on the provided training data uses no external information; the deployed model is single. Treated as allowed; listed as an organizer question. | U |
| Ensembles | Not allowed as the deployed model (single-model rule). | V |
| Training-set transformations | "Participants may preprocess, sample, apply over/under-sampling, select a subset of the attributes, perform feature engineering, split the training set into a model-fine-tuning validation set, etc." | V |
| Test-set transformations | "you may apply any preprocessing or feature engineering on test dataset except sampling, rebalancing, undersampling, or oversampling techniques." | V |
| Quantization, thresholds, post-processing | Not explicitly addressed. Thresholding is part of the official inference (`probs > 0.5`); choosing other thresholds tuned on training data only is treated as part of the model. Quantization: not addressed (organizer question). | U |
| Checkpoint download during evaluation | The official notebook downloads the baseline from the HF Hub, so hosting the final model on the Hub matches the official workflow. | V (by construction) |
| Evaluation hardware | "**the final evaluation must be run on a Google Colab T4 instance**." | V |
| Hidden test | 61,311 instances, held by organizers; "the delta between F1 (public) and F1 (hidden) is less than 0.5%". | V |
| Paper length | 2-4 pages. CFP: "Tool competition entries … 4" pages. | V |
| Paper content | architecture; preprocessing; tuning procedure on the training set; results on the validation (public test) set; link to documented code. | V |
| Format | ICSE'27 formatting and submission instructions: **IEEE** this year ("IEEE format is being used this year, whereas last year it was ACM format"); LaTeX `\documentclass[10pt,conference]{IEEEtran}` without compsoc options; 10pt body; PDF. | V (conf.researchr.org/track/icse-2027/icse-2027-research-track) |
| Generative-AI disclosure | ICSE'27 quotes IEEE policy: AI-generated content "shall be disclosed in the acknowledgements section of any paper submitted to an IEEE Conference". The AEGIS paper must include this disclosure. | V (same page) |
| ORCID | ICSE asks authors to obtain an ORCID iD. | V (same page) |
| Anonymity | Competition page: "need not be double-blind". General CFP: "All submissions must be anonymized". Resolved in favour of the competition-specific page (D1); confirmation requested. | V (conflict documented) |
| Submission channel | Google Form on the competition page (not HotCRP). **As of 2026-10-21 the linked form is still the NLBSE'26 form** ("fifth edition", deadline 19 Dec 2025). Fields: submission name, authors (`name, email, affiliation`), contact e-mails, authorship confirmation, paper PDF (max 10 MB), challenge (Code Comment Classification / Skill Classification), repository URL. It also states that submissions must cite the related work listed on the competition website. | V (form opened read-only, nothing filled) |
| Acceptance criteria | clarity and detail; open-source code including training/tuning/evaluation pipeline; correct training/tuning/evaluation on the provided data; correct metric reporting; clear documentation. | V |
| Deadlines (AoE) | submission 2026-11-13; notification 2026-12-11; camera-ready 2027-01-29. | V |
| Venue | ICSE 2027, Dublin, 25 Apr - 1 May 2027; NLBSE day not yet announced. | V |
| Registration / presentation / fees | Not stated on NLBSE pages; ICSE registration terms not yet published for workshops. | U (check closer to camera-ready) |
| Contacts | Moritz Mock (momock@unibz.it), Thomas Borsani (tborsani@unibz.it), Barbara Russo (brusso@unibz.it). | V |

**Organizers' answers (e-mail from Moritz Mock, 2026-10-22; research/organizer_questions.md).** A new
submission form will be released; own inference code is allowed if the timing/GFLOPs measurement is kept; the
constants are rescaled for the hidden split; the final evaluation uses 10 forward passes per batch (the
notebook's 2 was an oversight); the baseline score is 0.73 (0.7299 with 10 passes); distillation, tuned
thresholds and fp16 are allowed; the competition paper need not be double-blind. These supersede the "U" rows
above on quantization/thresholds and the anonymity conflict.

## 3. Labels and inputs

- Columns: `Projectname`, `Filepath`, `Function`, `Label` (`int8[2]`, order `[weakness, MAT]`). There is **no separate comment column**. The instructions notebook says so explicitly: "The current two-label CSV files contain Function but no separate developer-comment column." Comments occur only inside `Function`. (V)
- Weakness label = Boolean OR of the Devign, Big-Vul and "W" annotations of MADE-WIC. (V, README)
- MAT = the SATD annotation produced by the "Matches task Annotation Tags" heuristic of Guo et al. (TOSEM 2021, doi 10.1145/3447247). MADE-WIC labels a function TD "if at least one comment" carries TODO, FIXME, XXX or HACK (MADE-WIC, arXiv 2408.05163). The data agrees: in the public test split, a word-boundary match of `todo|fixme|hack|xxx` inside comments gives MAT P=1.000, R=0.877, F1=0.935 (reports/dataset_stats.json). (V)
- The weakness source "W" consists of CWE-derived heuristics from the WeakSATD replication package (Russo, Camilli, Mock, MSR 2022), per the MADE-WIC paper. (V, via research agent reading arXiv 2408.05163)
- Data preparation (README, V): clang-format (LLVM style); functions with > 500 `cl100k_base` tokens removed; over-represented joint classes downsampled; deterministic jointly stratified 70/10/20 split.

## 4. Official evaluation and scoring (extracted from the notebook)

```
score = 0.6 * avg_F1
      + 0.2 * max((1000 - avg_runtime) / 1000, 0)
      + 0.2 * max((3_000_000 - avg_GFLOPs) / 3_000_000, 0)
```
- `avg_F1` = unweighted mean of the per-task F1 scores (threshold 0.5 on sigmoid outputs).
- The whole public test split is tokenized at once with `padding=True, truncation=True, max_length=512`, so **every sample is padded to the longest sequence (512)**. Batch size 600.
- For each batch, wall-clock time covers **two** forward passes under `torch.no_grad()`, `torch.cuda.amp.autocast()` (fp16) and `torch.profiler.profile(with_flops=True)`. Tokenization is not timed.
- GFLOPs = Σ `key_averages().flops` / 1e9. Reported runtime and GFLOPs = totals / 2, i.e. **per-pass totals over the whole split, not per-sample averages**.
- Official baseline output (Colab T4): GFLOPs 2,666,509.71; runtime 198.93 s; F1 0.938979 / 0.881393; mean 0.910186.

### Verified discrepancies / ambiguities
1. **Displayed score.** The notebook displays `0.71`, but the formula applied to its own printed values gives **0.7286** (rounds to 0.73). 0.71 is what the formula gives if the FLOPs term is 0, which suggests the cell ran with a different constant. (V, arithmetic in tests/test_official.py)
2. **Repetitions.** The instructions say runtime "should be measured 10 times"; the notebook code uses 2 repetitions. (V)
3. **Hidden-test scale.** The hidden split (61,311) is 2x the public test, and runtime/GFLOPs are totals. For the baseline the FLOPs term would be 0 on the hidden split (~5.3M GFLOPs > 3M) unless the constants are rescaled. Lower is better either way, so the optimisation target is unchanged. (V + arithmetic)
4. **Profiler coverage.** `torch.profiler` counts FLOPs only for mm/addmm/bmm/baddbmm/mul/add/conv2d; fused `scaled_dot_product_attention` and GELU count as 0. The official baseline figure equals the analytic linear-layer count (24·d²·L·tokens = 2.67e15), so attention is not included. (V, local test)
5. **Template leftovers.** The instructions mention "19 scores", "sentences", "language categories" (copied from the 2024-26 comment-classification tasks). (V)
6. **Tooling.** `torch.cuda.amp.autocast()` is deprecated in current PyTorch; it is equivalent to `torch.autocast("cuda")`. (V)

### Local reproduction of the official baseline (checkpoint reproduction, D6)
`scripts/official_notebook_replica.py` on an RTX 5080 Laptop GPU (power-limited to ~54 W during the run):

| | Official (T4) | AEGIS replica |
|---|---|---|
| Weakness P/R/F1 | 0.9427 / 0.9353 / 0.9390 | 0.9428 / 0.9353 / 0.9391 |
| MAT P/R/F1 | 0.9783 / 0.8020 / 0.8814 | 0.9783 / 0.8020 / 0.8814 |
| GFLOPs | 2,666,509.7 | 2,666,497.7 |
| Runtime (s) | 198.93 | 193.90 |
| Score (formula) | 0.7286 | 0.7296 |

Training hyperparameters were recovered from `training_args.bin` by disassembling the pickle (not executed): HF Trainer, batch 8, lr 2e-5, weight decay 0.01, 5 epochs, linear decay without warmup, seed 42, fp32, no evaluation during training, transformers 4.51.3.

They are confirmed by the organizers' setup repository **github.com/moritzmock/NLBSE2027-setup** (created 2026-08-15, last push 2026-08-25; read 2026-10-15). It contains `curate_dataset.py`, `codebert_train.py` and `codebert_train.sbatch`. (V)
- `codebert_train.py`: `microsoft/codebert-base`, `max_length=512` with truncation, `DataCollatorWithPadding` (dynamic padding during training), batch 8, lr 2e-5, weight decay 0.01, 5 epochs, seed 42, threshold 0.5.
- The job script passes `--train-limit 0`, i.e. the full training split, on one 80 GB GPU.
- `curate_dataset.py` loads `LeadingComment` (the comment *above* a function) but the final output columns are `("Projectname", "Filepath", "Function", "Label")`. **The leading comment is not released.** This is consistent with the 13% of MAT positives that have no SATD keyword inside `Function`: their tag sits in the dropped leading comment.
- Split naming: the organizers' pipeline creates train/validation/test = 70/10/20. The Hugging Face `test` split (30,655) corresponds to their **validation** split; their `test` split (61,311) is the hidden one.

## 5. Implications for strategy
- Exchange rate implied by the formula: **0.01 mean-F1 = 0.006 score = 90,000 GFLOPs = 30 s**.
- The baseline loses almost the whole FLOPs term (0.022 of 0.2) because of global padding to 512 tokens plus a 12-layer, 768-wide encoder. Its weakness F1 is high, its MAT recall is low (0.80).
- MAT is a keyword task on comments; its practical ceiling given the visible input is ~0.935 F1 (about 12% of MAT positives have no SATD keyword in `Function`).
- Efficiency terms saturate. Below ~50k GFLOPs and ~5 s the two terms are within 0.005 of their maximum (0.4), after which mean F1 dominates again.

## 6. Prior NLBSE competitions (2024-2026)
Different tasks (sentence-level comment classification per language) and much smaller scoring constants (2025-26: 5 s and 5,000 GFLOPs), so **no numeric comparison with 2027 is meaningful**.

| Year | Baseline | Score formula | Notable entries (self-reported) | Winner |
|---|---|---|---|---|
| 2024 | STACC (SetFit, all-mpnet-base-v2) | 0.75·F1 + 0.25·runtime term (participant copy of notebook; U) | Dopamin: CodeBERT + domain post-training + HSUM, max_length 32; F1 0.74, score 0.703 | not verified (team claims "Best Tool Award") |
| 2025 | SetFit paraphrase-MiniLM-L3-v2 (score 0.70) | 0.6/0.2/0.2, 5 s, 5,000 GFLOPs, unclipped | Peña & Herbold (SBERT + classical heads; score 0.6536 on Colab, below baseline); Mock et al. (CodeBERT + weighted BCE; F1 0.726, efficiency terms clipped); GRAPHiC; CodeComClassify; back-translation | Peña & Herbold per Univ. Passau news; organizer ranking not verified |
| 2026 | SetFit paraphrase-MiniLM-L6-v2 (F1 0.6508, score 0.70) | same constants, clipped at 0 | TraCC (CodeBERT→TinyBERT-4L-312D KD, asymmetric focal loss, per-label thresholds; F1 0.671, 481 GFLOPs, 0.565 s); Lightweight KD / "Distilling Semantics" (CodeBERT→MiniLM KD; F1 0.668, score 0.74); LoRA-MME (4 LoRA encoders ensemble; macro F1 0.6867, efficiency terms 0, score 0.412); Q-SYNTH (augmentation + SetFit MTL head; score 0.73) | not verified |

Sources (all read by the research agent): nlbse2024/2025/2026.github.io tools and program pages; github.com/nlbse2024, nlbse2025, nlbse2026 repositories and baseline notebooks; arXiv 2408.04663 (Dopamin), 2506.08581 (Peña & Herbold), 2501.15854 (Mock et al.), 2603.03959 (LoRA-MME), 2601.19383 (Q-SYNTH); DOIs 10.1145/3786164.3794838 (TraCC), 10.1145/3786164.3794836 (Lightweight KD), 10.1145/3786164.3794837 (LoRA-MME), 10.1145/3786164.3805715 (Q-SYNTH), 10.1109/NLBSE66842.2025.00010/12/13/14/16/17; Univ. Passau news page. ACM full texts returned HTTP 403, so the TraCC and Lightweight-KD claims above come from abstracts and Crossref metadata only.

**Lessons that transfer:** distillation into 4-6-layer students and per-label thresholds were the only clear winners under efficiency-weighted scores. Large encoders and ensembles lost the efficiency terms entirely. Several entries mis-reported metrics (weighted instead of macro F1, wrong formula variant), and several had no code or an empty model card. Clear, correct metric reporting and a published model with a working Colab are differentiators for acceptance.

## 7. Open questions for the organizers
See `research/organizer_questions.md`.
