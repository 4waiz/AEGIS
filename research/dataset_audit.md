# Dataset audit: NLBSE'27 multitask code classification

All numbers are produced by `scripts/dataset_audit.py` (outputs: `reports/dataset_stats.json`,
`reports/label_distribution.csv`, `reports/project_distribution.csv`, `reports/data_quality.md`,
`figures/dataset_overview.pdf`). Dataset revision `feca19b90d1cc9fcf8ba5d6e693c4a94e249585c`.
Labels of the public test split are used here only for descriptive statistics; no model or
hyperparameter is selected on them.

## 1. Size and label distribution

| Split | n | weakness prevalence | MAT prevalence |
|---|---:|---:|---:|
| official train | 214,586 | 0.3279 | 0.0165 |
| official public test | 30,655 | 0.3279 | 0.0165 |
| internal fit (90% of train) | 193,343 | 0.3279 | 0.0165 |
| internal validation (10% of train) | 21,243 | 0.3284 | 0.0164 |

Joint distribution (train): `[0,0]` 66.65%, `[1,0]` 31.70%, `[1,1]` 1.09%, `[0,1]` 0.56%.
The counts match the README (214,586 / 30,655), and the splits are stratified by joint label as documented.

The tasks are weakly correlated (phi = 0.093): P(weakness | MAT) = 0.663 vs P(weakness | not MAT) = 0.322.
MAT is the minority task (1.65%), so its F1 is sensitive to a few hundred positives
(505 in the public test).

## 2. Inputs: length, comments, languages

- **Length (CodeBERT BPE incl. special tokens, train):** mean 234, median 176, p90 522, p99 788,
  max 2,745. **10.5%** of training and 10.6% of test functions exceed 512 tokens and are truncated by
  the official pipeline. (The 500-token filter of the organizers is in `cl100k_base` tokens, which
  are coarser than RoBERTa BPE on indented C code.)
- **Weakness correlates with length:** weakness prevalence by length quartile is 8.5%, 23.4%, 39.2%, 60.3%.
  A length-only threshold reaches weakness F1 0.603 on the test split, so length is a strong
  confounder (see Limitations).
- **Whitespace dominates token counts** (`scripts/token_budget.py`, `reports/token_budget.csv`, public test):

  | Tokenizer | Normalization | mean tokens | > 512 | real tokens (clipped at 512) |
  |---|---|---:|---:|---:|
  | CodeBERT | raw (official) | 235.1 | 10.6% | 6.75M |
  | CodeBERT | single line | 173.2 | 2.6% | 5.27M |
  | UniXcoder | raw | 182.0 | 3.8% | 5.52M |
  | UniXcoder | single line | 153.5 | 0.5% | 4.70M |

  clang-format indentation becomes many BPE tokens. Collapsing whitespace and using UniXcoder's
  tokenizer shortens inputs by 35% and nearly removes truncation.
- **Official padding efficiency:** the official notebook pads the whole split to 512 positions
  (15.70M positions for 30,655 functions), so only 43.0% of processed positions are real tokens.
  With length-sorted batching and the UniXcoder/single-line input, a same-size encoder would process
  about 3.3x fewer positions.
- **Comments:** 14.2% of functions contain at least one comment. Comments are embedded in `Function`;
  there is no separate comment column.
- **Languages (file extensions, train):** `.c` 174,311; `.h` 18,816; `.cc` 14,317; `.cpp` 6,322;
  535 without extension and a few hundred with unusual extensions (`.objs`, `.txt`, `.s`, ...).
  The corpus is overwhelmingly C, with some C++.

## 3. MAT is a keyword annotation

| Rule on the public test split | Precision | Recall | F1 |
|---|---:|---:|---:|
| word-boundary `todo/fixme/hack/xxx` inside comments | 1.000 | 0.877 | **0.935** |
| case-insensitive substring inside comments | 0.991 | 0.877 | 0.931 |
| substring anywhere in the function | 0.771 | 0.879 | 0.822 |

Every train function whose comments contain `todo`, `fixme` or `hack` as a word is MAT-positive
(`xxx`: 99.3%). 463 MAT-positive training functions (13.1% of positives) contain no SATD keyword in
`Function`, and 329 contain no comment at all. The annotation was therefore most likely computed on
text that is not part of the released function (e.g. comments preceding the function). For these
instances the label is not recoverable from the input, so **~0.935 is a practical F1 ceiling for MAT**.
The official baseline reaches 0.881 (recall 0.802).

## 4. Projects and metadata

15 project names, all present in both splits. Linux dominates: `Linux Kernel` 163,881 + `linux` 11,547
(81.7% of train). Name variants (`Linux Kernel`/`linux`, `Chrome`/`Chromium`, `php`/`php-src`) probably
reflect the source collections (OSPR vs Big-Vul/Devign). Weakness prevalence varies strongly by
project: Chrome 18.1%, Linux Kernel 32.6%, qemu 68.6%, FFmpeg 77.7%. qemu and FFmpeg are the Devign
projects.

Metadata is **not** a model input (competition rule). As a leakage probe, predicting weakness from the
majority label of the same `Filepath` in train gives test F1 0.426 (95.6% of test rows share a file
with train). A project-majority probe gives F1 0.125. File identity carries some signal but far less than
the code itself.

## 5. Duplicates and train/test similarity

| Check | Count |
|---|---:|
| exact duplicate rows inside train (whitespace-normalized) | 1,315 |
| test rows with an exact duplicate in train | 386 (1.3%) |
| weakness-label agreement for those | 95.6% |
| test rows with an identifier/literal-abstracted clone (type-2 proxy) in train | 5,553 (18.1%) |
| weakness-label agreement for those (train majority) | 98.2% |
| train duplicate groups with conflicting weakness label (exact / abstracted) | 39 / 245 |
| train duplicate groups with conflicting MAT label (exact / abstracted) | 3 / 87 |

Exact leakage is small, but 18% of test functions have a structural clone in train with almost always
the same weakness label. Part of the high weakness F1 of any model probably comes from such
near-duplicates. The internal validation split moves exact duplicates to the fit side, but keeps type-2
clones, mirroring the official split. Results are reported per clone status in the error analysis.

## 6. Annotation consistency

- Conflicting labels among exact duplicates are rare (39 weakness, 3 MAT groups), which indicates
  label noise of the order of 0.1-1% among duplicates.
- MAT false positives of the keyword rule are `XXX` placeholders (e.g. `"XXXXXX\0"`, `XXX_reserved`),
  which the annotation itself sometimes labels negative. These cases are ambiguous by construction.
- The weakness label is the OR of three sources (Devign, Big-Vul, W). Prior work reports substantial
  label noise in Big-Vul and Devign (see research/related_work.md). We cannot audit vulnerability labels
  against ground truth and treat them as given.

## 7. Consequences for AEGIS

1. Whitespace normalization and a compact tokenizer cut sequence length by about 35%. That reduces FLOPs
   and truncation at no modelling cost.
2. Length-sorted dynamic padding removes the ~57% padding waste of the official pipeline.
3. MAT needs comment-keyword recognition, which is cheap. Its recall (not precision) limits the
   baseline, so threshold calibration on internal validation is a natural fix.
4. Weakness carries most of the remaining difficulty; model depth/width should be spent there.
5. Report clone-stratified results and a length-only reference so readers can judge shortcut learning.
