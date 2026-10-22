# AEGIS decision log

Each entry: date, decision, evidence, consequences. Newest last.

## D1 (2026-10-15): Competition paper is not anonymized
- Evidence: nlbse2027.github.io/tools/ states competition papers "must conform to the ICSE'27
  formatting and submission instructions and need not be double-blind". The general NLBSE CFP on
  conf.researchr.org says "All submissions must be anonymized"; the competition-specific page is
  more specific and is treated as authoritative. Submission goes through the competition Google Form,
  not HotCRP. Kept as an open confirmation question for the organizers (research/organizer_questions.md).
- Consequence: the paper carries author names; the repository link can be non-anonymous.

## D2 (2026-10-15): Only the `Function` text is a model input
- Evidence: official instructions notebook: "No external information sources can be used ...
  **only the functions** can be used for classification." `Projectname`/`Filepath` are used only
  for analysis (audit, project-held-out diagnostic), never as features.

## D3 (2026-10-15): Optimize the official score, measured with the official loop
- Score = 0.6*meanF1 + 0.2*max((1000-t)/1000,0) + 0.2*max((3e6-G)/3e6,0), t and G being the
  whole-split totals divided by 2 (two timed forward passes per batch inside torch.profiler).
- Exchange rates implied by the formula: +0.01 mean F1 = +0.006 score = -90,000 GFLOPs = -30 s.
- `src/aegis/evaluation/official.py` re-implements the notebook loop unchanged; our own models may
  change batching (length-sorted dynamic padding) and preprocessing, which the instructions allow
  ("any preprocessing or feature engineering on test dataset except sampling"). Every reported
  number states its batching mode.

## D4 (2026-10-15): No exploitation of profiler blind spots
- torch.profiler attributes FLOPs only to addmm/mm/bmm/baddbmm/mul/add/conv2d; fused SDPA attention
  and GELU count as 0 (verified locally). We use standard Hugging Face modules only, never custom
  ops chosen to hide FLOPs, and we report an analytic FLOP count (linear + attention) next to the
  official profiler number.

## D5 (2026-10-15): Data splits
- Internal validation = 10% of the official train split, stratified by joint label, seed 42,
  exact-duplicate functions moved to the fit side. Used for model selection, early stopping and
  threshold tuning.
- Official public test split: final local reporting only; no tuning against it.
- Hidden test split: never accessed or inferred.

## D6 (2026-10-15): Baseline reproduction = checkpoint reproduction
- The organizers publish only the evaluation notebook and the trained checkpoint; training
  hyperparameters were recovered from `training_args.bin` (HF Trainer, bs 8, lr 2e-5, wd 0.01,
  5 epochs, linear decay, no warmup, seed 42, fp32) by disassembling the pickle without executing
  it. We evaluate the official checkpoint with the official loop (checkpoint reproduction) and do
  not claim a training reproduction.

## D7 (2026-10-15): Heavy artifacts outside the OneDrive-synced repo
- Checkpoints go to `$AEGIS_ARTIFACTS` (default `./artifacts`, git-ignored). On the development
  machine it points outside OneDrive to avoid syncing multi-GB files.

## D8 (2026-10-15): Encoder family and input = UniXcoder + single-line normalization
- Evidence: reports/token_budget.csv. UniXcoder's tokenizer on whitespace-collapsed input yields
  153.5 tokens/function on public test vs 235.1 for CodeBERT on raw input (-35%), and truncates 0.5%
  vs 10.6% of functions at 512 tokens. The 12-layer UniXcoder teacher reaches internal-val mean F1
  0.940 (3 epochs). Other encoders were not tried; the budget went to depth/distillation instead.

## D9 (2026-10-15): Student depth chosen by projected score, not F1
- Projected public-test score (internal-val F1, val-measured cost scaled by 30,655/21,243):
  L1 0.911, L2 0.937, L3 0.941, L4 0.937, L6 0.928, L12 0.897 (no KD). Efficiency terms saturate,
  so depth beyond 3 costs more score than its F1 gain returns.
- Narrow 4x256 encoder trained from scratch plateaus at val mean F1 0.859: code pretraining matters.

## D10 (2026-10-15): Multitask distillation from the 12-layer teacher
- Same-seed comparison, internal val mean F1: L2 0.914 -> 0.917, L3 0.930 -> 0.934, L4 0.932 -> 0.937.
  Initializing the L2 student from the teacher or from pretrained UniXcoder makes no difference (0.9169 vs 0.9172).
- Lead candidate: kd_teacher_L3 (projected 0.943). Seed repeats (43, 44) of L3 with and without KD
  run in round 3 to test whether the gain exceeds seed variance.

## D11 (2026-10-15): MAT is at its ceiling
- Error analysis (reports/error_analysis_kd_teacher_L3.md): every MAT false negative lacks an SATD tag
  in the input (tags sit in the unreleased leading comment); on functions with a tag, MAT F1 is 0.998.
  No further modelling effort goes into MAT; weakness is the only lever.

## D12 (2026-10-15): Final model = kd_teacher_L3 (fixed before any public-test scoring of students)
- 3-layer UniXcoder student, initialized from the fine-tuned teacher's first 3 layers, multitask
  distillation (alpha 0.5, T 2), 4 epochs, seed 42, single-line input, per-task thresholds from
  internal validation.
- Seed repeats (internal val, 3 seeds): KD 0.9337 +/- 0.0002 vs no-KD 0.9306 +/- 0.0008 mean F1
  (Welch p = 0.018); weakness F1 0.9366 +/- 0.0009 vs 0.9301 +/- 0.0006 (p = 0.0013); MAT unchanged.
- A 6-epoch schedule gives no gain (0.9338 vs 0.9338). The seed-42 run is used because it was the
  pre-declared primary run, not because it scored best (the three seeds agree within 0.0003).
- Public test is now scored once for: the final model, the no-KD L3 student and the teacher
  (reported comparisons; no further selection).

## D13 (2026-10-15): D12 reopened for input normalization (before any public-test scoring)
- The raw-input ablation (kd_teacher_L3_raw, internal val) gave the same mean F1 (0.9332 vs 0.9337) but
  higher weakness F1 (0.9419 vs 0.9366) and lower MAT F1 (0.9245 vs 0.9308) at +17% GFLOPs.
- Before scoring anything on the public test, we run `collapse` normalization (indentation removed,
  line breaks kept) and a second raw seed. The final input format is chosen on internal-validation
  projected score only; round-5 public-test scoring was stopped before it started and waits for this.

## D14 (2026-10-15): Final model = kd_teacher_L3_mixed with thresholds folded into the output bias
- Still decided on internal validation only (no public-test scores of any student had been computed).
- Input-format ablation (internal val, KD, 3 layers): single-line 0.9337 (3 seeds), collapse 0.9347,
  raw 0.9332/0.9336 (2 seeds). Line structure helps weakness (raw 0.942 vs 0.937) and hurts MAT;
  the extra tokens cost about as much score as the F1 changes return.
- The organizers may run the unmodified notebook (raw input, global padding, threshold 0.5) instead of
  our inference code (open question 1). A single-line-trained model drops to 0.925 mean F1 there.
  Training on a seeded 50/50 mix of raw and single-line inputs gives 0.9329 with our pipeline and 0.9289
  in the vanilla notebook (raw model: 0.9332 / 0.9332). Projected score, our pipeline / vanilla:
  single-line 0.9434 / 0.9009, mixed 0.9430 / 0.9033, raw 0.9403 / 0.9059. The mixed model has the
  smallest worst-case loss (0.0026) and is chosen.
- Threshold folding: subtracting logit(t_k) from the bias of output k makes sigmoid > 0.5 equivalent to
  sigmoid > t_k, so the tuned thresholds also apply in any 0.5-threshold pipeline. Exported model:
  `aegis_final` (see src/aegis/optimization/export.py); thresholds in its config become [0.5, 0.5].
- If the organizers confirm that participants' own inference code is used, single-line-only training
  would be +0.0004 projected (within seed noise); not worth a switch.

## D15 (2026-10-15): Incident: orphaned round-5 run; all public-test measurements repeated
- Stopping the round-5 background task killed only its wrapper shell. The inner script kept waiting and,
  after round 4 finished, scored kd_teacher_L3, student_unix_L3 and the teacher on the public test
  (08:10-08:40) while round 4b was training on the same GPU. These files were never read before D14
  (decided on internal-validation numbers only, at 09:38) and were committed unchanged in commit 28bc8e7.
- Their F1 and GFLOPs are unaffected, but their runtimes were measured under GPU contention and are
  invalid. All public-test measurements (baselines included) are re-run sequentially on an idle GPU
  by scripts/queue_round5b.sh; the paper uses only those.
- Lesson: background queues are now stopped by killing the python/bash processes, not only the wrapper.

## D16 (2026-10-21): Corrections after the internal reviewer audit (reports/reviewer_audit.md)
- D14's statement "no public-test scores of any student had been computed" is inaccurate: the orphaned run of
  D15 had computed (but nobody had read) public-test scores before D14. The decision itself used internal
  validation only.
- D14's 0.9289 is the mixed model in the vanilla-notebook setting *with its tuned thresholds* (what the folded
  export does). The committed robust_val_vanilla_notebook.json files were measured at 0.5 before folding
  (0.9248 / 0.9280 / 0.9324). Worst-case losses recomputed from those committed numbers (0.0045 / 0.0026 /
  0.0031) select the same model.
- Tuned thresholds do not improve public-test F1 (|change| <= 0.0004 for students, -0.0019 for the teacher;
  reports/test_statistics.json). They are kept, folded into the bias, only so that a 0.5 cut-off reproduces the
  validated operating point.
- Weakness-label provenance: OSPR projects (80% of the data) have weakness = W heuristic (MADE-WIC Table 2).
  The paper now says "weakness" rather than "vulnerability" and reports F1 per label source.
- The 3-layer optimum is flat: with costs totalled over the hidden split and unscaled constants, 2 layers would
  lead by 0.001. Kept 3 layers; organizer question 2 asks how the hidden split is scored.
- Control for teacher initialization (student_tinit_L3, 3 seeds) and two more seeds of the final configuration
  (kd_teacher_L3_mixed_s43/s44) were run in round 6.
- Round-6 results: teacher init without KD 0.9302 / 0.9317 / 0.9313 (mean 0.9311) vs KD 0.9337 and no-KD
  pretrained init 0.9306; 73% of the weakness gain comes from the distillation loss. Final configuration
  across seeds: 0.9329 / 0.9334 / 0.9325 (seed 42, the submitted model, is the median).
- Official T4 (Colab, 2026-10-21): baseline 192.7 s / 0.730, AEGIS 16.4 s / 0.942; the public Hub model
  reproduces the reported predictions exactly (experiments/reproduce_hub/).

## D17 (2026-10-22): Organizers' answers (research/organizer_questions.md)
- Our own inference code (input normalization, length-sorted batching with dynamic padding) is allowed as long
  as the timing and GFLOPs measurement is kept. The headline numbers therefore use the AEGIS pipeline; the
  unmodified-notebook result stays in the paper as a robustness check.
- The constants are rescaled for the hidden split, so the public-test depth optimum (3 layers) carries over; the
  D16 caveat (2 layers ahead by 0.001 with unscaled constants) no longer applies.
- The final evaluation averages 10 forward passes per batch, not the notebook's 2. `scripts/measure.py
  --repetitions` supports both; scripts/queue_round7.sh measures AEGIS and the baseline both ways in one
  session (experiments/repetitions_check/). The paper's tables keep the 2-pass protocol of the released notebook.
- The official baseline score is 0.73 (0.7299 in the organizers' 10-pass rerun; 0.7286 from the printed values).
- Distillation, tuned thresholds and fp16 are allowed; the competition paper need not be anonymized.
- A new submission form will be published; do not use the NLBSE'26 form.
- Round-7 result (experiments/repetitions_check/): GFLOPs per pass are identical with 2 and 10 passes; scores moved
  by at most 0.00012 (AEGIS 0.94250 -> 0.94238, baseline 0.73151 -> 0.73150). Runtimes in that session were about
  twice those of Table I although the PD_Turbo scheme was active (cause not identified), so only the 2-vs-10
  difference within the session is used; Table I is unchanged.
