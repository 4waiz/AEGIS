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
