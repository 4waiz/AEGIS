# Next steps (ordered)

1. Finish teacher candidates on the internal split: UniXcoder-base (running), CodeBERT-base with
   the AEGIS pipeline (whitespace normalization, dynamic padding, bf16).
2. Student sweep without distillation: UniXcoder truncated to 1/2/3/4/6 layers (768 wide), and narrow
   models (e.g. 4 x 256) trained from scratch; max_length 256 vs 512.
3. Multitask distillation from the best teacher into the best 1-2 student shapes.
4. Score every candidate with `scripts/measure.py` (official loop, sorted dynamic padding) and keep
   reports/leaderboard.csv; pick the final model by validation F1 + measured cost.
5. Seeds x3 for the final model and main ablations; error analysis on internal val.
6. Colab T4 notebook for the final model (the user must run it; the agent cannot access Colab).
7. Paper (ICSE'27 / IEEE two-column template). Needs a LaTeX toolchain: install MiKTeX or
   TeX Live locally, or compile on Overleaf (user action).
8. Publish the final model to the Hugging Face Hub under the user's account (requires approval).
9. Decide with the user whether to send research/organizer_questions.md.

## How to resume
- `export AEGIS_ARTIFACTS=C:/Users/awaiz/aegis_artifacts` (checkpoints live outside OneDrive).
- Training logs: experiments/<name>/train_log.jsonl and metrics.json.
