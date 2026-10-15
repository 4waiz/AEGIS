# Next steps (ordered)

1. Paper compiles (tools/tectonic, 4 pages). Re-run `python scripts/paper_numbers.py` and recompile after any
   new result.
2. **Fill author details** in paper/AEGIS.tex (affiliation, e-mail) and confirm the AI-use disclosure wording.
3. **Host the model** on the Hugging Face Hub under the user's account (needs approval and a token), then run
   notebooks/AEGIS_T4_evaluation.ipynb on a Colab T4 (the user runs it) to get T4 runtime numbers.
4. **Decide on sending** research/organizer_questions.md (needs the user's approval).
5. Reviewer audit (reports/reviewer_audit.md) and fixes.
6. Submission package: submission/ directory, checklist, README reproduction commands tested end to end.
7. Make the GitHub repository public at submission time (needs the user's approval).
8. Optional: demo UI; extra seeds for the mixed-format model; project-held-out diagnostic.

## How to resume
- `export AEGIS_ARTIFACTS=C:/Users/awaiz/aegis_artifacts` (checkpoints live outside OneDrive).
- Final model: `$AEGIS_ARTIFACTS/aegis_final` (experiments/FINAL_MODEL names its experiment).
- Regenerate everything derived: `python scripts/leaderboard.py && python scripts/ablation_table.py &&
  python scripts/paper_numbers.py && python scripts/make_figures.py`.
- When stopping a background queue, kill its python/bash processes (see DECISIONS.md D15).
