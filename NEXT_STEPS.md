# Next steps (ordered)

1. Paper compiles (tools/tectonic, 4 pages). Re-run `python scripts/paper_numbers.py` and recompile after any
   new result.
2. **Fill author details** in paper/AEGIS.tex (affiliation, e-mail) and confirm the AI-use disclosure wording.
3. **Run the T4 evaluation** (user): open notebooks/AEGIS_T4_evaluation.ipynb in Colab with a T4 runtime, run
   all cells (the login cell needs a read token while the HF repo 4waiz/aegis-nlbse27 is private), and save the
   outputs into experiments/colab_t4/ so the paper can report T4 numbers.
4. **Decide on sending** research/organizer_questions.md (needs the user's approval).
5. Reviewer audit (reports/reviewer_audit.md) and fixes.
6. Submission package: submission/ directory, checklist, README reproduction commands tested end to end.
7. Make the GitHub repository and the HF model public at submission time (needs the user's approval).
8. Optional: demo UI; extra seeds for the mixed-format model; project-held-out diagnostic.

## How to resume
- `export AEGIS_ARTIFACTS=C:/Users/awaiz/aegis_artifacts` (checkpoints live outside OneDrive).
- Final model: `$AEGIS_ARTIFACTS/aegis_final` (experiments/FINAL_MODEL names its experiment).
- Regenerate everything derived: `python scripts/leaderboard.py && python scripts/ablation_table.py &&
  python scripts/paper_numbers.py && python scripts/make_figures.py`.
- When stopping a background queue, kill its python/bash processes (see DECISIONS.md D15).
