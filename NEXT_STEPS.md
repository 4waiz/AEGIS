# Next steps (ordered)

1. Paper compiles (tools/tectonic, 4 pages). Re-run `python scripts/paper_numbers.py` and recompile after any
   new result.
2. **Fill author details** in paper/AEGIS.tex (affiliation, e-mail) and confirm the AI-use disclosure wording.
3. T4 evaluation done (experiments/colab_t4/); the HF model is public.
4. Organizer questions **sent** 2026-10-22 (awaiz42ad@gmail.com). When they reply: confirm the submission form,
   the inference procedure (our pipeline vs. unmodified notebook) and hidden-split scoring; adjust the paper if needed.
5. Reviewer audit done (reports/reviewer_audit.md).
6. Submission package ready in submission/ (checklist, REPRODUCE.md, form answers, model manifest).
7. Make the GitHub repository and the HF model public at submission time (needs the user's approval).
8. Optional: demo UI; extra seeds for the mixed-format model; project-held-out diagnostic.

## How to resume
- `export AEGIS_ARTIFACTS=C:/Users/awaiz/aegis_artifacts` (checkpoints live outside OneDrive).
- Final model: `$AEGIS_ARTIFACTS/aegis_final` (experiments/FINAL_MODEL names its experiment).
- Regenerate everything derived: `python scripts/leaderboard.py && python scripts/ablation_table.py &&
  python scripts/paper_numbers.py && python scripts/make_figures.py`.
- When stopping a background queue, kill its python/bash processes (see DECISIONS.md D15).
