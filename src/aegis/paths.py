"""Filesystem locations. Heavy artifacts can live outside the repo via AEGIS_ARTIFACTS."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AEGIS_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / "nlbse27"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("AEGIS_ARTIFACTS", REPO_ROOT / "artifacts"))
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "figures"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Pinned upstream revisions (verified 2026-10-15 via the Hugging Face API).
DATASET_REPO = "NLBSE/nlbse27-code-comment-classification"
DATASET_REVISION = "feca19b90d1cc9fcf8ba5d6e693c4a94e249585c"
BASELINE_REPO = "NLBSE/nlbse27-code-comment-classification-baseline"
BASELINE_REVISION = "b8ffd89b848e0b3fdf8a043816b5d1801c923a0b"
