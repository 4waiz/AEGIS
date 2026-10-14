"""Loading the official NLBSE'27 splits and building the internal validation split.

Only the ``Function`` column is used as model input. The competition rules state that
no external information sources may be used and that "only the functions can be used
for classification", so ``Projectname`` and ``Filepath`` are kept for analysis only.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from aegis.paths import DATASET_REPO, DATASET_REVISION, PROCESSED_DIR, RAW_DIR

SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

# Fraction of the official training split held out for model selection.
INTERNAL_VAL_FRACTION = 0.10
INTERNAL_VAL_SEED = 42

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def download(revision: str = DATASET_REVISION, local_dir: Path = RAW_DIR) -> Path:
    """Download the official dataset snapshot at a pinned revision."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=DATASET_REPO, repo_type="dataset", revision=revision, local_dir=str(local_dir)
    )
    return Path(path)


def load_split(split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load an official split ('train' or 'test') with labels expanded into columns."""
    path = raw_dir / SPLIT_FILES[split]
    if not path.exists():
        download(local_dir=raw_dir)
    df = pd.read_parquet(path)
    labels = np.stack(df["Label"].values).astype(np.int64)
    df = df.assign(y_vuln=labels[:, 0], y_mat=labels[:, 1])
    df["joint"] = df["y_vuln"] * 2 + df["y_mat"]  # 0:[0,0] 1:[0,1] 2:[1,0] 3:[1,1]
    df["row_id"] = np.arange(len(df))
    return df


def function_hash(code: str) -> str:
    """Whitespace-insensitive hash used for duplicate analysis."""
    return hashlib.md5(re.sub(r"\s+", " ", code).strip().encode("utf-8")).hexdigest()


def extract_comments(code: str) -> list[str]:
    """Return C/C++ line and block comments in source order."""
    return _COMMENT_RE.findall(code)


def internal_split(
    train: pd.DataFrame,
    fraction: float = INTERNAL_VAL_FRACTION,
    seed: int = INTERNAL_VAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by joint label) split of the official train split into fit / internal-val.

    Functions whose normalized text occurs in both parts are moved entirely into the fit part,
    so that internal validation never contains an exact duplicate of a training function.
    """
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for _, group in train.groupby("joint"):
        idx = group.index.to_numpy()
        n_val = int(round(len(idx) * fraction))
        val_idx.extend(rng.choice(idx, size=n_val, replace=False).tolist())
    is_val = train.index.isin(val_idx)
    hashes = train["Function"].map(function_hash)
    fit_hashes = set(hashes[~is_val])
    is_val &= ~hashes.isin(fit_hashes).to_numpy()
    return train[~is_val].copy(), train[is_val].copy()


def load_internal_splits(cache: bool = True) -> dict[str, pd.DataFrame]:
    """Return {'fit', 'val', 'test'}; 'test' is the official public test split."""
    cache_path = PROCESSED_DIR / f"internal_val_ids_seed{INTERNAL_VAL_SEED}.npy"
    train = load_split("train")
    if cache and cache_path.exists():
        val_ids = np.load(cache_path)
        is_val = train["row_id"].isin(val_ids)
        fit, val = train[~is_val].copy(), train[is_val].copy()
    else:
        fit, val = internal_split(train)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, val["row_id"].to_numpy())
    return {"fit": fit, "val": val, "test": load_split("test")}
