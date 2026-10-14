"""Input normalization and cached tokenization.

Normalization is applied identically at training and inference time. It only uses the
``Function`` text (competition rule: no other information sources).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from aegis.data.dataset import extract_comments
from aegis.paths import PROCESSED_DIR

_WS_RUN = re.compile(r"[ \t]+")
_LEADING_WS = re.compile(r"\n[ \t]+")
_ANY_WS = re.compile(r"\s+")
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def normalize(code: str, mode: str = "raw") -> str:
    """Whitespace normalization variants.

    raw          -- unchanged clang-formatted function
    collapse     -- strip indentation and collapse runs of spaces/tabs, keep line breaks
    single_line  -- collapse every whitespace run (incl. newlines) to one space
    comments_first -- single_line, with all comments moved in front of the code so that
                      truncation cannot drop them (SATD evidence lives in comments)
    """
    if mode == "raw":
        return code
    if mode == "collapse":
        return _WS_RUN.sub(" ", _LEADING_WS.sub("\n", code)).strip()
    if mode == "single_line":
        return _ANY_WS.sub(" ", code).strip()
    if mode == "comments_first":
        comments = " ".join(extract_comments(code))
        body = _COMMENT_RE.sub(" ", code)
        return _ANY_WS.sub(" ", f"{comments} {body}").strip()
    raise ValueError(f"unknown normalization mode: {mode}")


class TokenCache:
    """Untruncated token ids (no special tokens) stored as flat int32 + offsets."""

    def __init__(self, ids: np.ndarray, offsets: np.ndarray):
        self.ids = ids
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, i: int) -> np.ndarray:
        return self.ids[self.offsets[i] : self.offsets[i + 1]]

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def tokenize_cached(texts: list[str], tokenizer, tag: str, cache_dir: Path = PROCESSED_DIR) -> TokenCache:
    """Tokenize without special tokens or truncation; cache keyed by tokenizer + content hash."""
    h = hashlib.md5()
    h.update(tokenizer.name_or_path.encode())
    h.update(str(len(texts)).encode())
    for t in texts[:: max(1, len(texts) // 2000)]:
        h.update(t.encode("utf-8", "ignore"))
    path = cache_dir / f"tok_{tag}_{h.hexdigest()[:12]}.npz"
    if path.exists():
        z = np.load(path)
        return TokenCache(z["ids"], z["offsets"])
    enc = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    offsets = np.zeros(len(enc) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(e) for e in enc])
    ids = np.fromiter((t for e in enc for t in e), dtype=np.int32, count=int(offsets[-1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, offsets=offsets)
    return TokenCache(ids, offsets)


def build_inputs(
    cache: TokenCache,
    cls_id: int,
    sep_id: int,
    max_length: int,
    prefix: list[int] | None = None,
    truncation: str = "head",
) -> list[np.ndarray]:
    """Add special tokens and truncate. ``head_tail`` keeps the start and the end of the function."""
    prefix = prefix or []
    budget = max_length - 2 - len(prefix)
    out = []
    for i in range(len(cache)):
        ids = cache.get(i)
        if len(ids) > budget:
            if truncation == "head":
                ids = ids[:budget]
            elif truncation == "head_tail":
                head = budget * 2 // 3
                ids = np.concatenate([ids[:head], ids[len(ids) - (budget - head) :]])
            else:
                raise ValueError(truncation)
        out.append(np.concatenate([[cls_id], prefix, ids, [sep_id]]).astype(np.int64))
    return out
