"""Lexical reference model: TF-IDF over code tokens + logistic regression (one weight vector per
task, i.e. a single 2-output linear model). Diagnostic only: how much of each task is lexical?

Thresholds are tuned on internal validation; the public test split is scored once.
Usage: python scripts/linear_reference.py
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.data.dataset import load_internal_splits  # noqa: E402
from aegis.evaluation.metrics import f1_at, tune_thresholds  # noqa: E402
from aegis.paths import EXPERIMENTS_DIR  # noqa: E402

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|[^\sA-Za-z_\d]{1,3}")


def main() -> None:
    splits = load_internal_splits()
    t0 = time.time()
    vec = TfidfVectorizer(token_pattern=None, tokenizer=TOKEN.findall, lowercase=False, ngram_range=(1, 2),
                          min_df=3, max_features=500_000, sublinear_tf=True, dtype=np.float32)
    x_fit = vec.fit_transform(splits["fit"].Function)
    x_val, x_test = vec.transform(splits["val"].Function), vec.transform(splits["test"].Function)
    y_fit = splits["fit"][["y_vuln", "y_mat"]].to_numpy()
    probs_val, probs_test = [], []
    for k in range(2):
        clf = LogisticRegression(C=8.0, max_iter=2000, solver="liblinear")
        clf.fit(x_fit, y_fit[:, k])
        probs_val.append(clf.predict_proba(x_val)[:, 1])
        probs_test.append(clf.predict_proba(x_test)[:, 1])
    pv, pt = np.stack(probs_val, 1), np.stack(probs_test, 1)
    y_val = splits["val"][["y_vuln", "y_mat"]].to_numpy()
    y_test = splits["test"][["y_vuln", "y_mat"]].to_numpy()
    thr = tune_thresholds(pv, y_val)
    out = {"model": "tfidf_uni_bigram_logreg", "n_features": len(vec.vocabulary_),
           "fit_time_s": round(time.time() - t0, 1), "thresholds": thr,
           "val_at_0.5": f1_at(pv, y_val, [0.5, 0.5]), "val_tuned": f1_at(pv, y_val, thr),
           "test_tuned_thresholds_from_val": f1_at(pt, y_test, thr)}
    d = EXPERIMENTS_DIR / "linear_reference"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
