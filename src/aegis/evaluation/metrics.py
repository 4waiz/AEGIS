"""Metrics and threshold calibration (thresholds are tuned on internal validation only)."""

from __future__ import annotations

import numpy as np

from aegis.evaluation.official import class_metrics


def f1_at(probs: np.ndarray, labels: np.ndarray, thresholds) -> dict:
    preds = (probs > np.asarray(thresholds)[None, :]).astype(int)
    scores = class_metrics(preds, labels)
    return {
        "f1_vuln": scores[0]["f1"], "f1_mat": scores[1]["f1"],
        "p_vuln": scores[0]["precision"], "r_vuln": scores[0]["recall"],
        "p_mat": scores[1]["precision"], "r_mat": scores[1]["recall"],
        "mean_f1": (scores[0]["f1"] + scores[1]["f1"]) / 2,
    }


def best_threshold(probs: np.ndarray, labels: np.ndarray, grid=None) -> tuple[float, float]:
    """Threshold maximizing F1 for one task (exhaustive over a fixed grid)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best = (0.5, -1.0)
    pos = labels == 1
    for t in grid:
        pred = probs > t
        tp = np.sum(pred & pos); fp = np.sum(pred & ~pos); fn = np.sum(~pred & pos)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def tune_thresholds(probs: np.ndarray, labels: np.ndarray) -> list[float]:
    return [best_threshold(probs[:, k], labels[:, k])[0] for k in range(labels.shape[1])]
