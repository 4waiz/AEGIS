"""Export a trained AEGIS checkpoint for submission.

Threshold folding: the per-task decision thresholds t_k tuned on internal validation are folded into
the bias of the output layer, b_k <- b_k - logit(t_k). Then sigmoid(z'_k) > 0.5 <=> z_k > logit(t_k)
<=> sigmoid(z_k) > t_k, so any pipeline that thresholds at 0.5 (such as the official notebook)
reproduces the tuned decisions. The exported config records thresholds [0.5, 0.5] and the original
thresholds under ``aegis.folded_thresholds``.

Usage: python -m aegis.optimization.export <checkpoint_dir> <out_dir>
"""

from __future__ import annotations

import math
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def output_layer(model) -> torch.nn.Linear:
    head = model.classifier
    return head.out_proj if hasattr(head, "out_proj") else head


@torch.no_grad()
def fold_thresholds(model) -> list[float]:
    cfg = dict(model.config.aegis)
    thr = cfg["thresholds"]
    layer = output_layer(model)
    for k, t in enumerate(thr):
        layer.bias[k] -= math.log(t / (1 - t))
    cfg["folded_thresholds"] = thr
    cfg["thresholds"] = [0.5] * len(thr)
    model.config.aegis = cfg
    return thr


def export(checkpoint: str, out_dir: str) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    thr = fold_thresholds(model)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"folded thresholds {thr} into the output bias; saved to {out_dir}")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
