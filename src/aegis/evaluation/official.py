"""Faithful re-implementation of the NLBSE'27 official evaluation notebook.

Source: CodeBERT_baseline.ipynb in github.com/nlbse2027/code-comment-classification
(commit 310c2b0, 2026-09-03) and the identical Colab notebook 1q1fGPpU7uxaK1mUVHa3b2xJOtEwhWwLG.

Measurement algorithm, unchanged from the notebook:
  * tokenize everything up front (not timed);
  * for each batch: start a wall clock, then under ``torch.no_grad()``, fp16 autocast and
    ``torch.profiler.profile(with_flops=True)`` run the forward pass twice; stop the clock;
  * GFLOPs = sum of ``key_averages().flops`` / 1e9 over all batches;
  * reported runtime = total_time / 2 and reported GFLOPs = total_flops / 2;
  * predictions = sigmoid(logits) > 0.5 (the participant may change the threshold logic of
    their own model, but the notebook uses 0.5).

Deviation: the notebook applies ``sigmoid`` to the fp16 logits; we cast the logits to fp32 first. Decisions can
differ only when the logit is within about 2e-3 of zero (at most 3 of 30,655 public-test functions for the final
model, reports/test_statistics.json).

Note: ``torch.profiler`` only attributes FLOPs to aten::mm/addmm/bmm/baddbmm/mul/add/conv2d.
Fused attention (``scaled_dot_product_attention``) and GELU contribute 0 FLOPs to the
official count. AEGIS reports an analytic FLOP estimate alongside the official number.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

LABELS_NAME = ["Vulnerability", "MAT"]
MAX_AVG_RUNTIME = 1000
MAX_AVG_FLOPS = 3_000_000


def submission_score(avg_f1: float, avg_runtime: float, avg_flops: float) -> float:
    """Official ranking formula (notebook cell 4), unmodified."""
    return (
        0.6 * avg_f1
        + 0.2 * max((MAX_AVG_RUNTIME - avg_runtime) / MAX_AVG_RUNTIME, 0)
        + 0.2 * max((MAX_AVG_FLOPS - avg_flops) / MAX_AVG_FLOPS, 0)
    )


def class_metrics(predictions: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Per-class precision/recall/F1 computed exactly as in the notebook."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    scores = []
    for class_idx, name in enumerate(LABELS_NAME):
        p_, l_ = predictions[:, class_idx], labels[:, class_idx]
        tp = int(np.sum((p_ == 1) & (l_ == 1)))
        fp = int(np.sum((p_ == 1) & (l_ == 0)))
        tn = int(np.sum((p_ == 0) & (l_ == 0)))
        fn = int(np.sum((p_ == 0) & (l_ == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(l_)
        scores.append(
            dict(cat=name, precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                 tp=tp, fp=fp, tn=tn, fn=fn)
        )
    return scores


@dataclass
class OfficialResult:
    scores: list[dict]
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_runtime: float  # seconds, total over the split / 2 (notebook definition)
    avg_flops: float  # GFLOPs, total over the split / 2 (notebook definition)
    score: float
    n_samples: int
    batching: str
    batch_size: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def make_batches(
    input_ids: Sequence[Sequence[int]],
    pad_id: int,
    batch_size: int,
    mode: str = "official",
    max_length: int | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
    """Build (input_ids, attention_mask) CPU batches.

    mode="official": pad every sequence to the global maximum length (``padding=True`` on the
        whole split, as in the notebook) and keep the original order.
    mode="sorted": sort by length and pad each batch only to its own maximum (dynamic padding);
        predictions are mapped back to the original order with the returned permutation.
    Returns the batches and ``order`` such that batch rows correspond to ``order``.
    """
    lengths = np.array([len(x) for x in input_ids])
    if max_length is not None:
        lengths = np.minimum(lengths, max_length)
    if mode == "official":
        order = np.arange(len(input_ids))
        global_len = int(lengths.max())
    elif mode == "sorted":
        order = np.argsort(-lengths, kind="stable")
        global_len = None
    else:
        raise ValueError(mode)
    batches = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        width = global_len if global_len is not None else int(lengths[idx].max())
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            seq = list(input_ids[i])[: lengths[i]]
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
        batches.append((ids, mask))
    return batches, order


def _logits(outputs) -> torch.Tensor:
    return outputs.logits if hasattr(outputs, "logits") else outputs


def run_measurement(
    forward: Callable[[torch.Tensor, torch.Tensor], object],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    threshold: float | Sequence[float] = 0.5,
    autocast: bool = True,
    progress: bool = True,
    repetitions: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The notebook's per-batch measurement loop. Returns (preds, probs, avg_runtime, avg_gflops).

    The released notebook runs 2 forward passes per batch; the organizers' final evaluation runs 10
    (e-mail of 2026-10-22). Runtime and GFLOPs are averaged over the passes either way."""
    total_flops = 0.0
    total_time = 0.0
    preds, probs_all = [], []
    thr = torch.as_tensor(threshold, dtype=torch.float32, device=device)
    iterator = batches
    if progress:
        from tqdm import tqdm

        iterator = tqdm(batches, desc="Predicting...")
    for input_ids, attention_mask in iterator:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        begin = time.time()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
                with torch.profiler.profile(with_flops=True) as p:
                    for _ in range(repetitions):
                        outputs = forward(input_ids, attention_mask)
        total_time += time.time() - begin
        total_flops += sum(k.flops for k in p.key_averages()) / 1e9
        probs = torch.sigmoid(_logits(outputs).float())
        preds.append((probs > thr).int().cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        del input_ids, attention_mask, outputs, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(probs_all), total_time / repetitions, total_flops / repetitions


def evaluate_official(
    forward: Callable,
    input_ids: Sequence[Sequence[int]],
    labels: np.ndarray,
    pad_id: int,
    device: torch.device,
    batch_size: int = 600,
    batching: str = "official",
    threshold: float | Sequence[float] = 0.5,
    max_length: int | None = None,
    progress: bool = True,
    repetitions: int = 2,
) -> tuple[OfficialResult, np.ndarray]:
    """Run the official measurement and scoring. Returns the result and per-sample probabilities
    in the original sample order."""
    batches, order = make_batches(input_ids, pad_id, batch_size, batching, max_length)
    preds_sorted, probs_sorted, avg_runtime, avg_flops = run_measurement(
        forward, batches, device, threshold=threshold, progress=progress, repetitions=repetitions
    )
    preds = np.empty_like(preds_sorted)
    probs = np.empty_like(probs_sorted)
    preds[order] = preds_sorted
    probs[order] = probs_sorted
    scores = class_metrics(preds, labels)
    avg_f1 = float(np.mean([s["f1"] for s in scores]))
    result = OfficialResult(
        scores=scores,
        avg_precision=float(np.mean([s["precision"] for s in scores])),
        avg_recall=float(np.mean([s["recall"] for s in scores])),
        avg_f1=avg_f1,
        avg_runtime=avg_runtime,
        avg_flops=avg_flops,
        score=submission_score(avg_f1, avg_runtime, avg_flops),
        n_samples=len(labels),
        batching=batching,
        batch_size=batch_size,
    )
    return result, probs
