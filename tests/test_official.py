import numpy as np
import torch

from aegis.evaluation.official import class_metrics, evaluate_official, make_batches, submission_score


def test_score_reproduces_notebook_formula():
    # Values printed by the official notebook (cell 3 output).
    s = submission_score(0.9101860725779012, 198.93055331707, 2666509.7131617293)
    expected = 0.6 * 0.9101860725779012 + 0.2 * (1000 - 198.93055331707) / 1000 + 0.2 * (3e6 - 2666509.7131617293) / 3e6
    assert abs(s - expected) < 1e-12
    assert round(s, 2) == 0.73  # the notebook displays 0.71; see research/competition_intelligence.md


def test_score_clips_negative_terms():
    assert submission_score(0.5, 5000, 9e9) == 0.3
    assert abs(submission_score(1.0, 0, 0) - 1.0) < 1e-12


def test_class_metrics_matches_hand_computation():
    preds = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    s = class_metrics(preds, labels)
    assert s[0]["tp"] == 1 and s[0]["fp"] == 1 and s[0]["fn"] == 1
    assert abs(s[0]["f1"] - 0.5) < 1e-12
    assert s[1]["precision"] == 0.5 and s[1]["recall"] == 1.0


def test_sorted_batching_restores_order():
    seqs = [[0, 5, 2], [0, 5, 5, 5, 2], [0, 2], [0, 7, 7, 2]]
    batches, order = make_batches(seqs, pad_id=1, batch_size=2, mode="sorted")
    assert sorted(order.tolist()) == [0, 1, 2, 3]
    assert batches[0][0].shape[1] == 5 and batches[1][0].shape[1] == 3

    def forward(ids, mask):  # logit encodes sequence length -> verifies order restoration
        n = mask.sum(1).float()
        return torch.stack([n - 3.5, -n], 1)

    labels = np.array([[0, 0], [1, 0], [0, 0], [1, 0]])
    res, probs = evaluate_official(forward, seqs, labels, 1, torch.device("cpu"), batch_size=2,
                                   batching="sorted", progress=False)
    assert res.scores[0]["f1"] == 1.0
    assert probs.shape == (4, 2)


def test_official_batching_pads_globally():
    seqs = [[0, 2], [0, 5, 5, 5, 2]]
    batches, _ = make_batches(seqs, pad_id=1, batch_size=1, mode="official")
    assert all(b[0].shape[1] == 5 for b in batches)
