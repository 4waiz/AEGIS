import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from aegis.data.preprocess import normalize
from aegis.optimization.export import fold_thresholds

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(thresholds):
    conf = RobertaConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=40, num_labels=2,
                         problem_type="multi_label_classification")
    torch.manual_seed(0)
    model = RobertaForSequenceClassification(conf).eval()
    model.config.aegis = {"thresholds": thresholds}
    return model


def test_threshold_folding_preserves_decisions():
    thr = [0.86, 0.95]
    model = tiny_model(thr)
    ids = torch.randint(3, 50, (64, 12))
    with torch.no_grad():
        before = torch.sigmoid(model(input_ids=ids).logits)
        fold_thresholds(model)
        after = torch.sigmoid(model(input_ids=ids).logits)
    assert torch.equal(before > torch.tensor(thr), after > 0.5)
    assert model.config.aegis["thresholds"] == [0.5, 0.5]
    assert model.config.aegis["folded_thresholds"] == thr
    # logit shift is exactly logit(t)
    shift = torch.logit(before) - torch.logit(after)
    assert torch.allclose(shift, torch.tensor([math.log(t / (1 - t)) for t in thr]).expand_as(shift), atol=1e-4)


def test_colab_notebook_normalization_matches_training():
    """The Colab notebook re-implements single-line normalization inline; it must match the package."""
    code = "int f(int a)\t{\n  // TODO: x\n\n  return  a;\r\n}\n"
    notebook_normalize = re.sub(r"\s+", " ", code).strip()
    assert notebook_normalize == normalize(code, "single_line")
    nb = (ROOT / "scripts" / "make_colab_notebook.py").read_text(encoding="utf-8")
    assert "re.sub(r'\\\\s+', ' ', code).strip()" in nb


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "nlbse27" / "data" / "train-00000-of-00001.parquet").exists(),
                    reason="dataset not downloaded")
def test_internal_split_is_deterministic_and_matches_committed_ids():
    from aegis.data.dataset import internal_split, load_split

    _, val = internal_split(load_split("train"))
    committed = np.loadtxt(ROOT / "data" / "splits" / "internal_val_row_ids.txt", dtype=np.int64)
    assert np.array_equal(np.sort(val["row_id"].to_numpy()), committed)
