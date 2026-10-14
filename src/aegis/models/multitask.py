"""Model construction.

Every AEGIS model is a standard Hugging Face ``*ForSequenceClassification`` with two sigmoid
outputs ([weakness, MAT]): one shared encoder and one 2-row output projection, i.e. a single
model with a task-specific weight vector per label. This keeps the final checkpoint loadable
by the official notebook (``from_pretrained``) without custom code.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def build_model(cfg: dict):
    """cfg keys: init (HF id | 'scratch'), tokenizer, num_layers, layer_selection,
    scratch (dict of config overrides), embed_init ('pca:<hf id>' | None)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or cfg["init"])
    labels = {"id2label": {0: "weakness", 1: "MAT"}, "label2id": {"weakness": 0, "MAT": 1}}
    if cfg["init"] == "scratch":
        base = AutoConfig.from_pretrained(cfg["scratch_base"])
        conf = copy.deepcopy(base)
        for k, v in cfg.get("scratch", {}).items():
            setattr(conf, k, v)
        conf.num_labels = 2
        conf.problem_type = "multi_label_classification"
        conf.id2label, conf.label2id = labels["id2label"], labels["label2id"]
        model = AutoModelForSequenceClassification.from_config(conf)
        if cfg.get("embed_init", "").startswith("pca:"):
            _pca_embedding_init(model, cfg["embed_init"][4:])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["init"], num_labels=2, problem_type="multi_label_classification",
            ignore_mismatched_sizes=True, **labels,
        )
    if cfg.get("num_layers"):
        truncate_layers(model, cfg["num_layers"], cfg.get("layer_selection", "first"))
    return model, tokenizer


def _encoder(model):
    base = getattr(model, model.base_model_prefix)
    return base.encoder


def truncate_layers(model, k: int, selection: str = "first") -> None:
    """Keep k transformer layers (first k, or evenly spaced) and update the config."""
    layers = _encoder(model).layer
    n = len(layers)
    if k >= n:
        return
    if selection == "first":
        keep = list(range(k))
    elif selection == "spread":
        keep = np.linspace(0, n - 1, k).round().astype(int).tolist()
    else:
        raise ValueError(selection)
    _encoder(model).layer = torch.nn.ModuleList([layers[i] for i in keep])
    model.config.num_hidden_layers = k


@torch.no_grad()
def _pca_embedding_init(model, source: str) -> None:
    """Initialize a narrow model's token embeddings with the top principal components of a
    pretrained model's embedding matrix (same tokenizer required)."""
    src = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2)
    emb = src.get_input_embeddings().weight.float()
    d = model.config.hidden_size
    centered = emb - emb.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=d, center=False)
    proj = centered @ v[:, :d]
    proj = proj / proj.std() * model.config.initializer_range * 1.0
    target = model.get_input_embeddings().weight
    n = min(target.shape[0], proj.shape[0])
    target[:n] = proj[:n].to(target.dtype)


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
    return {"total": total, "embedding": emb, "non_embedding": total - emb}
