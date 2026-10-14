"""Standalone inference for a trained AEGIS checkpoint.

The checkpoint is a standard Hugging Face sequence-classification model; its config carries an
``aegis`` block with the preprocessing (normalization, max length, truncation) and the per-task
decision thresholds selected on internal validation.

Example:
    from aegis.inference.predict import AegisPredictor
    p = AegisPredictor("path/or/hub-id")
    p.predict(["int f(char *s) { char b[8]; strcpy(b, s); /* TODO: bounds */ return 0; }"])
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aegis.data.preprocess import normalize

DEFAULTS = {"normalize": "raw", "max_length": 512, "truncation": "head", "thresholds": [0.5, 0.5],
            "prefix_tokens": None}


class AegisPredictor:
    def __init__(self, checkpoint: str, device: str | None = None, dtype: torch.dtype = torch.float16):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.cfg = {**DEFAULTS, **(getattr(self.model.config, "aegis", None) or {})}
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

    def encode(self, functions: list[str]) -> list[list[int]]:
        cfg, tok = self.cfg, self.tokenizer
        prefix = tok.convert_tokens_to_ids(cfg["prefix_tokens"]) if cfg.get("prefix_tokens") else []
        budget = cfg["max_length"] - 2 - len(prefix)
        out = []
        for ids in tok([normalize(f, cfg["normalize"]) for f in functions], add_special_tokens=False)["input_ids"]:
            if len(ids) > budget:
                if cfg["truncation"] == "head_tail":
                    head = budget * 2 // 3
                    ids = ids[:head] + ids[len(ids) - (budget - head):]
                else:
                    ids = ids[:budget]
            out.append([tok.cls_token_id, *prefix, *ids, tok.sep_token_id])
        return out

    @torch.no_grad()
    def predict_proba(self, functions: list[str], batch_size: int = 256) -> np.ndarray:
        enc = self.encode(functions)
        order = np.argsort([-len(e) for e in enc], kind="stable")
        probs = np.zeros((len(enc), 2), dtype=np.float32)
        pad = self.tokenizer.pad_token_id
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            width = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(enc[i])] = torch.tensor(enc[i])
                mask[r, : len(enc[i])] = 1
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
        return probs

    def predict(self, functions: list[str]) -> list[dict]:
        probs = self.predict_proba(functions)
        thr = np.asarray(self.cfg["thresholds"])
        return [{"weakness": int(p[0] > thr[0]), "MAT": int(p[1] > thr[1]),
                 "p_weakness": float(p[0]), "p_MAT": float(p[1])} for p in probs]
