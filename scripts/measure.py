"""Score a trained checkpoint with the official NLBSE'27 measurement protocol.

The measurement loop (2 forward passes inside torch.profiler, fp16 autocast, runtime and
GFLOPs halved) is the official one (aegis.evaluation.official). The AEGIS preprocessing stored
in the checkpoint config (normalization, max length, thresholds) is applied before timing,
as the notebook also tokenizes before timing.

Usage:
  python scripts/measure.py <checkpoint_dir> [--split test|val] [--batching sorted|official]
                            [--batch-size 600] [--threshold tuned|0.5] [--out result.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.data.dataset import load_internal_splits  # noqa: E402
from aegis.envinfo import environment  # noqa: E402
from aegis.evaluation.flops import encoder_flops  # noqa: E402
from aegis.evaluation.official import evaluate_official, make_batches  # noqa: E402
from aegis.paths import revision_for  # noqa: E402
from aegis.training.train import prepare_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batching", default="sorted", choices=["sorted", "official"])
    ap.add_argument("--batch-size", type=int, default=600)
    ap.add_argument("--threshold", default="tuned")
    ap.add_argument("--normalize", help="override the checkpoint's input normalization (e.g. raw)")
    ap.add_argument("--out")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rev = revision_for(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, revision=rev).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=rev)
    aegis_cfg = getattr(model.config, "aegis", None) or {"normalize": "raw", "max_length": 512,
                                                          "truncation": "head", "thresholds": [0.5, 0.5]}
    if args.normalize:
        aegis_cfg = {**aegis_cfg, "normalize": args.normalize}
    df = load_internal_splits()[args.split]
    tag = f"{args.split}_{Path(args.checkpoint).parent.name}_{aegis_cfg['normalize']}"
    inputs = prepare_split(df, tokenizer, aegis_cfg, tag)
    labels = df[["y_vuln", "y_mat"]].to_numpy()
    thr = aegis_cfg["thresholds"] if args.threshold == "tuned" else [float(args.threshold)] * 2

    def forward(ids, mask):
        return model(input_ids=ids, attention_mask=mask)

    # No warm-up: the official notebook times the very first batch too.
    result, probs = evaluate_official(forward, inputs, labels, tokenizer.pad_token_id, device,
                                      batch_size=args.batch_size, batching=args.batching, threshold=thr)
    batches, _ = make_batches(inputs, tokenizer.pad_token_id, args.batch_size, args.batching)
    result.extra = {
        "checkpoint": args.checkpoint, "split": args.split, "thresholds": thr, "aegis": aegis_cfg,
        "analytic": encoder_flops(model.config, [(b[0].shape[0], b[0].shape[1]) for b in batches]),
        "mean_input_tokens": float(np.mean([len(x) for x in inputs])),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "env": environment(),
    }
    print(json.dumps({k: v for k, v in result.to_dict().items() if k != "extra"}, indent=1, default=float))
    print("analytic", result.extra["analytic"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=2, default=float), encoding="utf-8")
        np.save(Path(args.out).with_suffix(".probs.npy"), probs)


if __name__ == "__main__":
    main()
