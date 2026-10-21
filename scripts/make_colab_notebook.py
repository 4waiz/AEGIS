"""Generate notebooks/AEGIS_T4_evaluation.ipynb: a self-contained Colab notebook that scores an
AEGIS checkpoint (and, in the same session, the official baseline) with the official NLBSE'27
measurement loop on a Colab T4.

Usage: python scripts/make_colab_notebook.py [hf_model_id]
"""

import json
import sys
from pathlib import Path

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "4waiz/aegis-nlbse27"
REVISION = sys.argv[2] if len(sys.argv) > 2 else "2011b2fa1e2577c24b56b4a8d551634af3522730"

CELLS = [
    ("markdown", f"""# AEGIS: official NLBSE'27 measurement on a Colab T4

Runtime -> Change runtime type -> **T4 GPU**. Then run all cells.

The measurement loop below is copied from the official `CodeBERT_baseline.ipynb`
(github.com/nlbse2027/code-comment-classification). AEGIS only changes what happens *before*
timing (input normalization, tokenizer, length-sorted batching with dynamic padding) and the
per-task decision thresholds stored in the model config. Model: `{MODEL_ID}`."""),
    ("code", "!pip -q install datasets\nimport torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"),
    ("code", "# Log in only if the model repository is still private (it is public at submission time).\n"
             "from huggingface_hub import HfApi, login\n"
             f"try:\n    HfApi().model_info('{MODEL_ID}')\nexcept Exception:\n    login()"),
    ("code", """import re, time, numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ds = load_dataset('NLBSE/nlbse27-code-comment-classification', revision='feca19b90d1cc9fcf8ba5d6e693c4a94e249585c')
test_data = ds['test'].to_pandas()
labels = np.array(test_data['Label'].tolist())
device = torch.device('cuda')

max_avg_runtime = 1000
max_avg_flops = 3000000
def score(avg_f1, avg_runtime, avg_flops):
    return (0.6 * avg_f1
            + 0.2 * max((max_avg_runtime - avg_runtime) / max_avg_runtime, 0)
            + 0.2 * max((max_avg_flops - avg_flops) / max_avg_flops, 0))

def report(predictions, total_time, total_flops):
    rows = []
    for c, name in enumerate(['Vulnerability', 'MAT']):
        p, l = predictions[:, c], labels[:, c]
        tp = np.sum((p == 1) & (l == 1)); fp = np.sum((p == 1) & (l == 0)); fn = np.sum((p == 0) & (l == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        rows.append({'cat': name, 'precision': precision, 'recall': recall, 'f1': f1})
    scores = pd.DataFrame(rows)
    avg_f1, avg_runtime, avg_flops = scores.f1.mean(), total_time / 2, total_flops / 2
    print(scores)
    print('Compute in GFLOPs:', avg_flops, '| Avg runtime in seconds:', avg_runtime, '| mean F1:', avg_f1)
    print('submission score:', score(avg_f1, avg_runtime, avg_flops))
    return dict(f1=avg_f1, runtime=avg_runtime, gflops=avg_flops, score=score(avg_f1, avg_runtime, avg_flops))

def measure(model, batches, thresholds):
    total_flops, total_time, predictions = 0, 0, []
    for input_ids, attention_mask in tqdm(batches, desc='Predicting...'):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        begin = time.time()
        with torch.no_grad():
            with torch.autocast('cuda'):
                with torch.profiler.profile(with_flops=True) as p:
                    for _ in range(2):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_time += time.time() - begin
        total_flops += sum(k.flops for k in p.key_averages()) / 1e9
        probs = torch.sigmoid(outputs.logits.float())
        predictions.append((probs > torch.tensor(thresholds, device=device)).int().cpu().numpy())
        del input_ids, attention_mask, outputs, probs
        torch.cuda.empty_cache()
    return np.concatenate(predictions), total_time, total_flops"""),
    ("markdown", "## 1. Official baseline (unchanged notebook procedure, same session)"),
    ("code", """BASE, BASE_REV = 'NLBSE/nlbse27-code-comment-classification-baseline', 'b8ffd89b848e0b3fdf8a043816b5d1801c923a0b'
tok_b = RobertaTokenizer.from_pretrained(BASE, revision=BASE_REV)
model_b = RobertaForSequenceClassification.from_pretrained(BASE, revision=BASE_REV).to(device).eval()
enc = tok_b(test_data['Function'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
loader = DataLoader(TensorDataset(enc['input_ids'], enc['attention_mask']), batch_size=600)
preds_b, t_b, f_b = measure(model_b, list(loader), [0.5, 0.5])
baseline = report(preds_b, t_b, f_b)
del model_b; torch.cuda.empty_cache()"""),
    ("markdown", "## 2. AEGIS (preprocessing and batching happen before timing, as tokenization does in the official notebook)"),
    ("code", f"""MODEL = '{MODEL_ID}'
REVISION = '{REVISION}'
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
cfg = model.config.aegis
print(cfg)

def normalize(code):  # 'single_line': collapse every whitespace run to one space
    return re.sub(r'\\s+', ' ', code).strip() if cfg['normalize'] == 'single_line' else code

ids = tok([normalize(f) for f in test_data['Function']], add_special_tokens=True, truncation=True,
          max_length=cfg['max_length'])['input_ids']
order = np.argsort([-len(x) for x in ids], kind='stable')   # length-sorted, dynamic padding
batches = []
for s in range(0, len(order), 600):
    idx = order[s:s + 600]
    w = max(len(ids[i]) for i in idx)
    a = torch.full((len(idx), w), tok.pad_token_id); m = torch.zeros((len(idx), w), dtype=torch.long)
    for r, i in enumerate(idx):
        a[r, :len(ids[i])] = torch.tensor(ids[i]); m[r, :len(ids[i])] = 1
    batches.append((a, m))
preds_sorted, t_a, f_a = measure(model, batches, cfg['thresholds'])
preds_a = np.empty_like(preds_sorted); preds_a[order] = preds_sorted
aegis = report(preds_a, t_a, f_a)"""),
    ("code", "pd.DataFrame({'official baseline': baseline, 'AEGIS': aegis}).T"),
]


def main() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in CELLS]}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "AEGIS_T4_evaluation.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
