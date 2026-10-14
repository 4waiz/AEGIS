"""Verbatim replica of the official NLBSE'27 evaluation notebook (cells 1-4).

Copied from CodeBERT_baseline.ipynb (github.com/nlbse2027/code-comment-classification @310c2b0).
Only changes: (1) data is read from the pinned local parquet snapshot instead of load_dataset
(identical content, revision feca19b), (2) ``torch.cuda.amp.autocast()`` is spelled
``torch.autocast("cuda")`` (same behavior, the old alias is deprecated), (3) the model
path can be passed on the command line, (4) results are also written to JSON.

Usage: python scripts/official_notebook_replica.py [model_id_or_path] [out.json]
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer

sys.path.insert(0, "src")
from aegis.data.dataset import load_split  # noqa: E402
from aegis.envinfo import environment  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "NLBSE/nlbse27-code-comment-classification-baseline"
OUT = sys.argv[2] if len(sys.argv) > 2 else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = RobertaTokenizer.from_pretrained(MODEL)

total_flops = 0
total_time = 0
scores = []

model = RobertaForSequenceClassification.from_pretrained(MODEL)
model.to(device)
model.eval()

predictions = []

test_data = load_split("test")

# Define batch size
batch_size = 600  # Adjust based on your GPU memory

# Tokenize all data at once
tokenized_data = tokenizer(
    test_data["Function"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

# Create a DataLoader for batch processing
dataset = TensorDataset(tokenized_data["input_ids"], tokenized_data["attention_mask"])
data_loader = DataLoader(dataset, batch_size=batch_size)

predictions = []
pbar = tqdm(total=len(test_data), desc="Predicting...")

for batch in data_loader:
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    # Perform forward pass and measure performance
    begin = time.time()
    with torch.no_grad():
        with torch.autocast("cuda"):
            with torch.profiler.profile(with_flops=True) as p:
                for _ in range(2):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    total = time.time() - begin
    total_time = total_time + total

    # Calculate FLOPs for the batch
    total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)

    logits = outputs.logits
    probs = torch.sigmoid(logits)
    threshold = 0.5
    batch_predictions = (probs > threshold).int().cpu().numpy().tolist()
    predictions.extend(batch_predictions)

    del input_ids, attention_mask, outputs, logits, probs
    torch.cuda.empty_cache()
    pbar.update(len(batch_predictions))

pbar.close()

labels = np.array(test_data["Label"].tolist())
labels_name = ["Vulnerability", "MAT"]
num_classes = len(labels_name)
predictions = np.array(predictions)

for class_idx in range(num_classes):
    tp = np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 1))
    fp = np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 0))
    tn = np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 0))
    fn = np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    scores.append({"cat": labels_name[class_idx], "precision": precision, "recall": recall, "f1": f1,
                   "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})

print("----------------RESULTS--------------------")
print("Compute in GFLOPs:", total_flops / 2)
print("Avg runtime in seconds:", total_time / 2)
scores = pd.DataFrame(scores)
print(scores)

max_avg_runtime = 1000
max_avg_flops = 3000000


def score(avg_f1, avg_runtime, avg_flops):
    return (
        0.6 * avg_f1
        + 0.2 * max((max_avg_runtime - avg_runtime) / max_avg_runtime, 0)
        + 0.2 * max((max_avg_flops - avg_flops) / max_avg_flops, 0)
    )


avg_f1 = scores.f1.mean()
avg_runtime = total_time / 2
avg_flops = total_flops / 2
final = score(avg_f1, avg_runtime, avg_flops)
print("avg P/R/F1", scores.precision.mean(), scores.recall.mean(), avg_f1)
print("score", final, "rounded", round(final, 2))

if OUT:
    with open(OUT, "w") as f:
        json.dump({
            "model": MODEL, "procedure": "official_notebook_replica",
            "padded_length": int(tokenized_data["input_ids"].shape[1]),
            "scores": scores.to_dict(orient="records"), "avg_f1": float(avg_f1),
            "avg_runtime": avg_runtime, "avg_flops": avg_flops, "score": final,
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "env": environment(),
        }, f, indent=2, default=float)
