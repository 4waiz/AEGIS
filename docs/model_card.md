---
license: apache-2.0
base_model: microsoft/unixcoder-base
datasets:
- NLBSE/nlbse27-code-comment-classification
language:
- code
pipeline_tag: text-classification
tags:
- vulnerability-detection
- self-admitted-technical-debt
- multitask
- knowledge-distillation
- nlbse
---

# AEGIS: efficiency-guided multitask classifier for the NLBSE'27 tool competition

A single model that reads a C/C++ function and predicts two labels, in the competition's order
`[weakness, MAT]`:

- **weakness**: the function is associated with a weakness or vulnerability (Devign / Big-Vul / CWE-heuristic labels);
- **MAT**: the function carries a self-admitted technical debt tag (TODO, FIXME, XXX, HACK) according to the MAT heuristic.

It is a 3-layer `RobertaForSequenceClassification` (UniXcoder vocabulary, hidden size 768, 62M parameters of which
40M are embeddings). It was distilled from a fine-tuned 12-layer UniXcoder teacher.
Code, experiment logs and the paper: https://github.com/4waiz/AEGIS

## How to use

The per-task decision thresholds tuned on validation data are folded into the bias of the output layer, so
**predict a label when its sigmoid probability is above 0.5**. That is exactly what the official competition
notebook does.

```python
import re, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "<this-repo-id>"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

code = "int f(char *s) {\n  char b[8];\n  strcpy(b, s); // TODO: check length\n  return 0;\n}"
text = re.sub(r"\s+", " ", code).strip()  # recommended input normalization (see below)
enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    probs = torch.sigmoid(model(**enc).logits)[0]
print({"weakness": int(probs[0] > 0.5), "MAT": int(probs[1] > 0.5)})
```

**Input.** Collapsing every whitespace run to a single space is recommended: it shortens inputs by about a third.
The model was trained on a 50/50 mix of normalized and raw (clang-formatted) functions, so raw input also works.
For batch inference, sorting functions by length and padding each batch only to its own maximum removes most of
the padding cost without changing predictions.

## Results (NLBSE'27 public test split, 30,655 functions, official measurement loop)

| Model | Weakness F1 | MAT F1 | Mean F1 | Runtime (s)* | GFLOPs | Score* |
|---|---|---|---|---|---|---|
| CodeBERT baseline (official procedure) | 0.939 | 0.881 | 0.910 | 71.0 | 2,666,498 | 0.754 |
| **This model** (normalized input, length-sorted batches) | 0.932 | 0.931 | **0.931** | 6.2 | 206,123 | **0.944** |
| This model in the unmodified official notebook (raw input, padding to 512) | 0.935 | 0.918 | 0.927 | 19.0 | 666,661 | 0.908 |

\*Runtime and score were measured on an RTX 5080 Laptop GPU in one session. The competition ranks on a Colab T4,
which is slower; the organizers report a score of 0.729 for the baseline on T4. GFLOPs follow the official
`torch.profiler` count and do not depend on the hardware.

## Training

- **Data:** `NLBSE/nlbse27-code-comment-classification`, revision `feca19b90d1cc9fcf8ba5d6e693c4a94e249585c`.
  Only the `Function` column is used. 90% of the official training split was used for fitting and 10% (stratified)
  for model selection and threshold tuning. The public test split was not used for training or tuning.
- **Teacher:** UniXcoder-base, 12 layers, 3 epochs, lr 2e-5, batch 32, bf16.
- **Student:** the teacher's embeddings and first 3 layers; loss 0.5 x BCE(labels) + 0.5 x T^2 x BCE(student/T, sigmoid(teacher/T))
  with T = 2, averaged over the two tasks; 4 epochs, lr 5e-5, batch 32, seed 42.

## Limitations

- The labels are partly heuristic. MAT is a keyword rule and about 12% of MAT positives have no tag in the released
  function text. Big-Vul and Devign vulnerability labels are known to be noisy. High scores mean agreement with
  these annotations, **not** reliable vulnerability detection. Do not use this model as a security tool.
- The competition split is random. 18% of test functions have a structural clone in training, so scores on unseen
  projects are likely lower. Weakness F1 varies strongly by project (0.955 on the Linux kernel, 0.707 on qemu).

## License

Apache-2.0, following the base model `microsoft/unixcoder-base`. The training data is distributed by the NLBSE'27
organizers under their own terms.
