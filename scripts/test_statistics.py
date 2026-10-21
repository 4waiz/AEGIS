"""Uncertainty and diagnostics on the public test split (reporting only; nothing here selects a model).

Inputs: per-sample probabilities saved by scripts/measure.py (experiments/<name>/*.probs.npy, same order as
the public test split) and the experiments' tuned thresholds. Writes reports/test_statistics.json.

- Paired bootstrap (functions resampled with replacement, B=2000) 95% CIs for F1 differences.
- Effect of the validation-tuned thresholds on the test split (vs. a fixed 0.5).
- MAT: AEGIS vs. the zero-cost tag rule; whether the baseline's MAT misses come from 512-token truncation.
- GFLOPs decomposition and attention FLOPs omitted by the profiler.
Usage: python scripts/test_statistics.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aegis.data.dataset import extract_comments, load_split  # noqa: E402

EXP = ROOT / "experiments"
B = 2000
RNG = np.random.default_rng(0)
TAG = re.compile(r"\b(?:todo|fixme|hack|xxx)\b", re.I)


def f1_counts(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    tp = (pred & (y == 1)).sum(0)
    fp = (pred & (y == 0)).sum(0)
    fn = (~pred & (y == 1)).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def preds(name: str, file: str, thr) -> np.ndarray:
    return np.load(EXP / name / file) > np.asarray(thr)[None, :]


def bootstrap_diff(pa: np.ndarray, pb: np.ndarray, y: np.ndarray) -> dict:
    """CI of F1(a) - F1(b) for weakness, MAT and their mean, resampling functions jointly."""
    n = len(y)
    diffs = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        fa, fb = f1_counts(pa[idx], y[idx]), f1_counts(pb[idx], y[idx])
        d = fa - fb
        diffs.append([d[0], d[1], d.mean()])
    diffs = np.array(diffs)
    point = f1_counts(pa, y) - f1_counts(pb, y)
    out = {}
    for j, k in enumerate(["weakness", "mat", "mean"]):
        p = point.mean() if k == "mean" else point[j]
        lo, hi = np.percentile(diffs[:, j], [2.5, 97.5])
        out[k] = {"diff": round(float(p), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def main() -> None:
    test = load_split("test")
    y = test[["y_vuln", "y_mat"]].to_numpy()
    thr = lambda e: json.loads((EXP / e / "metrics.json").read_text(encoding="utf-8"))["best"]["thresholds"]  # noqa: E731
    P = {
        "baseline": preds("baseline", "measure_test_official.probs.npy", [0.5, 0.5]),
        "teacher": preds("teacher_unixcoder", "measure_test.probs.npy", thr("teacher_unixcoder")),
        "no_kd_L3": preds("student_unix_L3", "measure_test.probs.npy", thr("student_unix_L3")),
        "kd_L3_single_line": preds("kd_teacher_L3", "measure_test.probs.npy", thr("kd_teacher_L3")),
        "aegis": preds("kd_teacher_L3_mixed", "measure_test.probs.npy", [0.5, 0.5]),  # thresholds folded
    }
    res: dict = {"bootstrap_B": B, "f1": {k: [round(float(v), 4) for v in f1_counts(p, y)] for k, p in P.items()}}
    res["paired_bootstrap"] = {
        "aegis_minus_baseline": bootstrap_diff(P["aegis"], P["baseline"], y),
        "aegis_minus_no_kd_L3": bootstrap_diff(P["aegis"], P["no_kd_L3"], y),
        "aegis_minus_kd_L3_single_line": bootstrap_diff(P["aegis"], P["kd_L3_single_line"], y),
        "kd_L3_single_line_minus_no_kd_L3": bootstrap_diff(P["kd_L3_single_line"], P["no_kd_L3"], y),
        "teacher_minus_aegis": bootstrap_diff(P["teacher"], P["aegis"], y),
    }

    # Threshold effect on the test split (tuned on internal validation vs. fixed 0.5)
    te = {}
    for e, file in (("teacher_unixcoder", "measure_test.probs.npy"), ("student_unix_L3", "measure_test.probs.npy"),
                    ("kd_teacher_L3", "measure_test.probs.npy")):
        p = np.load(EXP / e / file)
        te[e] = {"tuned": float(f1_counts(p > np.asarray(thr(e)), y).mean()), "at_0.5": float(f1_counts(p > 0.5, y).mean())}
    folded = np.asarray(thr("kd_teacher_L3_mixed"))
    p = np.clip(np.load(EXP / "kd_teacher_L3_mixed" / "measure_test.probs.npy"), 1e-7, 1 - 1e-7)
    unfolded = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + np.log(folded / (1 - folded)))))
    te["aegis"] = {"tuned": float(f1_counts(unfolded > folded, y).mean()), "at_0.5": float(f1_counts(unfolded > 0.5, y).mean())}
    for v in te.values():
        v["gain"] = round(v["tuned"] - v["at_0.5"], 4)
        v["tuned"], v["at_0.5"] = round(v["tuned"], 4), round(v["at_0.5"], 4)
    res["threshold_effect_test"] = te
    # Boundary cases where fp32 vs fp16 sigmoid could flip a decision (|p - 0.5| < 1e-3)
    res["near_boundary_probs_aegis"] = int((np.abs(np.load(EXP / "kd_teacher_L3_mixed" / "measure_test.probs.npy") - 0.5) < 1e-3).sum())

    # MAT: AEGIS vs the tag rule
    comments = test.Function.map(lambda s: " ".join(extract_comments(s)))
    rule = comments.str.contains(TAG).to_numpy()
    ym = y[:, 1] == 1
    def counts(pr):
        return {"tp": int((pr & ym).sum()), "fp": int((pr & ~ym).sum()), "fn": int((~pr & ym).sum())}
    res["mat_rule_vs_aegis"] = {"rule": counts(rule), "aegis": counts(P["aegis"][:, 1]),
                                "agree": int((rule == P["aegis"][:, 1]).sum()), "n": len(y)}

    # Baseline MAT misses vs truncation (CodeBERT BPE on raw text, max 512 incl. <s>, </s>)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    trunc = []
    for code, has_tag in zip(test.Function, rule):
        if not has_tag:
            trunc.append(False)
            continue
        m = TAG.search(" ".join(extract_comments(code)))
        pos = code.lower().find(m.group(0).lower())
        n_before = len(tok(code[:pos], add_special_tokens=False)["input_ids"]) if pos >= 0 else 0
        trunc.append(n_before + 1 >= 511)
    trunc = np.array(trunc)
    base_mat = P["baseline"][:, 1]
    tagged_pos = ym & rule
    res["baseline_mat_truncation"] = {
        "tagged_positives": int(tagged_pos.sum()),
        "tagged_positives_with_tag_beyond_512_tokens": int((tagged_pos & trunc).sum()),
        "baseline_fn_tagged": int((tagged_pos & ~base_mat).sum()),
        "baseline_fn_tagged_and_truncated": int((tagged_pos & ~base_mat & trunc).sum()),
        "aegis_fn_tagged": int((tagged_pos & ~P["aegis"][:, 1]).sum()),
    }

    # GFLOPs decomposition and attention share
    off = json.loads((EXP / "baseline" / "measure_test_official.json").read_text(encoding="utf-8"))
    srt = json.loads((EXP / "baseline" / "measure_test_sorted.json").read_text(encoding="utf-8"))
    fin = json.loads((EXP / "kd_teacher_L3_mixed" / "measure_test.json").read_text(encoding="utf-8"))
    res["gflops_decomposition"] = {
        "padding_factor": round(off["avg_flops"] / srt["avg_flops"], 2),
        "model_and_input_factor": round(srt["avg_flops"] / fin["avg_flops"], 2),
        "total_factor": round(off["avg_flops"] / fin["avg_flops"], 2),
    }
    res["attention_share_not_counted"] = {
        k: round(r["extra"]["analytic"]["attention_gflops"] / r["avg_flops"], 3)
        for k, r in (("baseline_official", off), ("aegis", fin))
    }
    (ROOT / "reports" / "test_statistics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
