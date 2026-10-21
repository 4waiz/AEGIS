"""Error analysis of a trained model on the internal validation split (or on public-test
predictions saved by scripts/measure.py for the final model only).

Usage:
  python scripts/error_analysis.py <experiment_name>                       # internal val
  python scripts/error_analysis.py <experiment_name> --test <probs.npy>    # final model, public test
Writes reports/error_analysis_<experiment>[_test].md and .json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.data.dataset import extract_comments, function_hash, load_internal_splits  # noqa: E402
from aegis.paths import EXPERIMENTS_DIR, REPORTS_DIR  # noqa: E402

SATD = re.compile(r"\b(?:todo|fixme|hack|xxx)\b", re.I)
# Project name -> MADE-WIC source collection. Derived from MADE-WIC Table 2 (arXiv 2408.05163): OSPR has
# Linux Kernel, Chromium and Mozilla Firefox (no original vulnerability labels, so weakness = W heuristic);
# Devign has FFmpeg and Qemu; Big-Vul has the other projects (its Chromium/Linux appear as "Chrome"/"linux"
# here). Consistent with ~35% retention per project after the organizers' filtering. FFmpeg is in both
# Devign and Big-Vul; it is grouped with Devign, which contributes most of its functions.
SOURCE = {"Linux Kernel": "OSPR", "Chromium": "OSPR", "Mozilla Firefox": "OSPR",
          "qemu": "Devign", "FFmpeg": "Devign"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_audit import abstract_hash  # noqa: E402


def rates(df: pd.DataFrame, task: str) -> dict:
    y, p = df[f"y_{task}"].to_numpy(), df[f"pred_{task}"].to_numpy()
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(n=len(df), pos=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn, f1=round(f1, 4),
                fpr=round(fp / max(1, fp + tn), 4), fnr=round(fn / max(1, fn + tp), 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--test", help="probs .npy from scripts/measure.py on the public test split")
    ap.add_argument("--threshold", type=float,
                    help="override thresholds, e.g. 0.5 for an exported model with folded thresholds")
    args = ap.parse_args()
    exp = EXPERIMENTS_DIR / args.experiment
    metrics = json.loads((exp / "metrics.json").read_text(encoding="utf-8"))
    thr = np.asarray([args.threshold] * 2 if args.threshold is not None else metrics["best"]["thresholds"])
    splits = load_internal_splits()
    if args.test:
        df = splits["test"].copy()
        probs = np.load(args.test)
        suffix = "_test"
    else:
        z = np.load(exp / "val_logits_best.npz")
        df = splits["val"][splits["val"].row_id.isin(z["row_id"])].copy()  # smoke runs use a subsample
        order = pd.Series(np.arange(len(z["row_id"])), index=z["row_id"])
        probs = 1 / (1 + np.exp(-z["logits"][order.loc[df.row_id].to_numpy()]))
        suffix = ""
    df["p_vuln"], df["p_mat"] = probs[:, 0], probs[:, 1]
    df["pred_vuln"] = (df.p_vuln > thr[0]).astype(int)
    df["pred_mat"] = (df.p_mat > thr[1]).astype(int)

    from transformers import AutoTokenizer

    ckpt = Path(metrics["checkpoint"])
    if ckpt.exists():
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        aegis_cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8")).get("aegis", {})
    else:  # checkpoint deleted: the tokenizer and preprocessing are recorded in the experiment config
        mcfg = metrics["config"]["model"]
        tok = AutoTokenizer.from_pretrained(mcfg.get("tokenizer") or mcfg["init"])
        aegis_cfg = metrics["config"]["data"]
    from aegis.data.preprocess import normalize

    df["n_tok"] = [len(x) + 2 for x in tok([normalize(s, aegis_cfg.get("normalize", "raw")) for s in df.Function],
                                            add_special_tokens=False)["input_ids"]]
    max_len = aegis_cfg.get("max_length", 512)
    comments = df.Function.map(lambda s: " ".join(extract_comments(s)))
    df["has_comment"] = comments.str.len() > 0
    df["satd_kw"] = comments.str.contains(SATD)
    fit = splits["fit"]
    fit_exact = set(fit.Function.map(function_hash))
    fit_abs = set(fit.Function.map(abstract_hash))
    df["exact_dup_in_fit"] = df.Function.map(function_hash).isin(fit_exact)
    df["clone_in_fit"] = df.Function.map(abstract_hash).isin(fit_abs)

    out: dict = {"experiment": args.experiment, "split": "test" if args.test else "internal_val",
                 "thresholds": thr.tolist(), "overall": {t: rates(df, t) for t in ("vuln", "mat")}}
    edges = sorted({e for e in (0, 64, 128, 256) if e < max_len} | {max_len}) + [10**6]
    labels = [f"{a + 1}-{b}" for a, b in zip(edges[:-2], edges[1:-1])] + [f">{max_len} (truncated)"]
    groups = {
        "joint_label": df.joint.map({0: "[0,0]", 1: "[0,1]", 2: "[1,0]", 3: "[1,1]"}),
        "length_bucket": pd.cut(df.n_tok, edges, labels=labels),
        "has_comment": df.has_comment, "satd_keyword_in_comment": df.satd_kw,
        "exact_dup_in_fit": df.exact_dup_in_fit, "type2_clone_in_fit": df.clone_in_fit,
        "project": df.Projectname,
        "source_dataset": df.Projectname.map(SOURCE).fillna("Big-Vul"),
    }
    for gname, g in groups.items():
        out[gname] = {str(k): {t: rates(sub, t) for t in ("vuln", "mat")} for k, sub in df.groupby(g, observed=True)}
    # MAT error anatomy
    fn_mat = df[(df.y_mat == 1) & (df.pred_mat == 0)]
    fp_mat = df[(df.y_mat == 0) & (df.pred_mat == 1)]
    out["mat_fn_with_keyword"] = int(fn_mat.satd_kw.sum())
    out["mat_fn_without_keyword"] = int((~fn_mat.satd_kw).sum())
    out["mat_fp_with_keyword"] = int(fp_mat.satd_kw.sum())
    out["mat_fp_examples"] = [c[:300] for c in comments[fp_mat.index][:5]]
    out["mat_fn_keyword_examples"] = [c[:300] for c in comments[fn_mat.index][fn_mat.satd_kw][:5]]

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"error_analysis_{args.experiment}{suffix}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    lines = [f"# Error analysis: {args.experiment} ({out['split']})", "",
             f"Thresholds (from internal validation): {thr.tolist()}", ""]
    for gname in ["overall"] + list(groups):
        lines.append(f"## {gname}")
        lines.append("| group | task | n | pos | F1 | FPR | FNR | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        items = {"all": out["overall"]} if gname == "overall" else out[gname]
        for k, v in items.items():
            for t in ("vuln", "mat"):
                r = v[t]
                lines.append(f"| {k} | {t} | {r['n']} | {r['pos']} | {r['f1']} | {r['fpr']} | {r['fnr']} | {r['fp']} | {r['fn']} |")
        lines.append("")
    lines += [f"MAT false negatives with a SATD keyword in a comment: {out['mat_fn_with_keyword']}",
              f"MAT false negatives without any SATD keyword: {out['mat_fn_without_keyword']}",
              f"MAT false positives with a SATD keyword: {out['mat_fp_with_keyword']}"]
    (REPORTS_DIR / f"error_analysis_{args.experiment}{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
