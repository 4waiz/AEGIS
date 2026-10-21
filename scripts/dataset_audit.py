"""Phase 1 dataset forensics for the NLBSE'27 multitask dataset.

Writes reports/label_distribution.csv, reports/dataset_stats.json, reports/data_quality.md
and figures/dataset_*.pdf/png. All numbers in research/dataset_audit.md come from here.

Usage: python scripts/dataset_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.data.dataset import extract_comments, function_hash, load_internal_splits, load_split  # noqa: E402
from aegis.paths import DATASET_REVISION, FIGURES_DIR, REPORTS_DIR  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SATD_WORD = re.compile(r"\b(?:todo|fixme|hack|xxx)\b", re.I)
SATD_SUB = re.compile(r"todo|fixme|hack|xxx", re.I)
C_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue", "return",
    "goto", "struct", "union", "enum", "typedef", "static", "const", "volatile", "extern", "inline",
    "sizeof", "void", "int", "char", "short", "long", "float", "double", "signed", "unsigned",
    "bool", "auto", "register", "class", "public", "private", "protected", "template", "typename",
    "namespace", "new", "delete", "this", "virtual", "operator", "nullptr", "NULL", "true", "false",
}
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d[\w.]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\S")


def abstract_hash(code: str) -> str:
    """Identifier/literal-abstracted hash (type-2 clone proxy), comments removed."""
    code = re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S)
    toks = []
    for t in TOKEN_RE.findall(code):
        if t[0].isalpha() or t[0] == "_":
            toks.append(t if t in C_KEYWORDS else "ID")
        elif t[0].isdigit():
            toks.append("NUM")
        elif t[0] in "\"'":
            toks.append("STR")
        else:
            toks.append(t)
    return function_hash(" ".join(toks))


def prf(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r), 4) if p + r else 0.0,
                tp=tp, fp=fp, fn=fn)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    train, test = load_split("train"), load_split("test")
    internal = load_internal_splits()
    stats: dict = {"dataset_revision": DATASET_REVISION}

    # 1-2. Label distributions -------------------------------------------------------------
    rows = []
    names = {0: "[0,0]", 1: "[0,1]", 2: "[1,0]", 3: "[1,1]"}
    for split, df in [("train", train), ("test_public", test), ("internal_fit", internal["fit"]),
                      ("internal_val", internal["val"])]:
        counts = df["joint"].value_counts().reindex(range(4), fill_value=0)
        for j in range(4):
            rows.append(dict(split=split, joint_label=names[j], count=int(counts[j]),
                             fraction=round(counts[j] / len(df), 5)))
        stats[f"{split}_n"] = len(df)
        stats[f"{split}_vuln_prev"] = round(df.y_vuln.mean(), 5)
        stats[f"{split}_mat_prev"] = round(df.y_mat.mean(), 5)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "label_distribution.csv", index=False)
    stats["label_correlation_phi_train"] = round(float(np.corrcoef(train.y_vuln, train.y_mat)[0, 1]), 4)
    stats["p_vuln_given_mat_train"] = round(train.loc[train.y_mat == 1, "y_vuln"].mean(), 4)
    stats["p_vuln_given_not_mat_train"] = round(train.loc[train.y_mat == 0, "y_vuln"].mean(), 4)

    # 3. Token lengths (CodeBERT BPE) ----------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    for split, df in [("train", train), ("test", test)]:
        lens = np.array([len(x) for x in tok(df["Function"].tolist(), add_special_tokens=True)["input_ids"]])
        df["n_tok"] = lens
        q = np.percentile(lens, [5, 25, 50, 75, 90, 95, 99])
        stats[f"{split}_tok"] = dict(mean=round(lens.mean(), 1), max=int(lens.max()), p5=q[0], p25=q[1],
                                     p50=q[2], p75=q[3], p90=q[4], p95=q[5], p99=q[6],
                                     frac_gt_128=round((lens > 128).mean(), 4),
                                     frac_gt_256=round((lens > 256).mean(), 4),
                                     frac_gt_512=round((lens > 512).mean(), 4))
    # Official notebook pads every test sequence to min(max_len, 512); FLOP proportionality check.
    stats["test_padding_efficiency_official"] = round(np.minimum(test.n_tok, 512).mean() / min(test.n_tok.max(), 512), 4)
    stats["vuln_prev_by_length_quartile_train"] = (
        train.groupby(pd.qcut(train.n_tok, 4, labels=["Q1", "Q2", "Q3", "Q4"]), observed=True)["y_vuln"]
        .mean().round(4).to_dict())

    # 4. Comments and SATD keywords -------------------------------------------------------------
    for split, df in [("train", train), ("test", test)]:
        comments = df["Function"].map(lambda s: " ".join(extract_comments(s)))
        has_c = comments.str.len().gt(0).to_numpy()
        y = df.y_mat.to_numpy()
        word = comments.str.contains(SATD_WORD).to_numpy()
        sub = comments.str.contains(SATD_SUB).to_numpy()
        stats[f"{split}_frac_with_comment"] = round(has_c.mean(), 4)
        stats[f"{split}_mat_given_comment"] = round(y[has_c].mean(), 4)
        stats[f"{split}_mat_pos_without_comment"] = int((y[~has_c] == 1).sum())
        stats[f"{split}_mat_pos_without_keyword"] = int(((y == 1) & ~sub).sum())
        stats[f"{split}_rule_satd_word_in_comments"] = prf(word, y)
        stats[f"{split}_rule_satd_substring_in_comments"] = prf(sub, y)
        stats[f"{split}_rule_satd_substring_anywhere"] = prf(df["Function"].str.contains(SATD_SUB).to_numpy(), y)
        for k in ["todo", "fixme", "hack", "xxx"]:
            m = comments.str.contains(re.compile(rf"\b{k}\b", re.I)).to_numpy()
            stats[f"{split}_kw_{k}"] = dict(n=int(m.sum()), p_mat=round(y[m].mean(), 4) if m.any() else None)
        df["has_comment"] = has_c

    # 5. Language (file extension) and 6. projects ------------------------------------------------
    ext = train.Filepath.str.extract(r"(\.[A-Za-z0-9+]+)$")[0].str.lower().fillna("<none>")
    stats["train_extensions"] = ext.value_counts().head(12).to_dict()
    proj = train.groupby("Projectname").agg(n=("y_vuln", "size"), vuln=("y_vuln", "mean"), mat=("y_mat", "mean"))
    proj = proj.sort_values("n", ascending=False).round(4)
    proj.to_csv(REPORTS_DIR / "project_distribution.csv")
    stats["n_projects_train"] = int(train.Projectname.nunique())
    stats["projects_only_in_test"] = sorted(set(test.Projectname) - set(train.Projectname))

    # 7-8. Duplicates and train/test similarity ---------------------------------------------------
    for df in (train, test):
        df["h_exact"] = df.Function.map(function_hash)
        df["h_abs"] = df.Function.map(abstract_hash)
    for key, label in [("h_exact", "exact_ws_normalized"), ("h_abs", "identifier_abstracted")]:
        stats[f"train_dup_rows_{label}"] = int(train[key].duplicated().sum())
        in_train = test[key].isin(set(train[key]))
        stats[f"test_rows_in_train_{label}"] = int(in_train.sum())
        majority = train.groupby(key)[["y_vuln", "y_mat"]].mean()
        ov = test[in_train]
        agree_v = ((majority.loc[ov[key], "y_vuln"].to_numpy() >= 0.5).astype(int) == ov.y_vuln.to_numpy()).mean()
        stats[f"test_overlap_vuln_label_agreement_{label}"] = round(float(agree_v), 4) if len(ov) else None
        grp = train.groupby(key).agg(n=("y_vuln", "size"), vn=("y_vuln", "nunique"), mn=("y_mat", "nunique"))
        stats[f"train_dup_groups_{label}"] = int((grp.n > 1).sum())
        stats[f"train_dup_groups_conflicting_vuln_{label}"] = int((grp.vn > 1).sum())
        stats[f"train_dup_groups_conflicting_mat_{label}"] = int((grp.mn > 1).sum())

    # 9. Metadata leakage probe: does file identity predict the label? (metadata is NOT a model input)
    file_rate = train.groupby("Filepath")["y_vuln"].agg(["mean", "size"])
    seen = test.Filepath.isin(file_rate.index)
    pred = np.zeros(len(test), dtype=bool)
    pred[seen.to_numpy()] = file_rate.loc[test.Filepath[seen], "mean"].to_numpy() >= 0.5
    stats["test_rows_with_filepath_in_train"] = int(seen.sum())
    stats["metadata_probe_filepath_majority_vuln"] = prf(pred, test.y_vuln.to_numpy())
    proj_rate = train.groupby("Projectname")["y_vuln"].mean()
    stats["metadata_probe_project_majority_vuln"] = prf(
        (test.Projectname.map(proj_rate).fillna(0) >= 0.5).to_numpy(), test.y_vuln.to_numpy())
    stats["length_probe_vuln_f1_best_threshold"] = max(
        (prf((test.n_tok >= t).to_numpy(), test.y_vuln.to_numpy())["f1"], int(t))
        for t in np.percentile(test.n_tok, np.arange(5, 100, 5)))

    (REPORTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    # Figures -------------------------------------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    bins = np.arange(0, 1025, 16)
    for (lab, sub), color, name in zip([(0, train[train.y_vuln == 0]), (1, train[train.y_vuln == 1])],
                                       SERIES, ["not weakness", "weakness"]):
        ax.hist(np.minimum(sub.n_tok, 1024), bins=bins, color=color, alpha=0.75, label=name,
                edgecolor="white", linewidth=0.3)
    ax.axvline(512, color=INK2, linewidth=1, linestyle="--")
    ax.text(520, ax.get_ylim()[1] * 0.9, "512", color=INK2, fontsize=7)
    ax.set_xlabel("CodeBERT tokens per function (train)")
    ax.set_ylabel("functions")
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    ax = axes[1]
    labels_ = ["[0,0]", "[1,0]", "[1,1]", "[0,1]"]
    vals = [train.joint.eq(j).mean() * 100 for j in (0, 2, 3, 1)]
    bars = ax.bar(labels_, vals, color=SERIES[0], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", fontsize=7, color=INK)
    ax.set_ylabel("% of train functions")
    ax.set_xlabel("joint label [weakness, MAT]")
    style(ax)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"dataset_overview.{fmt}", dpi=200)
    plt.close(fig)

    # data_quality.md --------------------------------------------------------------------------
    s = stats
    md = [
        "# Data quality report (auto-generated by scripts/dataset_audit.py)",
        f"Dataset revision: `{DATASET_REVISION}`",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Train / public test size | {s['train_n']} / {s['test_public_n']} |",
        f"| Weakness prevalence (train / test) | {s['train_vuln_prev']} / {s['test_public_vuln_prev']} |",
        f"| MAT prevalence (train / test) | {s['train_mat_prev']} / {s['test_public_mat_prev']} |",
        f"| Exact duplicate rows inside train (whitespace-normalized) | {s['train_dup_rows_exact_ws_normalized']} |",
        f"| Test rows with an exact duplicate in train | {s['test_rows_in_train_exact_ws_normalized']} |",
        f"| Test rows with an identifier-abstracted clone in train | {s['test_rows_in_train_identifier_abstracted']} |",
        f"| Train duplicate groups with conflicting weakness label (exact) | {s['train_dup_groups_conflicting_vuln_exact_ws_normalized']} |",
        f"| Train duplicate groups with conflicting MAT label (exact) | {s['train_dup_groups_conflicting_mat_exact_ws_normalized']} |",
        f"| Functions with at least one comment (train) | {s['train_frac_with_comment']} |",
        f"| MAT-positive train functions with no comment in `Function` | {s['train_mat_pos_without_comment']} |",
        f"| SATD keyword rule in comments, test (P/R/F1) | {s['test_rule_satd_word_in_comments']} |",
        f"| CodeBERT tokens > 512 (train / test) | {s['train_tok']['frac_gt_512']} / {s['test_tok']['frac_gt_512']} |",
        f"| Test padding efficiency under official global padding | {s['test_padding_efficiency_official']} |",
        f"| Filepath-majority probe for weakness on test (metadata, not used) | {s['metadata_probe_filepath_majority_vuln']} |",
    ]
    (REPORTS_DIR / "data_quality.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
