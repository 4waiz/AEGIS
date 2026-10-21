"""Paper figure: effect of encoder depth on internal-validation mean F1 (top) and on the projected
competition score (bottom), with and without multitask distillation.

Inputs: experiments/<name>/metrics.json and measure_val.json (official loop on internal validation).
Output: figures/depth_tradeoff.{pdf,png}
Usage: python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from leaderboard import test_over_val  # noqa: E402

from aegis.evaluation.official import submission_score  # noqa: E402
from aegis.paths import FIGURES_DIR  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NO_KD, KD = "#2a78d6", "#eb6834"  # validated categorical slots 1-2
LAYERS = [1, 2, 3, 4, 6, 12]
RUNS_NO_KD = {1: ["student_unix_L1"], 2: ["student_unix_L2"],
              3: ["student_unix_L3", "student_unix_L3_s43", "student_unix_L3_s44"],
              4: ["student_unix_L4"], 6: ["student_unix_L6"], 12: ["teacher_unixcoder"]}
RUNS_KD = {2: ["kd_teacher_L2"], 3: ["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"], 4: ["kd_teacher_L4"]}


def point(names: list[str]) -> tuple[float, float]:
    f1s, scores = [], []
    for n in names:
        b = json.loads((ROOT / f"experiments/{n}/metrics.json").read_text(encoding="utf-8"))["best"]
        r = json.loads((ROOT / f"experiments/{n}/measure_val.json").read_text(encoding="utf-8"))
        k = test_over_val(r)
        f1s.append(b["val_mean_f1_tuned"])
        scores.append(submission_score(b["val_mean_f1_tuned"], r["avg_runtime"] * k, r["avg_flops"] * k))
    return float(np.mean(f1s)), float(np.mean(scores))


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True)
    xpos = {L: i for i, L in enumerate(LAYERS)}
    for runs, color, label, marker in ((RUNS_NO_KD, NO_KD, "no distillation", "o"), (RUNS_KD, KD, "with distillation", "s")):
        Ls = sorted(runs)
        pts = [point(runs[L]) for L in Ls]
        xs = [xpos[L] for L in Ls]
        for ax, j in ((ax1, 0), (ax2, 1)):
            ax.plot(xs, [p[j] for p in pts], color=color, linewidth=2, marker=marker, markersize=5,
                    markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
    best = point(RUNS_KD[3])
    ax2.annotate(f"best: 3 layers + KD ({best[1]:.3f})", (xpos[3], best[1]), xytext=(xpos[1] - 0.6, 0.905),
                 textcoords="data", fontsize=7, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkB=4))
    ax1.set_ylabel("mean F1\n(internal val.)")
    ax2.set_ylabel("projected score")
    ax2.set_xticks(range(len(LAYERS)), [str(L) for L in LAYERS])
    ax2.set_xlabel("encoder layers (12 = teacher)")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK2)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower right", fontsize=7)
    fig.align_ylabels((ax1, ax2))
    fig.tight_layout(h_pad=0.6)
    FIGURES_DIR.mkdir(exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"depth_tradeoff.{fmt}", dpi=250)
    print("wrote figures/depth_tradeoff.{pdf,png}")


if __name__ == "__main__":
    main()
