"""Accuracy-efficiency frontier figure from official-loop measurements on the internal validation
split (experiments/*/measure_val.json).

Usage: python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.paths import EXPERIMENTS_DIR, FIGURES_DIR  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"teacher": "#2a78d6", "student": "#eb6834", "kd": "#1baf7a"}


def family(name: str) -> str:
    if name.startswith("kd_"):
        return "kd"
    if name.startswith("teacher"):
        return "teacher"
    return "student"


def main() -> None:
    pts = []
    for f in sorted(EXPERIMENTS_DIR.glob("*/measure_val.json")):
        r = json.loads(f.read_text())
        pts.append((f.parent.name, r["avg_flops"], r["avg_f1"], r["avg_runtime"]))
    if not pts:
        print("no measure_val.json files yet")
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK2})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    fam_seen = set()
    for name, g, f1, _ in pts:
        fam = family(name)
        ax.scatter(g / 1e3, f1, s=28, color=SERIES[fam], edgecolor="white", linewidth=0.8, zorder=3,
                   label={"teacher": "teacher (12 layers)", "student": "student, no KD", "kd": "student + KD"}[fam]
                   if fam not in fam_seen else None)
        fam_seen.add(fam)
        ax.annotate(name.replace("student_unix_", "").replace("kd_", "kd-").replace("teacher_unixcoder", "T"),
                    (g / 1e3, f1), xytext=(3, 3), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("official GFLOPs on internal validation (thousands, log)")
    ax.set_ylabel("mean F1 (internal validation)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"frontier_val.{fmt}", dpi=220)
    print("\n".join(f"{n:28s} GFLOPs {g:12.0f}  F1 {f:.4f}  runtime {t:7.2f}s" for n, g, f, t in pts))


if __name__ == "__main__":
    main()
