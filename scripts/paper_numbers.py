"""Generate paper/numbers.tex (LaTeX macros) from experiment outputs, so that every number in the
paper is traceable to a file produced by the code. Never edit numbers.tex by hand.

Usage: python scripts/paper_numbers.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aegis.evaluation.official import submission_score  # noqa: E402

EXP, REP = ROOT / "experiments", ROOT / "reports"
OFFICIAL_T4 = {"f1": 0.9101860725779012, "runtime": 198.93055331707, "gflops": 2666509.7131617293}


def load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def main() -> None:
    m: dict[str, str] = {}
    stats = load("reports/dataset_stats.json")
    tb = pd.read_csv(REP / "token_budget.csv")
    test_tb = tb[tb.split == "test"].set_index(["tokenizer", "normalize"])

    # Baseline (organizers' printed T4 numbers + our local measurements)
    m["BaseMeanFone"] = f"{OFFICIAL_T4['f1']:.3f}"
    m["BaseGflopsM"] = f"{OFFICIAL_T4['gflops'] / 1e6:.2f}"
    m["BaseScoreOfficial"] = f"{submission_score(**{'avg_f1': OFFICIAL_T4['f1'], 'avg_runtime': OFFICIAL_T4['runtime'], 'avg_flops': OFFICIAL_T4['gflops']}):.3f}"
    b_off = load("experiments/baseline/measure_test_official.json")
    b_sort = load("experiments/baseline/measure_test_sorted.json")
    m["BaseScoreLocal"] = f"{b_off['score']:.3f}"
    m["BaseSortedScore"] = f"{b_sort['score']:.3f}"
    m["BaseSortedGflopsM"] = f"{b_sort['avg_flops'] / 1e6:.2f}"

    # Data
    m["TrainN"] = f"{stats['train_n']:,}".replace(",", "{,}")
    m["TestN"] = f"{stats['test_public_n']:,}".replace(",", "{,}")
    m["ValN"] = f"{stats['internal_val_n']:,}".replace(",", "{,}")
    m["VulnPrev"] = f"{100 * stats['train_vuln_prev']:.1f}"
    m["MatPrev"] = f"{100 * stats['train_mat_prev']:.2f}"
    rule = stats["test_rule_satd_word_in_comments"]
    m["RuleMatP"], m["RuleMatR"], m["RuleMatFone"] = (f"{rule['precision']:.3f}", f"{rule['recall']:.3f}",
                                                      f"{rule['f1']:.3f}")
    n_mat_test = rule["tp"] + rule["fn"]
    m["MatNoTagPct"] = f"{100 * stats['test_mat_pos_without_keyword'] / n_mat_test:.0f}"
    q = stats["vuln_prev_by_length_quartile_train"]
    m["VulnQOne"], m["VulnQFour"] = f"{100 * q['Q1']:.1f}", f"{100 * q['Q4']:.1f}"
    m["CloneTestPct"] = f"{100 * stats['test_rows_in_train_identifier_abstracted'] / stats['test_public_n']:.0f}"
    m["CloneAgreePct"] = f"{100 * stats['test_overlap_vuln_label_agreement_identifier_abstracted']:.0f}"
    m["PadEff"] = f"{100 * stats['test_padding_efficiency_official']:.0f}"
    raw = test_tb.loc[("microsoft/codebert-base", "raw")]
    ours = test_tb.loc[("microsoft/unixcoder-base", "single_line")]
    m["TokRaw"], m["TokOurs"] = f"{raw['mean']:.0f}", f"{ours['mean']:.0f}"
    m["TruncRaw"], m["TruncOurs"] = f"{100 * raw['frac_gt_512']:.1f}", f"{100 * ours['frac_gt_512']:.1f}"

    # Cost-model accuracy over every official-loop measurement
    errs = []
    for f in EXP.glob("*/measure_*.json"):
        r = json.loads(f.read_text())
        lin = r.get("extra", {}).get("analytic", {}).get("linear_gflops")
        if lin:
            errs.append(abs(lin - r["avg_flops"]) / r["avg_flops"])
    m["CostModelErr"] = f"{100 * max(errs):.2f}"
    m["CostModelN"] = str(len(errs))

    # Final model (filled once chosen; see DECISIONS.md)
    final = ROOT / "experiments" / "FINAL_MODEL"
    if final.exists():
        name = final.read_text().strip()
        mt = load(f"experiments/{name}/measure_test.json")
        m["StudentLayers"] = str(_layers(name))
        m["FinalVulnFone"] = f"{mt['scores'][0]['f1']:.3f}"
        m["FinalMatFone"] = f"{mt['scores'][1]['f1']:.3f}"
        m["FinalMeanFone"] = f"{mt['avg_f1']:.3f}"
        m["FinalRuntime"] = f"{mt['avg_runtime']:.1f}"
        m["FinalGflopsK"] = f"{mt['avg_flops'] / 1e3:.0f}"
        m["FinalScore"] = f"{mt['score']:.3f}"
        van = ROOT / f"experiments/{name}/robust_test_vanilla_notebook.json"
        if van.exists():
            v = json.loads(van.read_text())
            m["VanillaMeanFone"], m["VanillaScore"] = f"{v['avg_f1']:.3f}", f"{v['score']:.3f}"
            m["VanillaGflopsK"] = f"{v['avg_flops'] / 1e3:.0f}"
        m["FinalGflopsRatio"] = f"{OFFICIAL_T4['gflops'] / mt['avg_flops']:.0f}"
        m["FinalRuntimeRatio"] = f"{b_off['avg_runtime'] / mt['avg_runtime']:.0f}"
        m["FinalMeanGain"] = f"{mt['avg_f1'] - b_off['avg_f1']:+.3f}"
        m["FinalMatGain"] = f"{mt['scores'][1]['f1'] - b_off['scores'][1]['f1']:+.3f}"
        m["FinalVulnDelta"] = f"{mt['scores'][0]['f1'] - b_off['scores'][0]['f1']:+.3f}"
        m["FinalVulnGap"] = f"{abs(mt['scores'][0]['f1'] - b_off['scores'][0]['f1']):.3f}"
        m["FinalMatRecall"], m["BaseMatRecall"] = f"{mt['scores'][1]['recall']:.2f}", f"{b_off['scores'][1]['recall']:.2f}"
        ea = load(f"reports/error_analysis_{name}_test.json")
        m["EaMatFn"] = str(ea["overall"]["mat"]["fn"])
        m["EaMatFnNoTag"] = str(ea["mat_fn_without_keyword"])
        m["EaMatFoneTag"] = f"{ea['satd_keyword_in_comment']['True']['mat']['f1']:.3f}"
        lb = ea["length_bucket"]
        m["EaVulnShort"] = f"{lb['1-64']['vuln']['f1']:.3f}"
        m["EaVulnLong"] = f"{lb['257-512']['vuln']['f1']:.3f}"
        pj = ea["project"]
        m["EaVulnLinux"], m["EaVulnQemu"] = f"{pj['Linux Kernel']['vuln']['f1']:.3f}", f"{pj['qemu']['vuln']['f1']:.3f}"
        m["EaVulnChrome"], m["EaVulnFfmpeg"] = f"{pj['Chrome']['vuln']['f1']:.3f}", f"{pj['FFmpeg']['vuln']['f1']:.3f}"
        cl = ea["type2_clone_in_fit"]
        m["EaVulnClone"], m["EaVulnNoClone"] = f"{cl['True']['vuln']['f1']:.3f}", f"{cl['False']['vuln']['f1']:.3f}"
    else:
        m["StudentLayers"] = "3"
    for key, e in (("Teacher", "teacher_unixcoder"), ("NoKD", "student_unix_L3")):
        p = ROOT / f"experiments/{e}/measure_test.json"
        if p.exists():
            r = json.loads(p.read_text())
            m[f"{key}TestMeanFone"], m[f"{key}TestScore"] = f"{r['avg_f1']:.3f}", f"{r['score']:.3f}"
            m[f"{key}TestVulnFone"] = f"{r['scores'][0]['f1']:.3f}"

    # Seed statistics (internal validation): distillation vs none at 3 layers
    from scipy import stats as st

    def seeds(names):
        b = [json.loads((EXP / n / "metrics.json").read_text())["best"] for n in names]
        return (np.array([x["val_mean_f1_tuned"] for x in b]), np.array([x["val_tuned"]["f1_vuln"] for x in b]))

    kd_m, kd_v = seeds(["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"])
    no_m, no_v = seeds(["student_unix_L3", "student_unix_L3_s43", "student_unix_L3_s44"])
    m["KDMean"], m["KDSd"] = f"{kd_m.mean():.4f}", f"{kd_m.std(ddof=1):.4f}"
    m["NoKDMean"], m["NoKDSd"] = f"{no_m.mean():.4f}", f"{no_m.std(ddof=1):.4f}"
    m["KDVulnGain"] = f"{kd_v.mean() - no_v.mean():.4f}"
    m["KDpMean"] = f"{st.ttest_ind(kd_m, no_m, equal_var=False).pvalue:.3f}"
    m["KDpVuln"] = f"{st.ttest_ind(kd_v, no_v, equal_var=False).pvalue:.4f}"

    # Ablation figures quoted in the text
    kd_b = [json.loads((EXP / n / "metrics.json").read_text())["best"]
            for n in ["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"]]
    m["ThrGain"] = f"{np.mean([b['val_mean_f1_tuned'] - b['val_at_0.5']['mean_f1'] for b in kd_b]):.3f}"
    raw_g = load("experiments/kd_teacher_L3_raw/measure_val.json")["avg_flops"]
    sl_g = load("experiments/kd_teacher_L3/measure_val.json")["avg_flops"]
    m["RawCostPct"] = f"{100 * (raw_g / sl_g - 1):.0f}"
    m["ScratchFone"] = f"{load('experiments/student_scratch_4x256/metrics.json')['best']['val_mean_f1_tuned']:.3f}"
    sys.path.insert(0, str(ROOT / "src"))
    from aegis.data.dataset import load_split

    m["LinuxSharePct"] = f"{100 * load_split('test').Projectname.eq('Linux Kernel').mean():.0f}"

    main_table(m)

    lines = ["% AUTO-GENERATED by scripts/paper_numbers.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


MAIN_ROWS = [  # (label, layers, measurement file or None for the organizers' printed T4 values)
    (r"CodeBERT baseline, organizers (T4)", 12, None),
    (r"CodeBERT baseline, official loop", 12, "experiments/baseline/measure_test_official.json"),
    (r"\quad + sorted dynamic padding", 12, "experiments/baseline/measure_test_sorted.json"),
    (r"UniXcoder teacher", 12, "experiments/teacher_unixcoder/measure_test.json"),
    (r"Student, no distillation", 3, "experiments/student_unix_L3/measure_test.json"),
    (r"\textbf{AEGIS}", 3, "experiments/kd_teacher_L3_mixed/measure_test.json"),
    (r"\quad AEGIS in the unmodified notebook", 3, "experiments/kd_teacher_L3_mixed/robust_test_vanilla_notebook.json"),
]


def main_table(m: dict) -> None:
    """paper/table_main.tex: public-test results under the official measurement loop."""
    rows = []
    for label, layers, path in MAIN_ROWS:
        if path is None:
            o = OFFICIAL_T4
            vals = (0.938979, 0.881393, o["f1"], o["runtime"], o["gflops"],
                    submission_score(o["f1"], o["runtime"], o["gflops"]))
        else:
            if not (ROOT / path).exists():
                continue
            r = load(path)
            vals = (r["scores"][0]["f1"], r["scores"][1]["f1"], r["avg_f1"], r["avg_runtime"], r["avg_flops"], r["score"])
        v, mt, f, t, g, s = vals
        gk = f"{g / 1e3:,.0f}".replace(",", "{,}")
        rows.append(f"{label} & {layers} & {v:.3f} & {mt:.3f} & {f:.3f} & {t:.1f} & {gk}k & {s:.3f} \\\\")
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Public-test results with the official measurement loop (two timed forward passes per batch, fp16, "
           r"\texttt{torch.profiler}). Runtime and GFLOPs are totals over the 30{,}655 functions. Except for the first row "
           r"(values printed in the official notebook, Colab T4), all rows were measured on one RTX 5080 Laptop GPU in the "
           r"same session; runtimes are hardware-dependent, GFLOPs are not. The last row runs AEGIS through the unmodified "
           r"official notebook procedure (raw input, padding to 512, threshold 0.5).}",
           r"\label{tab:main}", r"\begin{tabular}{lcrrrrrr}", r"\toprule",
           r"Model & Layers & Weak.\ F1 & MAT F1 & Mean F1 & Runtime (s) & GFLOPs & Score \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "paper" / "table_main.tex").write_text("\n".join(tex) + "\n")
    m["MainRows"] = str(len(rows))


def _layers(name: str) -> int:
    import yaml

    return int(yaml.safe_load((ROOT / f"experiments/{name}/config.yaml").read_text())["model"]["num_layers"])


if __name__ == "__main__":
    main()
