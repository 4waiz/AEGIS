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
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


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
        r = json.loads(f.read_text(encoding="utf-8"))
        lin = r.get("extra", {}).get("analytic", {}).get("linear_gflops")
        if lin:
            errs.append(abs(lin - r["avg_flops"]) / r["avg_flops"])
    m["CostModelErr"] = f"{100 * max(errs):.2f}"
    m["CostModelN"] = str(len(errs))

    # Final model (filled once chosen; see DECISIONS.md)
    final = ROOT / "experiments" / "FINAL_MODEL"
    if final.exists():
        name = final.read_text(encoding="utf-8").strip()
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
            v = json.loads(van.read_text(encoding="utf-8"))
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
        m["EaFnrClone"], m["EaFnrNoClone"] = f"{cl['True']['vuln']['fnr']:.3f}", f"{cl['False']['vuln']['fnr']:.3f}"
        src = ea["source_dataset"]
        for key, name in (("Ospr", "OSPR"), ("Bigvul", "Big-Vul"), ("Devign", "Devign")):
            m[f"Src{key}Fone"] = f"{src[name]['vuln']['f1']:.3f}"
            m[f"Src{key}Share"] = f"{100 * src[name]['vuln']['n'] / mt['n_samples']:.0f}"
            m[f"Src{key}Fnr"] = f"{src[name]['vuln']['fnr']:.2f}"
    else:
        m["StudentLayers"] = "3"
    for key, e in (("Teacher", "teacher_unixcoder"), ("NoKD", "student_unix_L3")):
        p = ROOT / f"experiments/{e}/measure_test.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            m[f"{key}TestMeanFone"], m[f"{key}TestScore"] = f"{r['avg_f1']:.3f}", f"{r['score']:.3f}"
            m[f"{key}TestVulnFone"] = f"{r['scores'][0]['f1']:.3f}"

    # Seed statistics (internal validation): distillation vs none at 3 layers
    from scipy import stats as st

    def seeds(names):
        b = [json.loads((EXP / n / "metrics.json").read_text(encoding="utf-8"))["best"] for n in names]
        return (np.array([x["val_mean_f1_tuned"] for x in b]), np.array([x["val_tuned"]["f1_vuln"] for x in b]))

    kd_m, kd_v = seeds(["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"])
    no_m, no_v = seeds(["student_unix_L3", "student_unix_L3_s43", "student_unix_L3_s44"])
    m["KDMean"], m["KDSd"] = f"{kd_m.mean():.4f}", f"{kd_m.std(ddof=1):.4f}"
    m["NoKDMean"], m["NoKDSd"] = f"{no_m.mean():.4f}", f"{no_m.std(ddof=1):.4f}"
    m["KDVulnGain"] = f"{kd_v.mean() - no_v.mean():.4f}"
    m["KDpMean"] = f"{st.ttest_ind(kd_m, no_m, equal_var=False).pvalue:.3f}"
    m["KDpVuln"] = f"{st.ttest_ind(kd_v, no_v, equal_var=False).pvalue:.4f}"

    # Ablation figures quoted in the text
    kd_b = [json.loads((EXP / n / "metrics.json").read_text(encoding="utf-8"))["best"]
            for n in ["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"]]
    m["ThrGain"] = f"{np.mean([b['val_mean_f1_tuned'] - b['val_at_0.5']['mean_f1'] for b in kd_b]):.3f}"
    raw_g = load("experiments/kd_teacher_L3_raw/measure_val.json")["avg_flops"]
    sl_g = load("experiments/kd_teacher_L3/measure_val.json")["avg_flops"]
    m["RawCostPct"] = f"{100 * (raw_g / sl_g - 1):.0f}"
    m["ScratchFone"] = f"{load('experiments/student_scratch_4x256/metrics.json')['best']['val_mean_f1_tuned']:.3f}"
    sys.path.insert(0, str(ROOT / "src"))
    from aegis.data.dataset import load_split

    m["LinuxSharePct"] = f"{100 * load_split('test').Projectname.eq('Linux Kernel').mean():.0f}"

    review_macros(m, stats)
    main_table(m)

    lines = ["% AUTO-GENERATED by scripts/paper_numbers.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def review_macros(m: dict, stats: dict) -> None:
    """Numbers added after the internal reviewer audit (reports/test_statistics.json and experiments)."""
    ts = load("reports/test_statistics.json")
    pb = ts["paired_bootstrap"]

    def ci(d):
        return f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"

    m["CiMeanVsBase"], m["CiMatVsBase"] = ci(pb["aegis_minus_baseline"]["mean"]), ci(pb["aegis_minus_baseline"]["mat"])
    m["CiVulnVsBase"] = ci(pb["aegis_minus_baseline"]["weakness"])
    m["KdTestVulnGain"] = f"{pb['kd_L3_single_line_minus_no_kd_L3']['weakness']['diff']:+.3f}"
    m["CiKdTestVuln"] = ci(pb["kd_L3_single_line_minus_no_kd_L3"]["weakness"])
    m["MixedVulnCost"] = f"{abs(pb['aegis_minus_kd_L3_single_line']['weakness']['diff']):.3f}"
    m["CiMixedVuln"] = ci(pb["aegis_minus_kd_L3_single_line"]["weakness"])
    te = ts["threshold_effect_test"]
    m["ThrTestMax"] = f"{max(abs(te[k]['gain']) for k in ('student_unix_L3', 'kd_teacher_L3', 'aegis')):.4f}"
    tr = ts["baseline_mat_truncation"]
    m["TruncMatMiss"], m["AegisTaggedFn"] = str(tr["baseline_fn_tagged_and_truncated"]), str(tr["aegis_fn_tagged"])
    m["BaseTaggedFn"] = str(tr["baseline_fn_tagged"])
    rv = ts["mat_rule_vs_aegis"]
    m["RuleAgree"], m["RuleN"] = f"{rv['agree']:,}".replace(",", "{,}"), f"{rv['n']:,}".replace(",", "{,}")
    m["AegisMatExtraFp"] = str(rv["aegis"]["fp"] - rv["rule"]["fp"])
    m["RuleMatFoneTrain"] = f"{stats['train_rule_satd_word_in_comments']['f1']:.3f}"
    m["LengthOnlyFone"] = f"{stats['length_probe_vuln_f1_best_threshold'][0]:.3f}"
    gd = ts["gflops_decomposition"]
    m["PadFactor"], m["ModelFactor"] = f"{gd['padding_factor']:.1f}", f"{gd['model_and_input_factor']:.1f}"
    at = ts["attention_share_not_counted"]
    m["AttnBase"], m["AttnAegis"] = f"{100 * at['baseline_official']:.0f}", f"{100 * at['aegis']:.1f}"
    # Seed ranges (internal validation)
    rng = lambda names: [json.loads((EXP / n / "metrics.json").read_text(encoding="utf-8"))["best"]["val_mean_f1_tuned"] for n in names]  # noqa: E731
    kd, no = rng(["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"]), rng(["student_unix_L3", "student_unix_L3_s43", "student_unix_L3_s44"])
    m["KdRange"], m["NoKdRange"] = f"{min(kd):.4f}--{max(kd):.4f}", f"{min(no):.4f}--{max(no):.4f}"
    tinit = [n for n in ["student_tinit_L3", "student_tinit_L3_s43", "student_tinit_L3_s44"] if (EXP / n / "metrics.json").exists()]
    if tinit:
        ti = rng(tinit)
        m["TinitMean"], m["TinitRange"], m["TinitN"] = f"{np.mean(ti):.4f}", f"{min(ti):.4f}--{max(ti):.4f}", str(len(ti))
    mixed = [n for n in ["kd_teacher_L3_mixed", "kd_teacher_L3_mixed_s43", "kd_teacher_L3_mixed_s44"] if (EXP / n / "metrics.json").exists()]
    mx = rng(mixed)
    m["MixedMean"], m["MixedRange"], m["MixedN"] = f"{np.mean(mx):.4f}", f"{min(mx):.4f}--{max(mx):.4f}", str(len(mx))
    # Depth optimum when costs are totalled over the hidden split (61,311) without rescaled constants
    def hidden(names):
        f1 = float(np.mean(rng(names)))
        r = json.loads((EXP / names[0] / "measure_val.json").read_text(encoding="utf-8"))
        k = 61_311 / r["n_samples"]
        return submission_score(f1, r["avg_runtime"] * k, r["avg_flops"] * k)
    h2, h3 = hidden(["kd_teacher_L2"]), hidden(["kd_teacher_L3", "kd_teacher_L3_s43", "kd_teacher_L3_s44"])
    m["HiddenTwo"], m["HiddenThree"] = f"{h2:.4f}", f"{h3:.4f}"
    # Single-line distilled model on the public test
    sl = load("experiments/kd_teacher_L3/measure_test.json")
    m["SingleMeanFone"], m["SingleScore"] = f"{sl['avg_f1']:.3f}", f"{sl['score']:.3f}"
    t4 = ROOT / "experiments" / "colab_t4" / "results.json"
    if t4.exists():
        r = json.loads(t4.read_text(encoding="utf-8"))
        for key, name in (("Base", "official_baseline"), ("Aegis", "aegis")):
            x = r[name]
            m[f"Tfour{key}Fone"], m[f"Tfour{key}Runtime"] = f"{x['f1']:.3f}", f"{x['runtime']:.1f}"
            m[f"Tfour{key}GflopsK"], m[f"Tfour{key}Score"] = f"{x['gflops'] / 1e3:,.0f}".replace(",", "{,}"), f"{x['score']:.3f}"
    rep = load("experiments/kd_teacher_L3_mixed/official_replica_test.json")
    van = load("experiments/kd_teacher_L3_mixed/robust_test_vanilla_notebook.json")
    m["ReplicaMatches"] = "identical" if all(rep["scores"][i][k] == van["scores"][i][k] for i in range(2)
                                             for k in ("tp", "fp", "fn")) else "different"


MAIN_ROWS = [  # (label, layers, source); "t4:<key>" = experiments/colab_t4/results.json, "rule" = midrule
    (r"\multicolumn{8}{l}{\emph{Colab T4 (official ranking hardware), one session}}", None, "header"),
    (r"CodeBERT baseline, official notebook", 12, "t4:official_baseline"),
    (r"\textbf{AEGIS}", 3, "t4:aegis"),
    (None, None, "rule"),
    (r"\multicolumn{8}{l}{\emph{RTX 5080 Laptop GPU, one session}}", None, "header"),
    (r"CodeBERT baseline, official loop", 12, "experiments/baseline/measure_test_official.json"),
    (r"\quad + sorted dynamic padding", 12, "experiments/baseline/measure_test_sorted.json"),
    (r"UniXcoder teacher", 12, "experiments/teacher_unixcoder/measure_test.json"),
    (r"Student, no distillation", 3, "experiments/student_unix_L3/measure_test.json"),
    (r"Student + KD, single-line input", 3, "experiments/kd_teacher_L3/measure_test.json"),
    (r"\textbf{AEGIS} (KD, mixed input)", 3, "experiments/kd_teacher_L3_mixed/measure_test.json"),
    (r"\quad AEGIS in the unmodified notebook", 3, "experiments/kd_teacher_L3_mixed/robust_test_vanilla_notebook.json"),
]


def main_table(m: dict) -> None:
    """paper/table_main.tex: public-test results under the official measurement loop."""
    rows = []
    t4_path = ROOT / "experiments" / "colab_t4" / "results.json"
    t4 = json.loads(t4_path.read_text(encoding="utf-8")) if t4_path.exists() else None
    for label, layers, src in MAIN_ROWS:
        if src == "rule":
            rows.append(r"\midrule")
            continue
        if src == "header":
            rows.append(label + r" \\")
            continue
        if src.startswith("t4:"):
            if t4 is None:
                continue
            x = t4[src[3:]]
            vals = (x["f1_weakness"], x["f1_mat"], x["f1"], x["runtime"], x["gflops"], x["score"])
        else:
            if not (ROOT / src).exists():
                continue
            r = load(src)
            vals = (r["scores"][0]["f1"], r["scores"][1]["f1"], r["avg_f1"], r["avg_runtime"], r["avg_flops"], r["score"])
        v, mt, f, t, g, s = vals
        gk = f"{g / 1e3:,.0f}".replace(",", "{,}")
        rows.append(f"{label} & {layers} & {v:.3f} & {mt:.3f} & {f:.3f} & {t:.1f} & {gk}k & {s:.3f} \\\\")
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Public-test results (30{,}655 functions) with the official measurement loop (two timed forward passes "
           r"per batch, fp16, \texttt{torch.profiler}); runtime and GFLOPs are totals over the split. GFLOPs do not depend on "
           r"the hardware; runtimes and scores are comparable only within a block. The organizers report 198.9\,s and a score "
           r"of 0.729 for the baseline on their T4. The last row runs AEGIS through the unmodified notebook procedure "
           r"(raw input, padding to 512, threshold 0.5).}",
           r"\label{tab:main}", r"\begin{tabular}{lcrrrrrr}", r"\toprule",
           r"Model & Layers & Weak.\ F1 & MAT F1 & Mean F1 & Runtime (s) & GFLOPs & Score \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "paper" / "table_main.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    m["MainRows"] = str(len(rows))


def _layers(name: str) -> int:
    import yaml

    return int(yaml.safe_load((ROOT / f"experiments/{name}/config.yaml").read_text(encoding="utf-8"))["model"]["num_layers"])


if __name__ == "__main__":
    main()
