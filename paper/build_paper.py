"""Fill paper/paper_template.tex with numbers and tables from results/ and compile the IEEE paper.

Usage: python paper/build_paper.py        (needs pdflatex with IEEEtran, run after scripts/evaluate.py)
Author details that are not in the project stay PENDING until filled in below.
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, PAPER = ROOT / "results", ROOT / "paper"

DEPT = "Department of Networking and Communications,"
EMAILS = {"FACULTY": "banuprip2@srmist.edu.in", "SEHJODH": "sg2827@srmist.edu.in", "DEVANSHU": "dk5504@srmist.edu.in",
          "DEEPANKAR": "dz9629@srmist.edu.in", "TANMAY": "ts4494@srmist.edu.in"}

SHORT = {"B-ACPN (proposed)": "B-ACPN (ours)", "Logistic Regression": "Logistic regression",
         "Random Forest": "Random forest", "Gradient-Boosted Trees": "Gradient-boosted trees",
         "Linear SVM": "Linear SVM", "Standard MLP": "Standard MLP"}
ABBR = {"B-ACPN (proposed)": "B-ACPN", "Logistic Regression": "LR", "Random Forest": "RF",
        "Gradient-Boosted Trees": "GBT", "Linear SVM": "SVM", "Standard MLP": "MLP"}


def j(name):
    return json.loads((RES / name).read_text())


def f3(v):
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def table(label, caption, cols, rows, star=False, size=r"\footnotesize", fit=False):
    env = "table*" if star else "table"
    head = " & ".join(cols) + r" \\"
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    tab = (f"\\begin{{tabular}}{{{'l' + 'c' * (len(cols) - 1)}}}\n\\toprule\n"
           f"{head}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
    if fit:  # scale to the column width so the table sits in one column
        tab = "\\resizebox{\\linewidth}{!}{" + tab + "}"
    return (f"\\begin{{{env}}}[{'!t' if star else '!htb'}]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
            f"\\setlength{{\\tabcolsep}}{{3.5pt}}\n{tab}\n\\end{{{env}}}")


def main():
    pm, ar, cc = j("preprocessing_metrics.json"), j("analysis_results.json"), j("preprocessing_colab_checks.json")
    km, hd, card, push = j("kmeans_regimes.json"), j("hdfs_usage_bytes.json"), j("table4_model_card.json"), \
        j("table6_climate_push.json")
    t5 = pd.read_csv(RES / "table5_comparison.csv").set_index("model")
    full = pd.read_csv(RES / "classifier_metrics_full.csv").set_index("model")
    sig = pd.read_csv(RES / "significance.csv").set_index("model")
    ab = pd.read_csv(RES / "table6_ablation.csv")
    ba = pd.read_csv(RES / "preprocessing_before_after.csv").set_index("model")
    tm = pd.read_csv(RES / "scalability_timings.csv").groupby("years").runtime_s.mean()
    M = "B-ACPN (proposed)"
    order = t5.sort_values("roc_auc", ascending=False).index

    T = {}
    T["TABLE_DATA"] = table("tab:data", "Data sources", ["Dataset", "Source", "Year", "Size", "Role"], [
        ["Vulnerability index", "NDAP 7160", "2021", f"{pm['ndap']['rows']}$\\times${pm['ndap']['cols']}",
         "target, 2 dev."],
        ["Census abstract", "Census", "2011", f"{pm['census']['districts']} dist.", "3 dev."],
        ["Daily rainfall, 0.25\\textdegree{}", "IMD", "1991--2020", "763 MB", "hazards"],
        ["Daily Tmax/Tmin, 1\\textdegree{}", "IMD", "1991--2020", "84 MB", "hazards"],
        ["Boundaries", "DataMeet", "2011", "641 polygons", "weights"]], size=r"\scriptsize")

    a = cc["area_vs_nearest_cell"]["metrics"]
    per = cc["mapreduce_vs_pandas"]["per_index"].values()
    worst = {k: max(v[k] for v in per) for k in ("rmse", "mse", "mae", "mape_pct", "nrmse", "medae")}
    worst["r2"] = min(v["r2"] for v in per)
    worst["bias"] = max((v["bias"] for v in per), key=abs)

    def g(v, k):
        if k == "r2":
            return f"{v:.6f}" if 0.9999 < v < 1 else f"{v:.3f}"
        if k == "mape_pct":
            return f"{v:.2f}\\%" if v >= 0.01 else f"{v:.3f}\\%"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            m, e = f"{v:.1e}".split("e")
            return f"${m}\\times10^{{{int(e)}}}$"
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
    names = [("RMSE", "rmse"), ("MSE", "mse"), ("MAE", "mae"), ("MAPE", "mape_pct"), ("$R^2$", "r2"),
             ("NRMSE", "nrmse"), ("MedAE", "medae"), ("Bias (ME)", "bias")]
    T["TABLE_PREP"] = table("tab:prep", "Preprocessing error metrics",
                            ["Metric", "Nearest cell vs ours (mm)", "MapReduce vs pandas$^{*}$"],
                            [[n, g(a[k], k), g(worst[k], k)] for n, k in names]).replace(
        "\\end{table}", "\\\\[2pt]\\parbox{0.95\\columnwidth}{\\scriptsize Left: annual rainfall of "
        f"{a['n']} districts, {cc['area_vs_nearest_cell']['year']}; reference = our area-weighted value. "
        "$^{*}$Worst of the seven indices over "
        f"{cc['mapreduce_vs_pandas']['rows_compared']:,} district-years.}}\n\\end{{table}}")

    T["TABLE_BIGDATA"] = table("tab:bigdata", "Big Data jobs (single node, Colab run)", ["Job", "Input", "Output", "Result"], [
        ["HDFS load", "raw + staging", "3 zones", f"{sum(hd.values()) / 1e9:.2f} GB"],
        ["MapReduce", "7,013,120 rows", "19,200 rows", f"{tm[30]:.0f} s"],
        ["Hive", "19,200 rows + dev.", "469 districts", "= pandas"],
        ["Spark k-means", "8 hazards", f"{km['k']} regimes", f"silh. {km['silhouette']:.3f}"]])

    rows = []
    for m in order:
        r = t5.loc[m]
        name = f"\\textbf{{{ABBR[m]}}}" if m == M else ABBR[m]
        rows.append([name, f"{r.roc_auc:.3f}$\\pm${r.roc_auc_sd:.3f}", f"{r.bal_acc:.3f}", f"{r.accuracy:.3f}",
                     f"{r.f1:.3f}", f"{r.loso_auc:.3f}", sig.loc[m, "p_value_fmt"].replace("—", "--").replace("<", "$<$")])
    T["TABLE_CMP"] = table("tab:cmp", "Comparison with existing models (mean over 25 folds)",
                           ["Model", "ROC-AUC", "Bal. acc.", "Acc.", "F1", "LOSO", "$p$"], rows, fit=True)

    cols = [("precision", "Prec."), ("recall", "Rec."), ("specificity", "Spec."), ("mcc", "MCC"), ("kappa", "$\\kappa$"),
            ("pr_auc", "PR-AUC"), ("log_loss", "LogL$\\downarrow$"), ("brier", "Brier$\\downarrow$"),
            ("ece", "ECE$\\downarrow$"), ("rmse", "RMSE$\\downarrow$")]
    _detail = table("tab:detail", "Further classification metrics ($\\downarrow$ = lower is better)",
                              ["Model"] + [c[1] for c in cols],
                              [[ABBR[m]] + [f3(full.loc[m, c]) for c, _ in cols] for m in order], fit=True)
    # Tables IV and V side by side: left and right halves of one full-width float
    def _half(tbl, w):
        body = tbl.split("\\centering\n", 1)[1].rsplit("\\end{table}", 1)[0]
        return f"\\begin{{minipage}}[t]{{{w}\\textwidth}}\n\\centering\n" + body + "\\end{minipage}"
    T["TABLE_CMP"] = ("\\begin{table*}[t]\n" + _half(T["TABLE_CMP"], 0.42) + "\\hfill\n" + _half(_detail, 0.56) +
                      "\n\\end{table*}")
    T["TABLE_DETAIL"] = ""

    bam = [("Accuracy", "acc"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
           ("Specificity", "specificity"), ("ROC-AUC", "auc"), ("MCC", "mcc"), ("Log loss$\\downarrow$", "log_loss"),
           ("RMSE$\\downarrow$", "rmse")]
    ms = [M, "Standard MLP", "Random Forest", "Logistic Regression"]
    T["TABLE_BA"] = table("tab:ba", "Before $\\rightarrow$ after preprocessing (raw vs.\\ preprocessed features)",
                          ["Metric"] + [ABBR[m] for m in ms],
                          [[n] + [f"{ba.loc[m, k + '_before']:.3f}$\\rightarrow${ba.loc[m, k + '_after']:.3f}" for m in ms]
                           for n, k in bam])

    cm = pd.read_csv(RES / "confusion_matrices.csv").set_index("model")
    cm["sens"], cm["spec"] = cm.tp / (cm.tp + cm.fn), cm.tn / (cm.tn + cm.fp)
    T["TABLE_CM"] = table("tab:cm", "Confusion counts summed over the 25 test folds", ["Model", "TP", "TN", "FP", "FN",
                          "Sens.", "Spec.", "Errors"],
                          [[(f"\\textbf{{{ABBR[m]}}}" if m == M else ABBR[m]), int(cm.loc[m, "tp"]), int(cm.loc[m, "tn"]),
                            int(cm.loc[m, "fp"]), int(cm.loc[m, "fn"]), f"{cm.loc[m, 'sens']:.3f}", f"{cm.loc[m, 'spec']:.3f}",
                            int(cm.loc[m, "fp"] + cm.loc[m, "fn"])] for m in order])
    T["TABLE_CI"] = table("tab:ci", "ROC-AUC with 95\\% confidence interval and corrected $t$-test against B-ACPN",
                          ["Model", "ROC-AUC", "95\\% CI", "$p$ vs B-ACPN"],
                          [[(f"\\textbf{{{ABBR[m]}}}" if m == M else ABBR[m]), f"{sig.loc[m, 'auc_mean']:.3f}",
                            sig.loc[m, "auc_ci"].replace("–", "--"),
                            "--" if pd.isna(sig.loc[m, "p_value"]) else f"{sig.loc[m, 'p_value']:.3f}"] for m in order])
    fm = pd.read_csv(RES / "fold_metrics.csv").groupby("model").roc_auc
    CMX = {"CM_TN": int(cm.loc[M, "tn"]), "CM_FP": int(cm.loc[M, "fp"]), "CM_TP": int(cm.loc[M, "tp"]),
           "CM_FN": int(cm.loc[M, "fn"]), "CM_SPEC": f"{cm.loc[M, 'spec']:.3f}", "CM_SENS": f"{cm.loc[M, 'sens']:.3f}",
           "CM_ERR": int(cm.loc[M, "fp"] + cm.loc[M, "fn"]),
           "CM_ERR_MIN_OTHER": int((cm.fp + cm.fn).drop(M).min()), "CM_ERR_MAX_OTHER": int((cm.fp + cm.fn).drop(M).max()),
           "FOLD_MED": f"{fm.median()[M]:.3f}", "FOLD_MIN": f"{fm.min()[M]:.3f}", "FOLD_MAX": f"{fm.max()[M]:.3f}"}
    T["TABLE_ABL"] = table("tab:abl", "Ablation of B-ACPN (ROC-AUC, 25 folds)", ["Variant", "ROC-AUC", "Change", "$p$"],
                           [[r.model, f"{r.roc_auc:.3f}", "--" if r.model == "Full B-ACPN" else f"{r.delta:+.3f}",
                             "--" if pd.isna(r.p_value) else f"{r.p_value:.2f}"] for r in ab.itertuples()])

    q = push["by_quadrant"]
    T["TABLE_PUSH"] = table("tab:push", "Mean climate push by group", ["Group", "$n$", "Push", "Push $>$ 0"], [
        ["Developed, low vul. (expected)", q["High dev / Low vul"]["n"], f"{q['High dev / Low vul']['push']:+.2f}",
         f"{q['High dev / Low vul']['share_positive']:.0%}".replace("%", "\\%")],
        ["Less developed, high vul. (expected)", q["Low dev / High vul"]["n"], f"{q['Low dev / High vul']['push']:+.2f}",
         f"{q['Low dev / High vul']['share_positive']:.0%}".replace("%", "\\%")],
        ["Anomaly A: developed, vulnerable", q["High dev / High vul"]["n"], f"{q['High dev / High vul']['push']:+.2f}",
         f"{q['High dev / High vul']['share_positive']:.0%}".replace("%", "\\%")],
        ["Anomaly B: less developed, resilient", q["Low dev / Low vul"]["n"], f"{q['Low dev / Low vul']['push']:+.2f}",
         f"{q['Low dev / Low vul']['share_positive']:.0%}".replace("%", "\\%")]])

    tn = ar["anomaly_profiles"]["A_vs_expected_high_dev"]["variables"]["tropical_nights"]
    am = t5.loc[M]
    N = {"N_DIST": ar["n_districts"], "N_STATES": ar["n_states"], "HDFS_GB": f"{sum(hd.values()) / 1e9:.2f}",
         "MR_IN": "7,013,120", "MR_OUT": "19,200", "ACP_AUC": f"{am.roc_auc:.3f}", "ACP_ACC": f"{am.accuracy:.3f}",
         "ACP_F1": f"{am.f1:.3f}", "P_GBT": f"{sig.loc['Gradient-Boosted Trees', 'p_value']:.2f}",
         "R2_DEV": f"{ar['regression']['m1_dev_only']['r2']:.3f}", "R2_DC": f"{ar['regression']['m2_dev_climate']['r2']:.3f}",
         "N_ANOM": ar["anomalies"], "ANOM_PCT": f"{ar['anomaly_share'] * 100:.1f}",
         "LIT_ERR": pm["census"]["literacy_abs_error_pp"], "MATCHED": pm["matching"]["matched"],
         "NDAP_N": pm["matching"]["ndap_districts"], "NDAP_MISS": f"{pm['ndap']['overall_missing_pct']:.0f}",
         "AGG_RMSE": f"{a['rmse']:.0f}", "AGG_MAPE": f"{a['mape_pct']:.1f}", "N_PARAMS": card["parameters"],
         "TRAIN_TIME": card["train_time"].replace(" per fold", ""),
         "CONF_COV": f"{card['conformal_coverage_mean'] * 100:.1f}", "CONF_SIZE": f"{card['conformal_avg_set_size']:.2f}",
         "BA_AUC0": f"{ba.loc[M, 'auc_before']:.3f}", "BA_AUC1": f"{ba.loc[M, 'auc_after']:.3f}",
         "PUSH_A": f"{q['High dev / High vul']['push']:+.2f}", "PUSH_EA": f"{q['High dev / Low vul']['push']:+.2f}",
         "PUSH_B": f"{q['Low dev / Low vul']['push']:+.2f}", "PUSH_EB": f"{q['Low dev / High vul']['push']:+.2f}",
         "P_A": f"{push['anomaly_A_vs_expected_p']:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-"),
         "P_B": f"{push['anomaly_B_vs_expected_p']:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-"),
         "TN_A": f"{tn['anomaly_median']:.0f}", "TN_E": f"{tn['expected_median']:.0f}", "TN_P": f"{tn['mw_p']:.3f}",
         "REG1": f"{push['by_regime']['1']['push']:+.2f}", "REG0": f"{push['by_regime']['0']['push']:+.2f}",
         "T1": f"{tm[1]:.0f}", "T10": f"{tm[10]:.0f}", "T30": f"{tm[30]:.0f}",
         "DEPT": DEPT, **{f"EMAIL_{k}": v for k, v in EMAILS.items()}}
    ts = json.loads((RES / "tuning_summary.json").read_text())
    TUNE = {"TUNE_N": ts["settings_tried"], "TUNE_WORST": f"{ts['cv_auc_worst']:.3f}",
                 "TUNE_BEST": f"{ts['cv_auc_best']:.3f}", "TUNE_GAIN": f"{ts['cv_auc_best'] - ts['cv_auc_default']:.3f}",
                 "TUNE_LGAIN": f"{ts['loso_auc_best'] - ts['loso_auc_default']:.3f}"}
    N.update(TUNE)
    N.update(CMX)
    cc = json.loads((RES / "ceew_comparison.json").read_text())
    N.update({"CE_N": cc["ceew_districts"], "CE_BOTH": cc["in_our_469"],
              "CE_R1": f"{cc['spearman_ceew_vi_vs_ndap_cvi']['rho']:.2f}", "CE_P1": f"{cc['spearman_ceew_vi_vs_ndap_cvi']['p']:.2f}",
              "CE_R2": f"{cc['spearman_ceew_vi_vs_bacpn_prob']['rho']:.2f}",
              "CE_R3": f"{cc['spearman_ceew_vi_vs_push']['rho']:.2f}", "CE_P3": f"{cc['spearman_ceew_vi_vs_push']['p']:.2f}",
              "CE_AGREE": f"{cc['high_low_agreement']:.0%}".replace("%", "\\%"), "CE_K": f"{cc['high_low_kappa']:.2f}"})
    for k in ("P_A", "P_B"):
        N[k] = "$" + N[k] + "}$"

    tex = (PAPER / "paper_template.tex").read_text()
    for k, v in {**T, **N}.items():
        tex = tex.replace(f"@@{k}@@", str(v))
    assert "@@" not in tex, [w for w in tex.split() if "@@" in w][:5]
    (PAPER / "paper.tex").write_text(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=PAPER, check=True,
                       stdout=subprocess.DEVNULL)
    print("written", PAPER / "paper.pdf")


if __name__ == "__main__":
    main()
