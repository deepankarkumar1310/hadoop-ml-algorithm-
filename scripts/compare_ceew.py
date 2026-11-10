"""Compare our results with the CEEW 2021 district climate vulnerability assessment (Review 3).

Source: A. Mohanty and S. Wadhawan, "Mapping India's Climate Vulnerability: A District Level Assessment",
CEEW, 2021 (CC BY), Annexure I, Table A1: 463 hotspot districts with exposure, sensitivity, adaptive capacity,
vulnerability index and class.

Step 1 (--extract report.txt): read the table from `pdftotext -layout` output of the report and write
        data/raw/ceew_2021_tableA1.csv. The PDF itself is not stored in the repository.
Step 2 (default): match CEEW districts to Census 2011 codes and compare with our 469 districts.

CEEW lists districts without their state, so matching is all-India by name: exact normalised name, then the
rename table of match_districts.py, then a unique fuzzy match (ratio >= 0.86). A CEEW name that fits two or
more Census districts (e.g. Hamirpur, Aurangabad) is dropped as ambiguous rather than guessed.

Writes results/ceew_matched.csv and results/ceew_comparison.json.
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import RES, ROOT  # noqa: E402
from scripts.match_districts import FUZZY_MIN, RENAME, norm_name  # noqa: E402

CSV = ROOT / "data" / "raw" / "ceew_2021_tableA1.csv"
ROW = re.compile(r"^\s*(\d+)\s+(.+?)\s{2,}((?:Flood|Drought|Cyclone)[A-Za-z ,&]*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                 r"\s+([\d.]+)\s+(Very High|High|Moderate|Low|Very Low)\s*$")
CLASSES = ["Very Low", "Low", "Moderate", "High", "Very High"]
# Spellings in the CEEW table that differ from Census 2011 (checked by hand).
CEEW_FIX = {"khamman": "khammam", "garhchiroli": "gadchiroli"}


def extract(txt):
    lines = Path(txt).read_text(encoding="utf-8", errors="replace").split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("Table A1 List"))
    j = next(k for k, l in enumerate(lines) if l.strip().startswith("Table A2 Comparison"))
    rows = [ROW.match(l).groups() for l in lines[i:j] if ROW.match(l)]
    missed = [l for l in lines[i:j] if re.search(r"\d\.\d{3}", l) and not ROW.match(l)]
    assert not missed, missed[:3]
    df = pd.DataFrame(rows, columns=["ceew_rank", "district", "event", "exposure", "sensitivity",
                                     "adaptive_capacity", "vulnerability_index", "vulnerability_class"])
    for c in ["ceew_rank", "exposure", "sensitivity", "adaptive_capacity", "vulnerability_index"]:
        df[c] = pd.to_numeric(df[c])
    df["event"] = df.event.str.strip()
    df.to_csv(CSV, index=False)
    print(f"wrote {CSV.relative_to(ROOT)} with {len(df)} districts")


def match(ceew, census):
    keys = census.assign(nkey=census.district_name.map(lambda s: norm_name(s.replace("(", " ("))))
    by_key = keys.groupby("nkey").censuscode.apply(list).to_dict()
    out = []
    for r in ceew.itertuples():
        k = norm_name(r.district)
        k = CEEW_FIX.get(k, k)
        hit = by_key.get(k) or by_key.get(RENAME.get(k, k)) or []
        how = "exact"
        if not hit:
            close = [c for c in by_key if difflib.SequenceMatcher(None, k, c).ratio() >= FUZZY_MIN]
            hit, how = (by_key[close[0]], "fuzzy") if len(close) == 1 else ([], "ambiguous" if close else "none")
        if len(hit) > 1:
            hit, how = [], "ambiguous"
        out.append({"district": r.district, "censuscode": hit[0] if hit else np.nan, "match": how})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", help="pdftotext -layout output of the CEEW report")
    a = ap.parse_args()
    if a.extract:
        extract(a.extract)
        return
    ceew = pd.read_csv(CSV)
    census = pd.read_csv(ROOT / "data" / "processed" / "census2011_district_variables.csv")
    m = match(ceew, census)
    ceew = ceew.join(m[["censuscode", "match"]])
    dup = ceew.censuscode.duplicated(keep=False) & ceew.censuscode.notna()
    ceew.loc[dup, ["censuscode", "match"]] = [np.nan, "duplicate"]

    tab = pd.read_csv(ROOT / "data" / "processed" / "analysis_table.csv")
    ex = pd.read_csv(RES / "district_explanations.csv")[["censuscode", "push", "quadrant", "vul_group"]]
    pr = pd.read_csv(RES / "predictions" / "B-ACPN (proposed)__scaled.csv")
    pr = pr[pr.kind == "cv"].groupby("censuscode").prob.mean().rename("bacpn_prob").reset_index()
    ours = tab[["censuscode", "district_name", "state_name", "climate_vul_in"]].merge(ex, on="censuscode") \
        .merge(pr, on="censuscode")
    j = ceew.dropna(subset=["censuscode"]).astype({"censuscode": int}).merge(ours, on="censuscode")
    j.to_csv(RES / "ceew_matched.csv", index=False)

    def rho(a_, b_):
        r, p = spearmanr(j[a_], j[b_])
        return {"rho": round(float(r), 3), "p": float(f"{p:.2e}")}

    ceew_high = j.vulnerability_index >= j.vulnerability_index.median()
    ours_high = j.vul_group == "High"
    a_ = j[j.quadrant == "High dev / High vul"]
    e_ = j[j.quadrant == "High dev / Low vul"]
    hi_cls = {"High", "Very High"}
    out = {
        "ceew_districts": int(len(ceew)), "matched_to_census": int(ceew.censuscode.notna().sum()),
        "match_types": {k: int(v) for k, v in ceew.match.value_counts().items()},
        "in_our_469": int(len(j)),
        "spearman_ceew_vi_vs_ndap_cvi": rho("vulnerability_index", "climate_vul_in"),
        "spearman_ceew_vi_vs_bacpn_prob": rho("vulnerability_index", "bacpn_prob"),
        "spearman_ceew_exposure_vs_push": rho("exposure", "push"),
        "spearman_ceew_vi_vs_push": rho("vulnerability_index", "push"),
        "high_low_agreement": round(float((ceew_high == ours_high).mean()), 3),
        "high_low_kappa": round(float(cohen_kappa_score(ceew_high, ours_high)), 3),
        "anomaly_A_n": int(len(a_)), "expected_dev_n": int(len(e_)),
        "anomaly_A_ceew_high_share": round(float(a_.vulnerability_class.isin(hi_cls).mean()), 3),
        "expected_dev_ceew_high_share": round(float(e_.vulnerability_class.isin(hi_cls).mean()), 3),
        "anomaly_A_ceew_exposure_median": round(float(a_.exposure.median()), 3),
        "expected_dev_ceew_exposure_median": round(float(e_.exposure.median()), 3),
        "anomaly_A_vs_expected_exposure_p": float(f"{mannwhitneyu(a_.exposure, e_.exposure).pvalue:.2e}"),
        "class_counts_in_469": {c: int((j.vulnerability_class == c).sum()) for c in CLASSES},
    }
    (RES / "ceew_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    figure(j, out)


def figure(j, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts import viz_style as V
    V.apply()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for q, st in V.QUADRANT_STYLE.items():
        d = j[j.quadrant == q]
        ax[0].scatter(d.climate_vul_in, d.vulnerability_index, s=16, alpha=0.75, color=st["color"],
                      marker=st["marker"], label=st["label"], linewidths=0)
    r = out["spearman_ceew_vi_vs_ndap_cvi"]
    ax[0].set(xlabel="NDAP climate vulnerability index (our target)", ylabel="CEEW 2021 vulnerability index",
              title=f"Same {out['in_our_469']} districts: Spearman ρ = {r['rho']:.2f} (p = {r['p']:.2f})")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=8)
    groups = [("Expected\ndeveloped", "High dev / Low vul"), ("Anomaly A\ndeveloped,\nvulnerable", "High dev / High vul"),
              ("Expected\nless dev.", "Low dev / High vul"), ("Anomaly B\nless dev.,\nresilient", "Low dev / Low vul")]
    share = [j[j.quadrant == k].vulnerability_class.isin({"High", "Very High"}).mean() for _, k in groups]
    cols = [V.QUADRANT_STYLE[k]["color"] for _, k in groups]
    b = ax[1].bar([g for g, _ in groups], share, color=cols, width=0.6)
    ax[1].bar_label(b, labels=[f"{s:.0%}" for s in share], padding=2, fontsize=9)
    ax[1].set(ylim=(0, max(share) * 1.25), ylabel="Share rated High / Very High by CEEW",
              title="CEEW high-vulnerability share by our group")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[1].tick_params(axis="x", labelsize=9)
    V.save(fig, ROOT / "figures" / "ceew_comparison.png")


if __name__ == "__main__":
    main()
