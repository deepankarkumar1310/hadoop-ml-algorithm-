"""Build demo/index.html: an interactive, offline walkthrough of the whole project for the viva.

Every number on the page is read from results/ and data/processed/ (no network, no external libraries).
Sections: data -> Big Data pipeline (replay of the logged Colab run) -> preprocessing -> B-ACPN district
explorer -> comparison with the five existing models -> India map of the climate push.

Usage: python demo/build_demo.py      then open demo/index.html in any browser (works offline).
Also writes demo/artifact.html (the same page without the <html>/<head> wrapper, for publishing).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from scripts.common import DEV, HAZ  # noqa: E402

MODELS = ["B-ACPN (proposed)", "Gradient-Boosted Trees", "Random Forest", "Standard MLP", "Linear SVM",
          "Logistic Regression"]
SHORT = {"B-ACPN (proposed)": "B-ACPN", "Gradient-Boosted Trees": "GBT", "Random Forest": "Random forest",
         "Standard MLP": "MLP", "Linear SVM": "Linear SVM", "Logistic Regression": "Logistic reg."}
HAZN = {"annual_rain_mm": ["Annual rain", "mm"], "rain_variability": ["Rain variability", "mm SD"],
        "rx5day_mm": ["Max 5-day rain", "mm"], "heavy_rain_days": ["Heavy-rain days", "days/yr"],
        "max_dry_spell": ["Longest dry spell", "days"], "mean_tmax": ["Mean Tmax", "°C"],
        "hot_days": ["Hot days (>35 °C)", "days/yr"], "tropical_nights": ["Tropical nights", "nights/yr"]}
DEVN = {"literacy_rate": ["Literacy", "%"], "female_literacy_rate": ["Female literacy", "%"],
        "agri_worker_share": ["Agricultural workers", "%"], "mgnrega_days": ["MGNREGA days", "days"],
        "yield_variability": ["Yield variability", "index"]}


def j(name):
    return json.loads((RES / name).read_text())


def map_paths(codes):
    """District outlines simplified to SVG paths in a lon/lat projection scaled by cos(latitude)."""
    import shapefile
    from shapely.geometry import shape
    sf = shapefile.Reader(str(ROOT / "data" / "raw" / "boundaries" / "2011_Dist.shp"))
    fields = [f[0] for f in sf.fields[1:]]
    key = next(f for f in fields if f.lower() in ("censuscode", "censuscod", "dist_cen", "c_code11"))
    k = np.cos(np.radians(22.0))
    x0, y1, sc = 68.0, 37.2, 10.0  # map origin and pixels per projected degree

    def path(geom):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        out = []
        for p in polys:
            if p.area < 0.002:
                continue
            pts = [(f"{(x - x0) * k * sc:.1f}", f"{(y1 - y) * sc:.1f}") for x, y in p.exterior.coords]
            out.append("M" + "L".join(f"{a},{b}" for a, b in pts) + "Z")
        return "".join(out)

    inside, outside = {}, []
    for rec, shp in zip(sf.records(), sf.shapes()):
        code = int(rec[fields.index(key)])
        g = shape(shp.__geo_interface__).simplify(0.03, preserve_topology=True)
        d = path(g)
        if not d:
            continue
        if code in codes:
            inside[str(code)] = d
        else:
            outside.append(d)
    return inside, outside


def main():
    tab = pd.read_csv(ROOT / "data" / "processed" / "analysis_table.csv")
    ex = pd.read_csv(RES / "district_explanations.csv")
    pr = pd.read_csv(RES / "predictions" / "B-ACPN (proposed)__scaled.csv")
    pr = pr[pr.kind == "cv"].groupby("censuscode").prob.mean()
    t5 = pd.read_csv(RES / "table5_comparison.csv").set_index("model")
    sig = pd.read_csv(RES / "significance.csv").set_index("model")
    cm = pd.read_csv(RES / "confusion_matrices.csv").set_index("model")
    full = pd.read_csv(RES / "classifier_metrics_full.csv").set_index("model")
    pm, ar, hd, km = j("preprocessing_metrics.json"), j("analysis_results.json"), j("hdfs_usage_bytes.json"), \
        j("kmeans_regimes.json")
    checks, push, ceew = j("preprocessing_colab_checks.json"), j("table6_climate_push.json"), j("ceew_comparison.json")
    tm = pd.read_csv(RES / "scalability_timings.csv").groupby("years").agg(rt=("runtime_s", "mean"),
                                                                          rec=("map_input_records", "first"),
                                                                          out=("reduce_output_records", "first"))
    m = tab.merge(ex.drop(columns=["district_name", "state_name", "quadrant", "vul_group"]), on="censuscode")
    districts = []
    for r in m.itertuples():
        districts.append({
            "c": int(r.censuscode), "n": r.district_name, "s": r.state_name, "q": r.quadrant, "y": r.vul_group,
            "dg": r.dev_group, "idx": round(r.climate_vul_in, 3), "p": round(float(pr[r.censuscode]), 3),
            "b": round(r.baseline, 2), "u": round(r.push, 2), "sd": round(r.push_mc_sd, 2), "reg": int(r.regime),
            "top": HAZN[r.top_factor][0], "att": [round(getattr(r, f"att_{h}"), 3) for h in HAZ],
            "dev": [round(getattr(r, d), 1) for d in DEV], "haz": [round(getattr(r, h), 1) for h in HAZ]})
    models = []
    for mo in MODELS:
        c = cm.loc[mo]
        models.append({"name": SHORT[mo], "tool": t5.loc[mo, "tool"], "auc": round(t5.loc[mo, "roc_auc"], 3),
                       "sd": round(t5.loc[mo, "roc_auc_sd"], 3), "acc": round(t5.loc[mo, "accuracy"], 3),
                       "f1": round(t5.loc[mo, "f1"], 3), "loso": round(t5.loc[mo, "loso_auc"], 3),
                       "lo": round(sig.loc[mo, "ci_low"], 3), "hi": round(sig.loc[mo, "ci_high"], 3),
                       "p": None if pd.isna(sig.loc[mo, "p_value"]) else round(sig.loc[mo, "p_value"], 3),
                       "prec": round(full.loc[mo, "precision"], 3), "rec": round(full.loc[mo, "recall"], 3),
                       "cm": [int(c.tn), int(c.fp), int(c.fn), int(c.tp)]})
    inside, outside = map_paths(set(m.censuscode))
    agg = checks["area_vs_nearest_cell"]["metrics"]
    q = ar["quadrants"]
    D = {
        "kpi": {"districts": len(m), "states": int(m.state_name.nunique()), "rows": pm["district_day"]["rows"],
                "hdfs_gb": round(sum(hd.values()) / 1e9, 2), "auc": models[0]["auc"], "acc": models[0]["acc"],
                "f1": models[0]["f1"], "anom": ar["anomalies"], "anom_pct": round(ar["anomaly_share"] * 100, 1),
                "r2dev": ar["regression"]["m1_dev_only"]["r2"], "r2both": ar["regression"]["m2_dev_climate"]["r2"]},
        "hdfs": {k: round(v / 1e6, 1) for k, v in hd.items()},
        "timings": [{"years": int(y), "rt": round(r.rt), "rec": int(r.rec), "out": int(r.out)} for y, r in tm.iterrows()],
        "km": {"k": km["k"], "sil": round(km["silhouette"], 3), "sizes": km["sizes"]},
        "prep": {"funnel": [["Census 2011 districts", 640], ["Matched to NDAP", pm["matching"]["matched"]],
                            ["With a vulnerability index", ar["preprocessing"]["with_index"]],
                            ["Complete on all 13 features", pm["final_table"]["rows"]]],
                 "ndap_missing": pm["ndap"]["overall_missing_pct"], "unmatched": pm["matching"]["match_types"]["unmatched"],
                 "match": pm["matching"]["match_types"], "pairs": pm["grid_rain"]["pairs"] + pm["grid_temp"]["pairs"],
                 "rain_missing": round(pm["district_day"]["rain_missing_pct"], 2),
                 "lit": pm["census"]["literacy_pct"], "lit_official": pm["census"]["literacy_official"],
                 "agg": {k: round(agg[k], 4) for k in ("rmse", "mae", "mape_pct", "r2", "medae", "bias")},
                 "mr_rows": checks["mapreduce_vs_pandas"]["rows_compared"]},
        "quad": {k: {"n": q[k], "push": round(push["by_quadrant"][k]["push"], 2)} for k in q},
        "p_anom": [push["anomaly_A_vs_expected_p"], push["anomaly_B_vs_expected_p"]],
        "ceew": {"n": ceew["in_our_469"], "rho": ceew["spearman_ceew_vi_vs_ndap_cvi"]["rho"]},
        "models": models, "districts": districts, "haz": [HAZN[h] for h in HAZ], "dev": [DEVN[d] for d in DEV],
        "map": {"in": inside, "out": outside},
    }
    body = TEMPLATE.replace("/*DATA*/", "const D=" + json.dumps(D, separators=(",", ":"), ensure_ascii=False) + ";")
    (ROOT / "demo" / "artifact.html").write_text(body, encoding="utf-8")
    full_doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                + body.replace("<body-start/>", "</head><body>") + "\n</body></html>\n")
    (ROOT / "demo" / "index.html").write_text(full_doc, encoding="utf-8")
    print("written demo/index.html", f"{len(full_doc) / 1e6:.2f} MB,", len(districts), "districts")


TEMPLATE = (Path(__file__).resolve().parent / "template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
