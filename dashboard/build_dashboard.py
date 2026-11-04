"""Build dashboard/index.html: an offline district explorer for the B-ACPN results.

Every number on the page is read from results/ (no network, no external libraries).
Usage: python dashboard/build_dashboard.py      then open dashboard/index.html in a browser.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
HAZ = ["annual_rain_mm", "rain_variability", "rx5day_mm", "heavy_rain_days", "max_dry_spell", "mean_tmax",
       "hot_days", "tropical_nights"]
NICE = {"annual_rain_mm": "Annual rain", "rain_variability": "Rain variability", "rx5day_mm": "Max 5-day rain",
        "heavy_rain_days": "Heavy-rain days", "max_dry_spell": "Longest dry spell", "mean_tmax": "Mean Tmax",
        "hot_days": "Hot days", "tropical_nights": "Tropical nights"}


def main():
    e = pd.read_csv(RES / "district_explanations.csv")
    t5 = pd.read_csv(RES / "table5_comparison.csv").set_index("model")
    push = json.loads((RES / "table6_climate_push.json").read_text())
    hd = json.loads((RES / "hdfs_usage_bytes.json").read_text())
    rows = []
    for r in e.itertuples():
        rows.append({"d": r.district_name, "s": r.state_name, "q": r.quadrant, "v": r.vul_group,
                     "b": round(r.baseline, 3), "p": round(r.push, 3), "u": round(r.push_mc_sd, 3),
                     "f": NICE.get(r.top_factor, r.top_factor), "a": [round(getattr(r, f"att_{h}"), 4) for h in HAZ]})
    a = t5.loc["B-ACPN (proposed)"]
    meta = {"auc": round(a.roc_auc, 3), "acc": round(a.accuracy, 3), "f1": round(a.f1, 3), "n": len(rows),
            "hdfs": round(sum(hd.values()) / 1e9, 2), "haz": [NICE[h] for h in HAZ],
            "quad": {k: round(v["push"], 3) for k, v in push["by_quadrant"].items()}}
    html = TEMPLATE.replace("/*DATA*/", f"const D={json.dumps(rows)};const META={json.dumps(meta)};")
    (ROOT / "dashboard" / "index.html").write_text(html, encoding="utf-8")
    print("written dashboard/index.html with", len(rows), "districts")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Climate Push Explorer</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b2430;--mut:#5b6573;--line:#e3e7ec;--navy:#1b3a6b;--teal:#0e6e6e;--red:#c0392b;--blue:#2e6fb7}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--navy);color:#fff;padding:14px 22px;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:19px;margin:0}header span{font-size:13px;opacity:.85}
main{padding:18px 22px;max-width:1280px;margin:auto}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi b{display:block;font-size:22px;color:var(--navy)}.kpi small{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:1.05fr 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card h2{font-size:15px;margin:0 0 10px;color:var(--navy)}
select,input{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:6px;width:100%;margin-bottom:10px}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px}
.fact{background:var(--bg);border-radius:6px;padding:8px 10px}.fact small{color:var(--mut);font-size:11px;display:block}.fact b{font-size:16px}
.verdict{padding:8px 10px;border-radius:6px;font-size:13px;margin-bottom:10px}
.bar{display:flex;align-items:center;font-size:12px;margin:4px 0}.bar span{width:120px;color:var(--mut)}
.bar div{height:12px;background:var(--teal);border-radius:3px}.bar em{font-style:normal;margin-left:6px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px}td.n{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--red);font-weight:600}.neg{color:var(--blue);font-weight:600}
</style></head><body>
<header><h1>Climate Push Explorer &middot; B-ACPN</h1><span>Beyond Development: Explaining Climate Resilience Across Indian Districts</span></header>
<main>
<div class="kpis" id="kpis"></div>
<div class="grid">
 <div class="card"><h2>District explanation</h2>
  <input id="q" placeholder="Search a district or state&hellip;"><select id="sel"></select>
  <div class="facts" id="facts"></div><div class="verdict" id="verdict"></div>
  <h2 style="margin-top:6px">Which climate factor drives the push (attention)</h2><div id="att"></div></div>
 <div class="card"><h2>Most climate-driven districts (largest positive push)</h2><table id="top"></table>
  <h2 style="margin-top:14px">Mean climate push by group</h2><div id="quad"></div></div>
</div></main>
<script>
/*DATA*/
const $=id=>document.getElementById(id), f=x=>(x>0?"+":"")+x.toFixed(2);
$("kpis").innerHTML=[["Districts",META.n],["B-ACPN ROC-AUC",META.auc],["Accuracy",META.acc],["F1-score",META.f1],["Data in HDFS",META.hdfs+" GB"]]
 .map(([k,v])=>`<div class="kpi"><small>${k}</small><b>${v}</b></div>`).join("");
const sorted=[...D].sort((a,b)=>a.d.localeCompare(b.d));
function fill(filter){const s=$("sel");const keep=s.value;s.innerHTML=sorted.filter(r=>(r.d+" "+r.s).toLowerCase().includes(filter))
 .map(r=>`<option value="${D.indexOf(r)}">${r.d} (${r.s})</option>`).join("");if([...s.options].some(o=>o.value==keep))s.value=keep;show();}
function show(){const r=D[+$("sel").value];if(!r)return;
 $("facts").innerHTML=[["Group",r.q],["Vulnerability",r.v],["Baseline (logit)",f(r.b)],["Climate push",`<span class="${r.p>0?"pos":"neg"}">${f(r.p)}</span>`],
  ["Uncertainty (MC SD)",r.u.toFixed(2)],["Top factor",r.f]].map(([k,v])=>`<div class="fact"><small>${k}</small><b>${v}</b></div>`).join("");
 const v=$("verdict");v.style.background=r.p>0?"#fbeaea":"#e8f0fa";
 v.textContent=r.p>0?`Climate adds risk in ${r.d} beyond what its development explains (main factor: ${r.f}).`:`Climate lowers the risk of ${r.d} relative to its development level.`;
 const m=Math.max(...r.a);$("att").innerHTML=META.haz.map((h,i)=>`<div class="bar"><span>${h}</span><div style="width:${(r.a[i]/m*220).toFixed(0)}px"></div><em>${r.a[i].toFixed(3)}</em></div>`).join("");}
$("top").innerHTML="<tr><th>District</th><th>State</th><th>Group</th><th class='n'>Push</th></tr>"+[...D].sort((a,b)=>b.p-a.p).slice(0,8)
 .map(r=>`<tr><td>${r.d}</td><td>${r.s}</td><td>${r.q}</td><td class="n pos">${f(r.p)}</td></tr>`).join("");
const qn={"High dev / Low vul":"Developed, low vul. (expected)","Low dev / High vul":"Less developed, high vul. (expected)","High dev / High vul":"Anomaly A: developed yet vulnerable","Low dev / Low vul":"Anomaly B: less developed yet resilient"};
$("quad").innerHTML=Object.entries(qn).map(([k,n])=>{const x=META.quad[k];return `<div class="bar"><span style="width:230px">${n}</span><div style="width:${Math.abs(x)*600}px;background:${x>0?"var(--red)":"var(--blue)"}"></div><em>${f(x)}</em></div>`}).join("");
$("q").addEventListener("input",e=>fill(e.target.value.toLowerCase()));$("sel").addEventListener("change",show);
fill("");const start=new URLSearchParams(location.search).get("d");const i=D.findIndex(r=>r.d===start);$("sel").value=i>=0?i:D.indexOf(sorted[0]);show();
</script></body></html>
"""

if __name__ == "__main__":
    main()
