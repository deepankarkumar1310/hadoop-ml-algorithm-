"""Build district-level Census 2011 variables for the climate vulnerability project.

Inputs : mirrored all-India district PCA CSV (640 districts, Total/Rural/Urban rows)
Outputs: census2011_district_variables.csv  (one row per district, Total only)

Derived variables
  literacy_rate         = P_LIT / (TOT_P - P_06)          * 100   (Census 7+ definition)
  female_literacy_rate  = F_LIT / (TOT_F - F_06)          * 100
  agri_worker_share     = (cultivators + agri labourers, main + marginal) / TOT_WORK_P * 100
"""
import sys
import pandas as pd
import shapefile

PCA = "data/raw/census/pca-full.csv"
SHP = "data/raw/boundaries/2011_Dist.shp"
OUT = "data/processed/census2011_district_variables.csv"

df = pd.read_csv(PCA, dtype={"State": str, "District": str})
d = df[df["TRU"] == "Total"].copy()

d["state_code"] = d["State"].str.zfill(2)
# The PCA "District" column is the all-India census code (1-640), the same key
# used by the 2011 district boundary shapefile's `censuscode` field.
d["censuscode"] = d["District"].astype(int)
d["district_name"] = d["Name"].str.strip()

pop7 = d["TOT_P"] - d["P_06"]
fpop7 = d["TOT_F"] - d["F_06"]
agri = d["MAIN_CL_P"] + d["MAIN_AL_P"] + d["MARG_CL_P"] + d["MARG_AL_P"]

d["literacy_rate"] = (d["P_LIT"] / pop7 * 100).round(2)
d["female_literacy_rate"] = (d["F_LIT"] / fpop7 * 100).round(2)
d["agri_worker_share"] = (agri / d["TOT_WORK_P"] * 100).round(2)

out = d[[
    "censuscode", "state_code", "district_name",
    "TOT_P", "TOT_F", "P_LIT", "F_LIT", "TOT_WORK_P",
    "literacy_rate", "female_literacy_rate", "agri_worker_share",
]].rename(columns={
    "TOT_P": "total_population", "TOT_F": "female_population",
    "P_LIT": "literates", "F_LIT": "female_literates",
    "TOT_WORK_P": "total_workers",
})

# --- sanity checks -------------------------------------------------------
problems = []
if len(out) != 640:
    problems.append(f"expected 640 districts, got {len(out)}")
if out[["literacy_rate", "female_literacy_rate", "agri_worker_share"]].isna().any().any():
    problems.append("missing values present in derived variables")
for col in ("literacy_rate", "female_literacy_rate", "agri_worker_share"):
    bad = out[(out[col] < 0) | (out[col] > 100)]
    if len(bad):
        problems.append(f"{col}: {len(bad)} values outside 0-100")

# --- join test against the 2011 district boundaries ----------------------
r = shapefile.Reader(SHP)
recs = pd.DataFrame(r.records(), columns=[f[0] for f in r.fields[1:]])
recs["censuscode"] = recs["censuscode"].astype(int)
merged = recs.merge(out.drop(columns=["state_code"]), on="censuscode", how="left", indicator=True)
matched = (merged["_merge"] == "both").sum()

out.to_csv(OUT, index=False)

print(f"districts written      : {len(out)}")
print(f"national population    : {out.total_population.sum():,}")
print(f"national literacy rate : {out.literates.sum() / (df[df.TRU=='Total'].TOT_P.sum() - df[df.TRU=='Total'].P_06.sum()) * 100:.2f}%")
print(f"boundary shapes        : {len(recs)}")
print(f"boundary join matched  : {matched}/{len(recs)}")
if matched < len(recs):
    print("unmatched boundary districts:")
    print(merged.loc[merged._merge != "both", ["DISTRICT", "ST_NM", "censuscode"]].to_string(index=False))
print("range check:")
print(out[["literacy_rate", "female_literacy_rate", "agri_worker_share"]].describe().loc[["min", "mean", "max"]].round(2).to_string())
print("PROBLEMS:", problems if problems else "none")
print("written ->", OUT)
