"""Build the integrated socio-economic district table (the analysis base).

Design decision recorded here: the NDAP 2021 file is too sparse to supply all
11 planned development indicators — no district has all of them, and keeping
six collapses the sample to six states. The development score therefore uses
the three complete Census 2011 variables plus the two best-covered NDAP
variables, giving 469 districts across 21 states.

Columns produced
    censuscode              all-India Census 2011 district code (join key)
    district_name, state_name
    climate_vul_in          OUTCOME - NDAP 2021 climate vulnerability index
    literacy_rate, female_literacy_rate, agri_worker_share      Census 2011
    mgnrega_days, yield_variability                             NDAP 2021
    z_*                     standardised, sign-aligned so higher = more developed
    development_score       mean of the five z-columns
    dev_group, vul_group    High/Low by median split
    quadrant                the four development x vulnerability patterns

Sign alignment: a higher raw value means more developed for literacy and
female literacy; it means less developed for agricultural worker share and
yield variability, so those two are negated.

MGNREGA days is deliberately NOT negated: this study follows the NDAP/CEEW
framing in which a functioning employment guarantee is adaptive capacity. The
opposite reading — that high uptake signals rural distress — is defensible,
and Section 10 of the report should say which reading was taken. Flip
MGNREGA_HIGHER_IS_BETTER to test the alternative.
"""
import numpy as np
import pandas as pd

NDAP = "data/raw/ndap_climate_vulnerability_2021.csv"
CROSSWALK = "data/processed/ndap_census_district_crosswalk.csv"
CENSUS = "data/processed/census2011_district_variables.csv"
OUT = "data/processed/integrated_district_table.csv"

MGNREGA_HIGHER_IS_BETTER = True

# raw column -> (output name, +1 if higher means more developed, else -1)
DEV = {
    "literacy_rate":        ("literacy_rate", +1),
    "female_literacy_rate": ("female_literacy_rate", +1),
    "agri_worker_share":    ("agri_worker_share", -1),
    "avg_days_employement": ("mgnrega_days", +1 if MGNREGA_HIGHER_IS_BETTER else -1),
    "yield_variability":    ("yield_variability", -1),
}


def main():
    nd = pd.read_csv(NDAP, encoding_errors="replace")
    xw = pd.read_csv(CROSSWALK)
    cv = pd.read_csv(CENSUS)

    xw = xw[xw.match_type != "unmatched"][
        ["district_code", "district_name", "state_name", "censuscode", "match_type"]]
    nd = nd.merge(xw, on=["district_code", "district_name", "state_name"], how="inner")

    df = nd[["censuscode", "district_name", "state_name", "climate_vul_in",
             "avg_days_employement", "yield_variability", "match_type"]].copy()
    df["censuscode"] = df["censuscode"].astype(int)
    df = df.merge(
        cv[["censuscode", "literacy_rate", "female_literacy_rate", "agri_worker_share"]],
        on="censuscode", how="left")

    before = len(df)
    need = ["climate_vul_in"] + list(DEV)
    df = df.dropna(subset=need).copy()
    df = df.rename(columns={"avg_days_employement": "mgnrega_days"})

    # standardise and sign-align
    zcols = []
    for raw, (name, sign) in DEV.items():
        col = "mgnrega_days" if raw == "avg_days_employement" else name
        z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df[f"z_{name}"] = (sign * z).round(4)
        zcols.append(f"z_{name}")

    df["development_score"] = df[zcols].mean(axis=1).round(4)

    dev_med = df.development_score.median()
    vul_med = df.climate_vul_in.median()
    df["dev_group"] = np.where(df.development_score >= dev_med, "High", "Low")
    df["vul_group"] = np.where(df.climate_vul_in >= vul_med, "High", "Low")
    df["quadrant"] = df.dev_group + " dev / " + df.vul_group + " vul"

    cols = (["censuscode", "district_name", "state_name", "climate_vul_in",
             "literacy_rate", "female_literacy_rate", "agri_worker_share",
             "mgnrega_days", "yield_variability"] + zcols +
            ["development_score", "dev_group", "vul_group", "quadrant", "match_type"])
    df = df[cols].sort_values("censuscode")
    df.to_csv(OUT, index=False)

    # ---------------- report ----------------
    print(f"matched districts in                 : {before}")
    print(f"complete on outcome + 5 dev variables: {len(df)}")
    print(f"states                               : {df.state_name.nunique()}")
    print(f"duplicate censuscodes                : {df.censuscode.duplicated().sum()}")
    print(f"missing values in output             : {int(df[need[:1]+list(df.filter(like='z_'))].isna().sum().sum())}")
    print(f"\nmedians  development_score {dev_med:.4f}   climate_vul_in {vul_med:.4f}")
    print("\nquadrants:")
    print(df.quadrant.value_counts().to_string())
    print("\ncorrelation, development score vs vulnerability index:")
    print(f"  Pearson  {df.development_score.corr(df.climate_vul_in):+.3f}")
    print(f"  Spearman {df.development_score.corr(df.climate_vul_in, method='spearman'):+.3f}")
    print("\ntop 10 states by district count:")
    print(df.state_name.value_counts().head(10).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
