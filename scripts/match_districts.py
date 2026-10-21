"""Crosswalk NDAP 2021 districts -> Census 2011 all-India district codes (1-640).

The two sources disagree on district identity in four ways, each handled below:

1. State reorganisation. Census 2011 predates Telangana (2014) and the Ladakh UT
   (2019), so its Andhra Pradesh contains today's Telangana districts and its
   Jammu & Kashmir contains Leh and Kargil. Matching is therefore done inside
   state *clusters*, not exact state names.
2. Transliteration variants. Directional prefixes appear in both Hindi and
   English (Purbi/Purba/Purv = East, Pashchim/Paschim = West, Uttar = North,
   Dakshin = South), and some districts were officially renamed after 2011.
3. Spacing and punctuation. "Bara Banki" vs "Barabanki", "Leh (ladakh)" vs
   "Leh", ampersands vs "and".
4. New districts. Many districts were carved out of 2011 parents after 2011,
   mostly in the north-east. These have no 2011 equivalent and are reported as
   unmatched rather than guessed at.

Matching is attempted in three passes and every row records which pass matched
it, so a reviewer can check the weaker ones:
    exact  - normalised name identical inside the state cluster
    alias  - matched after applying the rename/variant table
    fuzzy  - difflib ratio >= FUZZY_MIN inside the state cluster (review these)
"""
import difflib
import re
import sys

import pandas as pd
import shapefile

NDAP = "data/raw/ndap_climate_vulnerability_2021.csv"
CENSUS = "data/processed/census2011_district_variables.csv"
SHP = "data/raw/boundaries/2011_Dist.shp"
OUT = "data/processed/ndap_census_district_crosswalk.csv"

FUZZY_MIN = 0.86

# States that were one state in 2011 share a cluster key.
STATE_CLUSTER = {
    "telangana": "andhrapradesh",
    "andhrapradesh": "andhrapradesh",
    "ladakh": "jammukashmir",
    "jammukashmir": "jammukashmir",
    "uttaranchal": "uttarakhand",
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "arunanchalpradesh": "arunachalpradesh",
}

# Directional and common-word equivalences, applied token-wise.
WORD_ALIAS = {
    "purbi": "east", "purba": "east", "purv": "east", "purvi": "east",
    "pashchim": "west", "paschim": "west", "pachim": "west", "pashchimi": "west",
    "uttar": "north", "uttari": "north",
    "dakshin": "south", "dakshina": "south", "dakshini": "south",
    "madhya": "central",
    "and": "", "the": "",
    "paraganas": "parganas", "parganas": "parganas",
}

# Districts renamed between 2011 and 2021. NDAP (2021) name -> Census 2011 name.
RENAME = {
    # Punjab
    "sasnagar": "sahibzadaajitsinghnagar", "mohali": "sahibzadaajitsinghnagar",
    "nawanshahr": "shahidbhagatsinghnagar", "srimuktsarsahib": "muktsar",
    # Uttar Pradesh
    "prayagraj": "allahabad", "ayodhya": "faizabad", "kasganj": "kanshiramnagar",
    "bhadohi": "santravidasnagarbhadohi", "santravidasnagar": "santravidasnagarbhadohi",
    "amethi": "chhatrapatisahujimaharajnagar", "hapur": "panchsheelnagar",
    "sambhal": "bhimnagar", "shamli": "prabuddhnagar",
    # Chhattisgarh
    "dantewada": "dakshinbastardantewada", "southbastardantewada": "dakshinbastardantewada",
    "kanker": "uttarbastarkanker", "northbastarkanker": "uttarbastarkanker",
    "kabirdham": "kabeerdham", "korea": "koriya",
    # West Bengal
    "coochbehar": "kochbihar", "malda": "maldah", "darjeeling": "darjiling",
    "howrah": "haora", "hooghly": "hugli", "bardhaman": "barddhaman",
    "burdwan": "barddhaman",
    # Andhra Pradesh / Telangana
    "spsrnellore": "sripottisriramulunellore", "nellore": "sripottisriramulunellore",
    "kadapa": "ysr", "ysrkadapa": "ysr", "mahabubnagar": "mahbubnagar",
    # Maharashtra / Karnataka renames
    "ahilyanagar": "ahmadnagar", "ahmednagar": "ahmadnagar",
    "chhatrapatisambhajinagar": "aurangabad", "dharashiv": "osmanabad",
    "bengaluruurban": "bangalore", "bengalururural": "bangalorerural",
    "belagavi": "belgaum", "ballari": "bellary", "vijayapura": "bijapur",
    "kalaburagi": "gulbarga", "mysuru": "mysore", "shivamogga": "shimoga",
    "tumakuru": "tumkur", "chikkamagaluru": "chikmagalur",
    "chamarajanagara": "chamarajanagar",
    # Haryana / Gujarat / J&K / Assam
    "gurugram": "gurgaon", "nuh": "mewat",
    "leh": "lehladakh",
    "kamrupmetro": "kamrupmetropolitan",
    # Odisha
    "balasore": "baleshwar", "jajpur": "jajapur", "sonepur": "subarnapur",
    "nabarangpur": "nabarangapur", "keonjhar": "kendujhar", "boudh": "baudh",
    "anugul": "angul", "bolangir": "balangir",
    # Bihar
    "kaimur": "kaimurbhabua",
    # spellings verified against the Census 2011 name list
    "tuticorin": "thoothukkudi", "hathras": "mahamayanagar",
    "narsinghpur": "narsimhapur", "beed": "bid", "raigad": "raigarh",
    "deogarh": "debagarh", "dholpur": "dhaulpur", "koderma": "kodarma",
    "paurigarhwal": "garhwal", "shopian": "shupiyan", "budgam": "badgam",
    "poonch": "punch", "eastnimar": "khandwa", "westnimar": "khargone",
}


def sorted_key(s: str) -> str:
    """Alphabetically ordered tokens: matches names that differ only in word
    order ("Dinajpur Dakshin" vs "Dakshin Dinajpur")."""
    import re as _re
    t = str(s).lower().replace("24", " twenty four ")
    t = _re.sub(r"\(.*?\)", " ", t)
    t = _re.sub(r"[^a-z\s]", " ", t)
    toks = sorted(WORD_ALIAS.get(x, x) for x in t.split())
    return "".join(x for x in toks if x)

# Districts created after 2011 that a fuzzy match would wrongly attach to an
# unrelated 2011 district. Checked by hand against the fuzzy output.
NEVER_MATCH = {
    "sangareddy",          # carved from Medak 2016, not Rangareddy
    "southwestgarohills",  # carved from West Garo Hills 2012, not South Garo Hills
    "northgarohills", "southwestkhasihills",
}


def norm(s: str) -> str:
    """Lowercase, drop everything but letters."""
    return re.sub(r"[^a-z]", "", str(s).lower())


def norm_name(s: str) -> str:
    """Normalise a district name, applying token-level word aliases."""
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)              # drop parenthetical qualifiers
    s = s.replace("24", " twenty four ")
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = [WORD_ALIAS.get(t, t) for t in s.split()]
    return "".join(t for t in toks if t)


def state_key(s: str) -> str:
    k = norm(re.sub(r"\band\b|&", " ", str(s).lower()))
    return STATE_CLUSTER.get(k, k)


def load_census():
    """Census variables joined to state names from the boundary shapefile."""
    cv = pd.read_csv(CENSUS)
    r = shapefile.Reader(SHP)
    fields = [f[0] for f in r.fields[1:]]
    recs = pd.DataFrame(r.records(), columns=fields)
    recs = recs[recs["censuscode"].astype(int) > 0]
    recs["censuscode"] = recs["censuscode"].astype(int)
    cv = cv.merge(recs[["censuscode", "ST_NM"]], on="censuscode", how="left")
    cv = cv.rename(columns={"ST_NM": "state_name"})
    cv["skey"] = cv["state_name"].map(state_key)
    cv["nkey"] = cv["district_name"].map(norm_name)
    return cv


def main():
    nd = pd.read_csv(NDAP, encoding_errors="replace")
    cv = load_census()

    nd["skey"] = nd["state_name"].map(state_key)
    nd["nkey_raw"] = nd["district_name"].map(norm_name)
    nd["nkey"] = nd["nkey_raw"].map(lambda k: RENAME.get(k, k))

    # Census lookup: (state cluster, normalised name) -> censuscode
    cv_index, sort_index = {}, {}
    for _, row in cv.iterrows():
        cv_index.setdefault((row.skey, row.nkey), row.censuscode)
        cv_index.setdefault((row.skey, norm(row.district_name)), row.censuscode)
        sort_index.setdefault((row.skey, sorted_key(row.district_name)), row.censuscode)
    by_state = cv.groupby("skey").nkey.apply(list).to_dict()

    code, how, matched_to, score = [], [], [], []
    for _, row in nd.iterrows():
        hit = cv_index.get((row.skey, row.nkey_raw))
        if hit is not None:
            code.append(hit); how.append("exact"); matched_to.append(row.nkey_raw); score.append(1.0)
            continue
        hit = cv_index.get((row.skey, row.nkey))
        if hit is not None:
            code.append(hit); how.append("alias"); matched_to.append(row.nkey); score.append(1.0)
            continue
        hit = cv_index.get((row.skey, norm(row.district_name)))
        if hit is not None:
            code.append(hit); how.append("exact"); matched_to.append(norm(row.district_name)); score.append(1.0)
            continue
        hit = sort_index.get((row.skey, sorted_key(row.district_name)))
        if hit is not None:
            code.append(hit); how.append("wordorder"); matched_to.append(sorted_key(row.district_name)); score.append(1.0)
            continue
        if row.nkey in NEVER_MATCH or row.nkey_raw in NEVER_MATCH:
            code.append(pd.NA); how.append("unmatched"); matched_to.append(""); score.append(0.0)
            continue
        cands = by_state.get(row.skey, [])
        best = difflib.get_close_matches(row.nkey, cands, n=1, cutoff=FUZZY_MIN)
        if best:
            s = difflib.SequenceMatcher(None, row.nkey, best[0]).ratio()
            code.append(cv_index[(row.skey, best[0])]); how.append("fuzzy")
            matched_to.append(best[0]); score.append(round(s, 3))
            continue
        code.append(pd.NA); how.append("unmatched"); matched_to.append(""); score.append(0.0)

    nd["censuscode"] = code
    nd["match_type"] = how
    nd["matched_census_name"] = matched_to
    nd["match_score"] = score

    # A 2011 district must not receive two different NDAP rows: that means a
    # post-2011 split, where attributing either child's value to the parent
    # would be wrong. Flag, do not silently keep.
    dup_mask = nd["censuscode"].notna() & nd.duplicated("censuscode", keep=False)
    nd.loc[dup_mask, "match_type"] = nd.loc[dup_mask, "match_type"] + "_dup"

    cols = ["district_code", "district_name", "state_name", "censuscode",
            "matched_census_name", "match_type", "match_score", "climate_vul_in"]
    nd[cols].to_csv(OUT, index=False)

    # ---------------- report ----------------
    print(f"NDAP rows            : {len(nd)}")
    print(f"Census districts     : {len(cv)}")
    print("\nmatch type:")
    print(nd.match_type.value_counts().to_string())

    clean = nd[~nd.match_type.str.endswith("_dup") & nd.censuscode.notna()]
    print(f"\nusable 1:1 matches   : {len(clean)}")
    print(f"  with vulnerability index: {clean.climate_vul_in.notna().sum()}")
    print(f"Census districts covered : {clean.censuscode.nunique()} of {len(cv)}")

    fz = nd[nd.match_type.str.startswith("fuzzy")]
    if len(fz):
        print(f"\nfuzzy matches to review ({len(fz)}):")
        print(fz[["district_name", "state_name", "matched_census_name", "match_score"]]
              .sort_values("match_score").to_string(index=False))

    dups = nd[nd.match_type.str.endswith("_dup")]
    if len(dups):
        print(f"\npost-2011 splits — several NDAP districts share one 2011 parent ({len(dups)} rows, "
              f"{dups.censuscode.nunique()} parents):")
        print(dups[["district_name", "state_name", "matched_census_name"]]
              .sort_values("matched_census_name").head(30).to_string(index=False))

    un = nd[nd.match_type == "unmatched"]
    print(f"\nunmatched NDAP districts ({len(un)}) by state:")
    print(un.state_name.value_counts().to_string())

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
