"""Shared data loading and the fixed evaluation folds used by every model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "analysis_table.csv"
RES = ROOT / "results"
PRED = RES / "predictions"

DEV = ["literacy_rate", "female_literacy_rate", "agri_worker_share", "mgnrega_days", "yield_variability"]
HAZ = ["annual_rain_mm", "rain_variability", "rx5day_mm", "heavy_rain_days", "max_dry_spell",
       "mean_tmax", "hot_days", "tropical_nights"]
FEATURES = DEV + HAZ
SEED = 42


def load_table(path=DATA):
    """469 districts x 13 features; target = High vulnerability (median split of the 2021 index)."""
    df = pd.read_csv(path).dropna(subset=FEATURES).reset_index(drop=True)
    y = (df.vul_group == "High").astype(int).to_numpy()
    return df, df[FEATURES].to_numpy(dtype=float), y


def make_folds(df, y):
    """25 folds (5-fold x 5 repeats, stratified, seed 42) + leave-one-state-out, as censuscode lists."""
    cv = []
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(df, y)):
        cv.append({"fold": k, "train": df.censuscode.iloc[tr].tolist(), "test": df.censuscode.iloc[te].tolist()})
    loso = [{"fold": f"LOSO:{s}", "state": s,
             "train": df.censuscode[df.state_name != s].tolist(), "test": df.censuscode[df.state_name == s].tolist()}
            for s in sorted(df.state_name.unique())]
    return {"cv": cv, "loso": loso}


def get_folds(df, y):
    p = RES / "folds.json"
    if p.exists():
        return json.loads(p.read_text())
    folds = make_folds(df, y)
    RES.mkdir(exist_ok=True)
    p.write_text(json.dumps(folds))
    return folds


def index_of(df):
    pos = {c: i for i, c in enumerate(df.censuscode)}
    return lambda codes: np.array([pos[c] for c in codes])
