"""Tests for B-ACPN, its building blocks, the MLP baseline and the shared evaluation folds."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bacpn import BACPNClassifier, DeepStage  # noqa: E402
from models.layers import FeatureAttention  # noqa: E402
from models.mlp import MLPClassifier  # noqa: E402
from scripts.common import FEATURES, load_table, make_folds  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_table()


def test_target_never_an_input(data):
    df, X, y = data
    assert X.shape == (469, 13)
    assert "climate_vul_in" not in FEATURES and "vul_group" not in FEATURES


def test_attention_weights_sum_to_one():
    _, a = FeatureAttention(8)(torch.randn(6, 8))
    assert torch.allclose(a.sum(1), torch.ones(6), atol=1e-5)


def test_mlp_runs(data):
    df, X, y = data
    p = MLPClassifier().fit(X[:300], y[:300]).predict_proba(X[300:])
    assert p.shape == (169, 2)


def test_folds_are_disjoint_and_complete(data):
    df, _, y = data
    folds = make_folds(df, y)
    assert len(folds["cv"]) == 25 and len(folds["loso"]) == df.state_name.nunique()
    for f in folds["cv"] + folds["loso"]:
        assert not set(f["train"]) & set(f["test"])
        assert len(f["train"]) + len(f["test"]) == 469
    for r in range(5):
        tests = sum((folds["cv"][r * 5 + k]["test"] for k in range(5)), [])
        assert sorted(tests) == sorted(df.censuscode)


@pytest.mark.parametrize("arch", [{}, {"use_attention": False}, {"use_gate": False}, {"use_climate": False}])
def test_bacpn_deep_stage_variants(arch):
    logit, b, c, a = DeepStage(**arch).eval()(torch.randn(4, 14))
    assert logit.shape == b.shape == c.shape == (4,)
    assert torch.allclose(logit, b + c)
    if not arch.get("use_climate", True):
        assert torch.all(c == 0)


def test_bacpn_fit_predict_explain(data):
    df, X, y = data
    m = BACPNClassifier(seeds=(42, 1), n_trees=50, mc_passes=4).fit(X[:300], y[:300])
    p = m.predict_proba(X[300:])
    assert p.shape == (169, 2) and np.allclose(p.sum(1), 1)
    e = m.explain(X[300:])
    assert e["push"].shape == (169,) and e["attention"].shape == (169, 8)
    assert np.allclose(m.predict_proba(X[300:]), p)  # deterministic inference after MC Dropout
    assert m.conformal_sets(X[300:]).shape == (169, 2)
