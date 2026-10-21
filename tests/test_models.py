"""Tests for A-CPN, the MLP baseline and the shared evaluation folds."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.acpn import ACPN, ACPNClassifier, KANLayer  # noqa: E402
from models.mlp import MLPClassifier  # noqa: E402
from scripts.common import FEATURES, load_table, make_folds  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_table()


def test_target_never_an_input(data):
    df, X, y = data
    assert X.shape == (469, 13)
    assert "climate_vul_in" not in FEATURES and "vul_group" not in FEATURES


def test_kan_bases_partition_of_unity():
    k = KANLayer(3, 2)
    x = torch.linspace(-2.9, 2.9, 50).repeat(3, 1).T
    assert torch.allclose(k.bases(x).sum(-1), torch.ones(50, 3), atol=1e-5)


def test_acpn_forward_shapes():
    net = ACPN().eval()
    logit, b, c, a = net(torch.randn(7, 5), torch.randn(7, 8))
    assert logit.shape == b.shape == c.shape == (7,)
    assert torch.allclose(logit, b + c)
    assert torch.allclose(a.sum(1), torch.ones(7), atol=1e-5)


@pytest.mark.parametrize("arch", [{}, {"use_attention": False}, {"use_kan": False}, {"use_residual": False},
                                  {"use_dev": False}, {"use_climate": False}, {"linear_climate": True}])
def test_ablation_variants_run(arch):
    logit = ACPN(**arch).eval()(torch.randn(4, 5), torch.randn(4, 8))[0]
    assert logit.shape == (4,)


def test_development_only_has_zero_push():
    _, _, c, _ = ACPN(use_climate=False).eval()(torch.randn(4, 5), torch.randn(4, 8))
    assert torch.all(c == 0)


def test_classifier_fit_predict_explain(data):
    df, X, y = data
    m = ACPNClassifier().fit(X[:300], y[:300])
    p = m.predict_proba(X[300:])
    assert p.shape == (169, 2) and np.allclose(p.sum(1), 1)
    e = m.explain(X[:300])
    assert abs(e["push"].mean()) < 1e-4  # push centred on the training districts
    assert np.allclose(m.predict_proba(X[300:]), p)  # deterministic inference after MC Dropout
    s = m.conformal_sets(X[300:])
    assert s.shape == (169, 2) and s.any(1).all()


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
