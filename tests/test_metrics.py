"""The 8 preprocessing error metrics agree with scikit-learn."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             median_absolute_error, r2_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.colab_checks import reg_metrics  # noqa: E402


def test_reg_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    ref = rng.uniform(100, 3000, 200)
    pred = ref + rng.normal(5, 80, 200)
    m = reg_metrics(ref, pred)
    assert np.isclose(m["mse"], mean_squared_error(ref, pred))
    assert np.isclose(m["rmse"], np.sqrt(mean_squared_error(ref, pred)))
    assert np.isclose(m["mae"], mean_absolute_error(ref, pred))
    assert np.isclose(m["mape_pct"], 100 * mean_absolute_percentage_error(ref, pred))
    assert np.isclose(m["r2"], r2_score(ref, pred))
    assert np.isclose(m["medae"], median_absolute_error(ref, pred))
    assert np.isclose(m["nrmse"], m["rmse"] / (ref.max() - ref.min()))
    assert np.isclose(m["bias"], (pred - ref).mean())


def test_identical_values_give_zero_error():
    ref = np.array([0.0, 1.0, 2.0, 5.0])
    m = reg_metrics(ref, ref.copy())
    assert m["rmse"] == m["mae"] == m["medae"] == m["bias"] == 0 and m["r2"] == 1
