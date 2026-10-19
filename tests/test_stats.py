import numpy as np
import pandas as pd

from harmonic_signatures.stats.confidence import confidence_interval_mean, correlation_with_ci
from harmonic_signatures.stats.group_tests import mann_whitney, one_way_anova, sweep_channels, welch_ttest


def _synthetic_long_df():
    rng = np.random.default_rng(0)
    rows = []
    for label, mean in [("A", 0.0), ("B", 5.0), ("C", 10.0)]:
        for order in (1, 2):
            values = rng.normal(loc=mean + order, scale=0.5, size=200)
            for i, v in enumerate(values):
                rows.append({
                    "sample_id": f"{label}_{order}_{i}", "label": label, "domain": "harmonica",
                    "family": "H", "variable": "I", "phase": "a", "order": order,
                    "statistic": "avg", "value": v,
                })
    return pd.DataFrame(rows)


def test_welch_ttest_detects_difference():
    df = _synthetic_long_df()
    result = welch_ttest(df, "A", "C", order=1)
    assert result["p_value"] < 0.001
    assert result["n_a"] == 200 and result["n_b"] == 200


def test_mann_whitney_detects_difference():
    df = _synthetic_long_df()
    result = mann_whitney(df, "A", "C", order=1)
    assert result["p_value"] < 0.001


def test_one_way_anova_across_all_labels():
    df = _synthetic_long_df()
    result = one_way_anova(df, order=1)
    assert result["n_groups"] == 3
    assert result["p_value"] < 0.001


def test_sweep_channels_applies_fdr_correction():
    df = _synthetic_long_df()
    result = sweep_channels(df, test="anova")
    assert len(result) == 2  # order 1 and order 2
    assert "p_adj" in result.columns
    assert (result["p_adj"] >= result["p_value"]).all()
    assert result["reject_h0"].all()


def test_confidence_interval_mean_contains_true_mean():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=10.0, scale=1.0, size=1000)
    ci = confidence_interval_mean(values)
    assert ci["ci_low"] < 10.0 < ci["ci_high"]


def test_correlation_with_ci_detects_positive_correlation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    y = x * 2 + rng.normal(scale=0.1, size=500)
    result = correlation_with_ci(x, y)
    assert result["r"] > 0.95
    assert result["p_value"] < 1e-6
    assert result["ci_low"] < result["r"] < result["ci_high"]
