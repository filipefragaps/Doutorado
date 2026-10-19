import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from harmonic_signatures.sensitivity.effect_size import (
    cliffs_delta,
    cohens_d,
    eta_squared,
    interpret_cliffs_delta,
    interpret_eta_squared,
)
from harmonic_signatures.sensitivity.importance import permutation_importance_ranking, shap_importance_ranking
from harmonic_signatures.sensitivity.ranking import rank_channels_by_effect_size, rank_channels_by_eta_squared


def test_cohens_d_large_for_separated_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    b = rng.normal(5, 1, 300)
    assert cohens_d(a, b) < -4.0


def test_cohens_d_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert abs(cohens_d(a, b)) < 0.1


def test_cliffs_delta_extreme_for_fully_separated_groups():
    a = np.arange(100, 200)
    b = np.arange(0, 100)
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0
    assert interpret_cliffs_delta(cliffs_delta(a, b)) == "large"


def test_cliffs_delta_zero_for_identical_groups():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == 0.0


def test_rank_channels_by_effect_size_orders_by_magnitude():
    rows = []
    for label, means in [("A", {1: 0.0, 2: 0.0}), ("B", {1: 10.0, 2: 0.01})]:
        for order, mean in means.items():
            for i in range(50):
                rows.append({
                    "sample_id": f"{label}_{order}_{i}", "label": label, "domain": "harmonica",
                    "family": "H", "variable": "I", "phase": "a", "order": order,
                    "statistic": "avg", "value": mean + np.random.default_rng(i).normal(scale=0.05),
                })
    df = pd.DataFrame(rows)
    ranking = rank_channels_by_effect_size(df, "A", "B", method="cohens_d")
    assert ranking.iloc[0]["order"] == 1  # order 1 separates much more than order 2


def test_eta_squared_nan_when_all_values_identical():
    # zero total variance makes the ratio undefined (0/0), not zero
    groups = [np.full(50, 3.0), np.full(50, 3.0), np.full(50, 3.0)]
    assert np.isnan(eta_squared(groups))


def test_eta_squared_near_zero_when_group_means_equal_but_noisy():
    rng = np.random.default_rng(0)
    groups = [rng.normal(0, 1, 500), rng.normal(0, 1, 500), rng.normal(0, 1, 500)]
    value = eta_squared(groups)
    assert value < 0.01
    assert interpret_eta_squared(value) == "negligible"


def test_eta_squared_near_one_when_group_means_dominate_variance():
    rng = np.random.default_rng(0)
    groups = [rng.normal(0, 0.01, 200), rng.normal(10, 0.01, 200), rng.normal(20, 0.01, 200)]
    value = eta_squared(groups)
    assert value > 0.99
    assert interpret_eta_squared(value) == "large"


def test_rank_channels_by_eta_squared_ranks_more_separated_channel_first():
    rows = []
    for label, means in [("A", {1: 0.0, 2: 0.0}), ("B", {1: 5.0, 2: 0.01}), ("C", {1: 10.0, 2: 0.02})]:
        for order, mean in means.items():
            for i in range(50):
                rows.append({
                    "sample_id": f"{label}_{order}_{i}", "label": label, "domain": "harmonica",
                    "family": "H", "variable": "I", "phase": "a", "order": order,
                    "statistic": "avg", "value": mean + np.random.default_rng(i).normal(scale=0.05),
                })
    df = pd.DataFrame(rows)
    ranking = rank_channels_by_eta_squared(df)
    assert ranking.iloc[0]["order"] == 1


def test_permutation_and_shap_importance_agree_on_informative_feature():
    rng = np.random.default_rng(0)
    n = 300
    informative = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (informative > 0).astype(int)
    X = np.column_stack([informative, noise])
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)

    perm = permutation_importance_ranking(clf, X, y, ["informative", "noise"], n_repeats=5)
    assert perm.iloc[0]["feature"] == "informative"

    shap_rank = shap_importance_ranking(clf, X, ["informative", "noise"], max_samples=100)
    assert shap_rank.iloc[0]["feature"] == "informative"
