import numpy as np
import pandas as pd
import pytest

from harmonic_signatures.normalize.methods import (
    FundamentalNormalizer,
    MinMaxNormalizer,
    SpectralEnergyNormalizer,
    ZScoreNormalizer,
)


@pytest.fixture
def small_order_long_df():
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "family": ["H", "H", "H", "H", "H", "H"],
            "variable": ["V"] * 6,
            "phase": ["a"] * 6,
            "statistic": ["avg"] * 6,
            "order": [1, 2, 3, 1, 2, 3],
            "value": [10.0, 5.0, 2.5, 4.0, 2.0, 1.0],
        }
    )


def test_minmax_normalizer_scales_to_range():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    normalizer = MinMaxNormalizer()
    Xt = normalizer.fit_transform(X)

    assert np.allclose(Xt.min(axis=0), [0.0, 0.0])
    assert np.allclose(Xt.max(axis=0), [1.0, 1.0])
    assert np.allclose(normalizer.inverse_transform(Xt), X)


def test_minmax_normalizer_handles_constant_column():
    X = np.array([[5.0], [5.0], [5.0]])
    Xt = MinMaxNormalizer().fit_transform(X)
    assert np.allclose(Xt, 0.0)


def test_zscore_normalizer_mean_zero_std_one():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    normalizer = ZScoreNormalizer()
    Xt = normalizer.fit_transform(X)

    assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Xt.std(axis=0), 1.0)
    assert np.allclose(normalizer.inverse_transform(Xt), X)


def test_fundamental_normalizer_divides_by_order_one(small_order_long_df):
    out = FundamentalNormalizer().transform(small_order_long_df)

    s1 = out[out["sample_id"] == "s1"].sort_values("order")["value"].tolist()
    s2 = out[out["sample_id"] == "s2"].sort_values("order")["value"].tolist()
    assert s1 == pytest.approx([1.0, 0.5, 0.25])
    assert s2 == pytest.approx([1.0, 0.5, 0.25])


def test_fundamental_normalizer_leaves_other_families_untouched():
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "family": ["THD"],
            "variable": ["V"],
            "phase": ["a"],
            "statistic": ["avg"],
            "order": [pd.NA],
            "value": [7.5],
        }
    )
    out = FundamentalNormalizer().transform(df)
    assert out["value"].iloc[0] == 7.5


def test_spectral_energy_normalizer_unit_norm_per_group(small_order_long_df):
    out = SpectralEnergyNormalizer().transform(small_order_long_df)

    for sample_id in ("s1", "s2"):
        values = out[out["sample_id"] == sample_id]["value"].to_numpy()
        assert np.isclose(np.sqrt(np.sum(values**2)), 1.0)
