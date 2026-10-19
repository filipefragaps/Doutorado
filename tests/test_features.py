import numpy as np
import pandas as pd

from harmonic_signatures.features.selection import (
    build_candidate_pool,
    parse_feature_key,
    remove_redundant_features,
    select_features,
)


def test_parse_feature_key_extracts_parts():
    spec = parse_feature_key("H05_Ia_avg")
    assert spec == {
        "family": "H", "order": 5, "variable": "I", "phase": "a", "statistic": "avg", "source_sheet": None,
    }


def test_parse_feature_key_handles_scalar_family():
    spec = parse_feature_key("THD_Va_max")
    assert spec["family"] == "THD"
    assert spec["order"] is None


def test_parse_feature_key_handles_harm_inter_suffix():
    spec_harm = parse_feature_key("H00_Ia_max_harm")
    assert spec_harm["family"] == "H"
    assert spec_harm["order"] == 0
    assert spec_harm["source_sheet"] == "harmonicas"

    spec_inter = parse_feature_key("H00_Ia_max_inter")
    assert spec_inter["source_sheet"] == "interharmonicas"


def _synthetic_matrix():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 3, size=n)
    informative = y.astype(float) * 5 + rng.normal(scale=0.5, size=n)
    duplicate = informative + rng.normal(scale=0.01, size=n)  # near-perfect copy
    noise = rng.normal(size=n)
    return pd.DataFrame({
        "label": y, "informative": informative, "duplicate_of_informative": duplicate, "noise": noise,
    })


def test_remove_redundant_features_drops_near_duplicate():
    df = _synthetic_matrix()
    kept, dropped = remove_redundant_features(
        df, ["informative", "duplicate_of_informative", "noise"], corr_threshold=0.95
    )
    assert "informative" in kept
    assert "duplicate_of_informative" not in kept
    assert dropped["duplicate_of_informative"] == "informative"
    assert "noise" in kept


def test_select_features_ranks_informative_feature_first_and_drops_duplicate():
    df = _synthetic_matrix()
    result = select_features(
        df, ["noise", "duplicate_of_informative", "informative"], corr_threshold=0.95
    )
    assert result.ranking.iloc[0]["feature"] == "informative"
    assert "duplicate_of_informative" not in result.selected_features
    assert "informative" in result.selected_features


def test_build_candidate_pool_splits_harmonic_and_interharmonic_and_excludes_meta():
    df = pd.DataFrame({
        "sample_id": [1, 2], "label": ["A", "B"], "timestamp": [0, 1],
        "H05_Ia_avg": [1.0, 2.0], "H00_Ia_max_harm": [1.0, 2.0], "H00_Ia_max_inter": [3.0, 4.0],
        "IH20_Ic_avg": [1.0, 2.0], "THD_Va_max": [1.0, 2.0],
    })
    harm_cols, inter_cols = build_candidate_pool(df)

    assert set(harm_cols) == {"H05_Ia_avg", "H00_Ia_max_harm", "THD_Va_max"}
    assert set(inter_cols) == {"IH20_Ic_avg"}
    assert "H00_Ia_max_inter" not in harm_cols and "H00_Ia_max_inter" not in inter_cols
    assert not ({"sample_id", "label", "timestamp"} & (set(harm_cols) | set(inter_cols)))
