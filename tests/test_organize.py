import numpy as np
import pandas as pd

from harmonic_signatures.organize.join import nearest_timestamp_join
from harmonic_signatures.organize.long_format import to_long
from harmonic_signatures.organize.matrix_format import to_matrix
from harmonic_signatures.organize.tensor_format import build_sequence_tensor


def test_long_format_row_count(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    n_measurement_cols = len(small_harmonicas_sheet.measurement_columns())
    assert len(long_df) == len(small_harmonicas_sheet.dataframe) * n_measurement_cols


def test_long_format_preserves_values(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    h01_avg = long_df[(long_df["family"] == "H") & (long_df["order"] == 1) & (long_df["statistic"] == "avg")]
    assert sorted(h01_avg["value"].tolist()) == [10.0, 20.0, 30.0]


def test_long_format_normalizes_label(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    assert set(long_df["label"].unique()) == {"DIODO 1", "DIODO 2"}


def test_long_format_sample_ids_unique_per_row(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    n_rows = len(small_harmonicas_sheet.dataframe)
    assert long_df["sample_id"].nunique() == n_rows


def test_matrix_format_shape(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    matrix = to_matrix(long_df)
    assert len(matrix) == len(small_harmonicas_sheet.dataframe)
    assert "label" in matrix.columns and "timestamp" in matrix.columns
    assert "H01_Va_avg" in matrix.columns
    assert matrix.loc[matrix["timestamp"] == pd.Timestamp("2025-01-01 10:00:00"), "H01_Va_avg"].iloc[0] == 10.0


def test_matrix_round_trips_all_measurement_columns(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    matrix = to_matrix(long_df)
    n_measurement_cols = len(small_harmonicas_sheet.measurement_columns())
    n_feature_cols = matrix.shape[1] - 3  # minus sample_id, label, timestamp
    assert n_feature_cols == n_measurement_cols


def test_nearest_timestamp_join_matches_within_tolerance(small_harmonicas_sheet, small_interharmonicas_sheet):
    harm_matrix = to_matrix(to_long(small_harmonicas_sheet))
    inter_matrix = to_matrix(to_long(small_interharmonicas_sheet))

    joined = nearest_timestamp_join(harm_matrix, inter_matrix, tolerance_seconds=30)

    assert joined["matched"].all()
    assert np.allclose(joined["join_delta_seconds"], 20.0)


def test_nearest_timestamp_join_respects_tolerance(small_harmonicas_sheet, small_interharmonicas_sheet):
    harm_matrix = to_matrix(to_long(small_harmonicas_sheet))
    inter_matrix = to_matrix(to_long(small_interharmonicas_sheet))

    joined = nearest_timestamp_join(harm_matrix, inter_matrix, tolerance_seconds=5)

    assert not joined["matched"].any()


def test_nearest_timestamp_join_suffixes_overlapping_columns(small_harmonicas_sheet, small_interharmonicas_sheet):
    harm_matrix = to_matrix(to_long(small_harmonicas_sheet))
    inter_matrix = to_matrix(to_long(small_interharmonicas_sheet))

    joined = nearest_timestamp_join(harm_matrix, inter_matrix, tolerance_seconds=30)

    assert "H01_Va_max_harm" in joined.columns
    assert "H01_Va_max_inter" in joined.columns
    assert joined["H01_Va_max_harm"].tolist() == [11.0, 21.0, 31.0]
    assert joined["H01_Va_max_inter"].tolist() == [99.0, 98.0, 97.0]


def test_build_sequence_tensor_shape(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    bundle = build_sequence_tensor(long_df)

    n_samples = len(small_harmonicas_sheet.dataframe)
    assert bundle.tensor.shape[0] == n_samples
    assert set(bundle.order_values) == {0, 1, 2}
    assert set(bundle.channel_names) == {"H_Va_avg", "H_Va_max"}
    assert list(bundle.y) == ["DIODO 1", "DIODO 1", "DIODO 2"]


def test_build_sequence_tensor_values_correct(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    bundle = build_sequence_tensor(long_df)

    order_idx = bundle.order_values.index(1)
    channel_idx = bundle.channel_names.index("H_Va_avg")
    assert bundle.tensor[:, order_idx, channel_idx].tolist() == [10.0, 20.0, 30.0]


def test_build_sequence_tensor_static_features(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    bundle = build_sequence_tensor(long_df)

    assert "THD_Va_avg" in bundle.static.columns
    assert "DC_Va_avg" in bundle.static.columns
    assert "K-Factor_Ia_avg" in bundle.static.columns
    assert bundle.static["THD_Va_avg"].tolist() == [1.0, 2.0, 3.0]
