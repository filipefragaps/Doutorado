from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedDataset:
    X: np.ndarray  # (n_windows, window_size, n_features)
    y_class: np.ndarray  # (n_windows,) — condition label (constant within a run in this dataset)
    y_forecast: np.ndarray  # (n_windows, n_features) — features `forecast_horizon` steps after the window, NaN-free rows only
    groups: np.ndarray  # (n_windows,) — which run/label each window belongs to
    window_size: int
    stride: int
    forecast_horizon: int
    feature_cols: list[str]


def make_sequence_windows(
    matrix_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    group_col: str = "label",
    time_col: str = "timestamp",
    window_size: int = 10,
    stride: int = 1,
    forecast_horizon: int = 1,
) -> WindowedDataset:
    """Slide a fixed-size window of `window_size` consecutive acquisitions over
    each run (group_col), sorted by time_col. Each window carries the run's
    class label (for sequence classification) and, when available, the feature
    vector `forecast_horizon` steps ahead (for 'assinatura futura' forecasting).

    Note: in this dataset each label is a single ~4-6h stationary run (confirmed
    empirically — no run-to-failure trajectory), so y_class is constant within a
    run and windowing here provides *temporal context* for classification/
    forecasting, not a remaining-useful-life signal."""

    X_list, y_class_list, y_forecast_list, groups_list = [], [], [], []

    for group, grp in matrix_df.groupby(group_col, observed=True):
        grp = grp.sort_values(time_col)
        values = grp[feature_cols].to_numpy(dtype="float64")
        n = len(values)

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            forecast_idx = end + forecast_horizon - 1
            X_list.append(values[start:end])
            y_class_list.append(group)
            y_forecast_list.append(values[forecast_idx] if forecast_idx < n else np.full(len(feature_cols), np.nan))
            groups_list.append(group)

    return WindowedDataset(
        X=np.stack(X_list),
        y_class=np.array(y_class_list),
        y_forecast=np.stack(y_forecast_list),
        groups=np.array(groups_list),
        window_size=window_size,
        stride=stride,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols,
    )
