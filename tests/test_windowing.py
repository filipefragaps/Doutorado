import numpy as np
import pandas as pd

from harmonic_signatures.deep_learning.windowing import make_sequence_windows


def _synthetic_matrix():
    rows = []
    for label, n in [("A", 25), ("B", 12)]:
        for t in range(n):
            rows.append({"label": label, "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=t * 10),
                         "f1": float(t), "f2": float(t) * 2})
    return pd.DataFrame(rows)


def test_make_sequence_windows_shapes():
    df = _synthetic_matrix()
    wd = make_sequence_windows(df, ["f1", "f2"], window_size=5, stride=1, forecast_horizon=1)
    assert wd.X.shape[1:] == (5, 2)
    assert wd.X.shape[0] == wd.y_class.shape[0] == wd.y_forecast.shape[0]
    assert set(wd.y_class) == {"A", "B"}


def test_make_sequence_windows_forecast_target_correct():
    df = _synthetic_matrix()
    wd = make_sequence_windows(df, ["f1"], window_size=3, stride=1, forecast_horizon=1)
    # first window of group A: f1 values [0,1,2], forecast target should be f1=3
    first_a_idx = np.where(wd.groups == "A")[0][0]
    assert wd.X[first_a_idx, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert wd.y_forecast[first_a_idx, 0] == 3.0


def test_make_sequence_windows_drops_out_of_range_forecast_to_nan():
    df = _synthetic_matrix()
    wd = make_sequence_windows(df, ["f1"], window_size=5, stride=1, forecast_horizon=1)
    last_a_idx = np.where(wd.groups == "A")[0][-1]
    # last window of a 25-length run with window=5 ends exactly at the boundary -> no forecast target
    assert np.isnan(wd.y_forecast[last_a_idx, 0])


def test_stride_reduces_window_count():
    df = _synthetic_matrix()
    wd_stride1 = make_sequence_windows(df, ["f1"], window_size=5, stride=1)
    wd_stride5 = make_sequence_windows(df, ["f1"], window_size=5, stride=5)
    assert len(wd_stride5.X) < len(wd_stride1.X)
