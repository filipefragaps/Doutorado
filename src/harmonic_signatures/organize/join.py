from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_JOIN_TOLERANCE_SECONDS


def nearest_timestamp_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance_seconds: float = DEFAULT_JOIN_TOLERANCE_SECONDS,
    left_suffix: str = "_harm",
    right_suffix: str = "_inter",
) -> pd.DataFrame:
    """Match rows of two sample-level frames (as produced by matrix_format.to_matrix)
    by nearest `timestamp` within the same `label`, since the two source sheets are
    independent time series and are not row-aligned."""

    left_sorted = left.sort_values("timestamp").reset_index(drop=True)
    right_sorted = (
        right.sort_values("timestamp")
        .reset_index(drop=True)
        .assign(_right_timestamp=lambda d: d["timestamp"])
    )

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="timestamp",
        by="label",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=(left_suffix, right_suffix),
    )

    right_sample_col = "sample_id" + right_suffix
    merged = merged.rename(columns={"_right_timestamp": f"timestamp{right_suffix}"})
    merged["join_delta_seconds"] = (
        (merged["timestamp"] - merged[f"timestamp{right_suffix}"]).abs().dt.total_seconds()
    )
    merged["matched"] = merged[right_sample_col].notna()
    return merged
