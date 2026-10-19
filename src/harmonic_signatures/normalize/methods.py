from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ArrayNormalizer, LongFormatNormalizer

_GROUP_COLS = ["sample_id", "family", "variable", "phase", "statistic"]


class MinMaxNormalizer(ArrayNormalizer):
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxNormalizer":
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.feature_range
        span = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return lo + (X - self.min_) * (hi - lo) / span

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.feature_range
        span = self.max_ - self.min_
        return self.min_ + (X - lo) * span / (hi - lo)


class ZScoreNormalizer(ArrayNormalizer):
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreNormalizer":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1.0, self.std_)
        return (X - self.mean_) / std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_


def _group_key(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[_GROUP_COLS])


class FundamentalNormalizer(LongFormatNormalizer):
    """Divide each H/IH order's value by the fundamental (order == 1) value of
    the same sample/family/variable/phase/statistic group."""

    def __init__(self, families: tuple[str, ...] = ("H", "IH")):
        self.families = set(families)

    def transform(self, long_df: pd.DataFrame) -> pd.DataFrame:
        mask = long_df["family"].isin(self.families)

        fundamentals = (
            long_df[mask & (long_df["order"] == 1)]
            .drop_duplicates(subset=_GROUP_COLS)
            .set_index(_GROUP_COLS)["value"]
        )

        out = long_df.copy()
        denom = _group_key(out).map(fundamentals).to_numpy(dtype="float64").copy()
        denom[denom == 0] = np.nan

        result = out["value"].to_numpy(dtype="float64").copy()
        result[mask.to_numpy()] = result[mask.to_numpy()] / denom[mask.to_numpy()]
        out["value"] = result
        return out


class SpectralEnergyNormalizer(LongFormatNormalizer):
    """Divide each H/IH order's value by sqrt(sum of squares) of all orders in
    the same sample/family/variable/phase/statistic group (normalization by
    spectral energy)."""

    def __init__(self, families: tuple[str, ...] = ("H", "IH")):
        self.families = set(families)

    def transform(self, long_df: pd.DataFrame) -> pd.DataFrame:
        mask = long_df["family"].isin(self.families)

        energy = long_df[mask].groupby(_GROUP_COLS, observed=True)["value"].apply(
            lambda v: np.sqrt(np.nansum(np.square(v)))
        )

        out = long_df.copy()
        denom = _group_key(out).map(energy).to_numpy(dtype="float64").copy()
        denom[denom == 0] = np.nan

        result = out["value"].to_numpy(dtype="float64").copy()
        result[mask.to_numpy()] = result[mask.to_numpy()] / denom[mask.to_numpy()]
        out["value"] = result
        return out
