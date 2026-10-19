from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ArrayNormalizer(ABC):
    """Column-wise normalizer over a 2D feature matrix (n_samples, n_features)."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "ArrayNormalizer": ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...


class LongFormatNormalizer(ABC):
    """Normalizer that needs the structured long-format columns (sample_id,
    family, variable, phase, order, statistic, value) because each value is
    normalized relative to another row of the same sample/channel group
    (e.g. dividing by the fundamental or by the group's spectral energy)."""

    @abstractmethod
    def transform(self, long_df: pd.DataFrame) -> pd.DataFrame: ...
