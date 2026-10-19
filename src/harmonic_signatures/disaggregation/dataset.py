from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pairs import DISAGGREGATION_PAIRS


@dataclass
class DisaggregationDataset:
    X: np.ndarray  # (n_samples, n_features) — combined-condition signatures
    y_machine: np.ndarray  # (n_samples, n_features) — pure machine-baseline target (repeated mean)
    y_source: np.ndarray  # (n_samples, n_features) — pure fault-source target (repeated mean)
    groups: np.ndarray  # (n_samples,) — which combined label each row came from (for leave-one-out CV)
    feature_cols: list[str]


def build_disaggregation_dataset(
    matrix_df: pd.DataFrame, feature_cols: list[str], *, pairs: dict[str, tuple[str, str]] = DISAGGREGATION_PAIRS,
) -> DisaggregationDataset:
    """Each combined-condition sample is paired with the *mean* signature of its
    two pure sources (machine-only run, fault-only run) as the regression
    target — the model learns to recover each source's typical signature from
    the combined one."""

    X_list, y_machine_list, y_source_list, groups_list = [], [], [], []

    for combined_label, (machine_label, source_label) in pairs.items():
        combined_rows = matrix_df.loc[matrix_df["label"] == combined_label, feature_cols]
        machine_mean = matrix_df.loc[matrix_df["label"] == machine_label, feature_cols].mean().to_numpy()
        source_mean = matrix_df.loc[matrix_df["label"] == source_label, feature_cols].mean().to_numpy()

        n = len(combined_rows)
        X_list.append(combined_rows.to_numpy())
        y_machine_list.append(np.tile(machine_mean, (n, 1)))
        y_source_list.append(np.tile(source_mean, (n, 1)))
        groups_list.extend([combined_label] * n)

    return DisaggregationDataset(
        X=np.concatenate(X_list),
        y_machine=np.concatenate(y_machine_list),
        y_source=np.concatenate(y_source_list),
        groups=np.array(groups_list),
        feature_cols=feature_cols,
    )
