from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ..features.selection import select_features
from ..validation.splitting import blocked_temporal_kfold, temporal_train_test_split
from .classical import compare_models


def evaluate_holdout(
    matrix_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    test_size: float = 0.3,
    top_k: int | None = 40,
    corr_threshold: float = 0.95,
    models: dict[str, Pipeline] | None = None,
):
    """Single temporal train/test split with attribute selection fit on the
    training rows only, replacing the leaky protocol (`select_features` on the
    full matrix, *then* `train_test_split`) that fed the 92% headline number in
    the thesis draft. Returns the same `(summary, results)` shape as
    `compare_models`, plus the `FeatureSelectionResult` so callers can report
    which features were actually selected on this split.
    """

    train_idx, test_idx = temporal_train_test_split(
        matrix_df, label_col=label_col, timestamp_col=timestamp_col, test_size=test_size,
    )
    train_df, test_df = matrix_df.iloc[train_idx], matrix_df.iloc[test_idx]

    sel = select_features(
        train_df, feature_cols, label_col=label_col, top_k=top_k, corr_threshold=corr_threshold,
    )
    X_train = train_df[sel.selected_features].fillna(0).to_numpy()
    X_test = test_df[sel.selected_features].fillna(0).to_numpy()
    y_train = train_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()

    summary, results = compare_models(X_train, y_train, X_test, y_test, models=models)
    return summary, results, sel


def evaluate_cv(
    matrix_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    n_splits: int = 5,
    top_k: int | None = 40,
    corr_threshold: float = 0.95,
    models: dict[str, Pipeline] | None = None,
) -> pd.DataFrame:
    """Blocked temporal K-fold with attribute selection re-fit on the training
    portion of *each* fold independently -- the leak-free, temporally-honest
    replacement for the notebook's `StratifiedKFold(5)` over a single global
    selection. Returns one row per (fold, model) with accuracy/f1_macro; the
    caller aggregates with `.groupby('model')[['accuracy','f1_macro']].agg(['mean','std'])`
    for the Tabela res_cv-style summary.
    """

    folds = blocked_temporal_kfold(
        matrix_df, label_col=label_col, timestamp_col=timestamp_col, n_splits=n_splits,
    )

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_df, test_df = matrix_df.iloc[train_idx], matrix_df.iloc[test_idx]

        sel = select_features(
            train_df, feature_cols, label_col=label_col, top_k=top_k, corr_threshold=corr_threshold,
        )
        X_train = train_df[sel.selected_features].fillna(0).to_numpy()
        X_test = test_df[sel.selected_features].fillna(0).to_numpy()
        y_train = train_df[label_col].to_numpy()
        y_test = test_df[label_col].to_numpy()

        summary, _ = compare_models(X_train, y_train, X_test, y_test, models=models)
        summary["fold"] = fold_idx
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)
