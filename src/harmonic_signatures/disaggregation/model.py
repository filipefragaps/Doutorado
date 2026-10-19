from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor

from .dataset import DisaggregationDataset


def evaluate_disaggregation(
    dataset: DisaggregationDataset, *, model_factory=lambda: RandomForestRegressor(n_estimators=200, random_state=42),
) -> pd.DataFrame:
    """Leave-one-combined-condition-out cross-validation: train on 3 of the 4
    known combined conditions, evaluate whether the model recovers the machine
    and source signatures on the held-out (unseen-combination) 4th. With only 4
    combined conditions in this dataset, treat these numbers as a proof of
    concept, not a statistically powered generalization claim.

    IMPORTANT — the R2 column here is structurally degenerate for BOTH targets,
    not just 'machine'. Each fold holds out exactly one combined condition, so
    within that fold the target vector (machine OR source) is constant across
    every row — sklearn's r2_score falls back to its zero-variance special case
    (0.0 or 1.0, not a real score) for every single-fold, single-output R2, and
    'uniform_average' across outputs then collapses to exactly 0.0 or 1.0. R2=1.0
    for 'machine' and R2=0.0 for 'source' are BOTH artifacts of this, not
    evidence of anything. MAE is still meaningful (it doesn't have this
    degeneracy). For an actual generalization signal, use `per_channel_r2`,
    which pools predictions across all 4 held-out folds *before* scoring — the
    pooled target has real variance (different conditions have different source
    means), so its R2 is not degenerate."""

    logo = LeaveOneGroupOut()
    rows = []

    for train_idx, test_idx in logo.split(dataset.X, groups=dataset.groups):
        held_out_label = dataset.groups[test_idx][0]

        # n_jobs on the outer MultiOutputRegressor parallelizes across output
        # columns (60 here) — that's the real bottleneck, not the inner trees,
        # so the inner RandomForestRegressor stays single-threaded to avoid
        # nested-parallelism contention.
        machine_model = MultiOutputRegressor(model_factory(), n_jobs=-1)
        source_model = MultiOutputRegressor(model_factory(), n_jobs=-1)
        machine_model.fit(dataset.X[train_idx], dataset.y_machine[train_idx])
        source_model.fit(dataset.X[train_idx], dataset.y_source[train_idx])

        machine_pred = machine_model.predict(dataset.X[test_idx])
        source_pred = source_model.predict(dataset.X[test_idx])

        rows.append({
            "held_out_condition": held_out_label,
            "target": "machine",
            "mae": mean_absolute_error(dataset.y_machine[test_idx], machine_pred),
            "r2": r2_score(dataset.y_machine[test_idx], machine_pred),
        })
        rows.append({
            "held_out_condition": held_out_label,
            "target": "source",
            "mae": mean_absolute_error(dataset.y_source[test_idx], source_pred),
            "r2": r2_score(dataset.y_source[test_idx], source_pred),
        })

    return pd.DataFrame(rows)


def per_channel_r2(
    dataset: DisaggregationDataset,
    *,
    target: str = "source",
    model_factory=lambda: RandomForestRegressor(n_estimators=200, random_state=42),
) -> pd.DataFrame:
    """Per-feature R2/MAE from pooled leave-one-condition-out out-of-fold
    predictions — answers 'which harmonic/inter-harmonic channels does the
    disaggregation actually recover well', not just one aggregate number across
    all channels at once. Use target='source' for the meaningful generalization
    test (see the constant-target caveat on `evaluate_disaggregation`)."""

    y = dataset.y_machine if target == "machine" else dataset.y_source
    logo = LeaveOneGroupOut()
    all_true = np.zeros_like(y)
    all_pred = np.zeros_like(y)

    for train_idx, test_idx in logo.split(dataset.X, groups=dataset.groups):
        model = MultiOutputRegressor(model_factory(), n_jobs=-1)
        model.fit(dataset.X[train_idx], y[train_idx])
        all_pred[test_idx] = model.predict(dataset.X[test_idx])
        all_true[test_idx] = y[test_idx]

    rows = [
        {
            "feature": feat,
            "r2": r2_score(all_true[:, i], all_pred[:, i]),
            "mae": mean_absolute_error(all_true[:, i], all_pred[:, i]),
        }
        for i, feat in enumerate(dataset.feature_cols)
    ]
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def fit_disaggregation_models(
    dataset: DisaggregationDataset, *, model_factory=lambda: RandomForestRegressor(n_estimators=200, random_state=42),
) -> tuple[MultiOutputRegressor, MultiOutputRegressor]:
    """Fit on every combined sample available — the models to actually use once
    the leave-one-out evaluation above has been reviewed."""

    machine_model = MultiOutputRegressor(model_factory(), n_jobs=-1).fit(dataset.X, dataset.y_machine)
    source_model = MultiOutputRegressor(model_factory(), n_jobs=-1).fit(dataset.X, dataset.y_source)
    return machine_model, source_model
