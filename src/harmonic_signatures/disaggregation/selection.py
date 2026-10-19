from __future__ import annotations

import pandas as pd

from ..features.selection import FeatureSelectionResult, select_features
from .pairs import DISAGGREGATION_PAIRS


def select_features_for_disaggregation(
    matrix_df: pd.DataFrame,
    candidate_cols: list[str],
    *,
    combined_labels: list[str] | None = None,
    label_col: str = "label",
    top_k: int | None = 60,
    corr_threshold: float = 0.95,
) -> FeatureSelectionResult:
    """Attribute selection for RQ2, fit only on the pure-condition rows (the 13
    mechanical-only/electrical-only labels), with every combined-condition row
    excluded -- not just the one held out by a given `LeaveOneGroupOut` fold.

    The disaggregation dataset itself is evaluated by leave-one-combined-
    condition-out CV (`evaluate_disaggregation`), so selecting attributes with
    `select_features(matrix_df, ...)` on the *full* 19.501-row matrix (as the
    notebook did before this fix) lets every fold's held-out combined condition
    influence which 60 channels get chosen in the first place -- selection
    leakage, independent of the model-fitting leakage the leave-one-out CV
    already guards against.

    Excluding all four combined labels (not only the fold's held-out one) is a
    stricter fix than fold-specific exclusion, but it buys something the
    fold-specific version would not: the same 60 channels are selected for
    every fold, so `per_channel_r2`'s pooling of predictions across folds still
    refers to one stable, comparable set of channels rather than a different
    60 per fold.
    """

    combined = combined_labels if combined_labels is not None else list(DISAGGREGATION_PAIRS)
    pure_df = matrix_df.loc[~matrix_df[label_col].isin(combined)]
    return select_features(pure_df, candidate_cols, label_col=label_col, top_k=top_k, corr_threshold=corr_threshold)
