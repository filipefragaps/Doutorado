from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .training import train_classifier


def random_search(
    model_builder,
    param_grid: dict[str, list],
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, n_trials: int = 8, epochs: int = 10, random_state: int = 42,
) -> tuple[pd.DataFrame, dict, "keras.Model"]:
    """Lightweight random hyperparameter search — samples `n_trials` combinations
    from `param_grid` independently per hyperparameter (no external tuning
    dependency), trains each briefly, and returns every trial's validation score
    plus the best model. 'Ajuste automático de hiperparâmetros' without pulling
    in a whole tuning framework for what is, here, a handful of knobs."""

    rng = random.Random(random_state)
    encoder = LabelEncoder().fit(np.concatenate([y_train, y_val]))

    rows = []
    best_result = None
    best_params = None

    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in param_grid.items()}
        result = train_classifier(
            model_builder, params, X_train, y_train, X_val, y_val,
            epochs=epochs, label_encoder=encoder, verbose=0,
        )
        rows.append({"trial": trial, **params, "val_accuracy": result.val_accuracy, "val_loss": result.val_loss})

        if best_result is None or result.val_accuracy > best_result.val_accuracy:
            best_result = result
            best_params = params

    trials_df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    return trials_df, best_params, best_result.model
