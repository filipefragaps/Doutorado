from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_ranking(
    estimator, X: np.ndarray, y: np.ndarray, feature_names: list[str], *, n_repeats: int = 10,
    random_state: int = 42, scoring: str | None = None, n_jobs: int = -1,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=n_jobs,
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_importance_ranking(
    estimator, X: np.ndarray, feature_names: list[str], *, max_samples: int = 300, random_state: int = 42,
) -> pd.DataFrame:
    import shap

    rng = np.random.default_rng(random_state)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.Explainer(estimator, X_sample)
    try:
        shap_values = explainer(X_sample)
    except Exception:
        # Tree ensembles with many trees can fail SHAP's additivity check due to
        # float32 accumulation error — the ranking is still informative even
        # when the exact sum-to-model-output check doesn't hold to full precision.
        shap_values = explainer(X_sample, check_additivity=False)
    values = shap_values.values
    if values.ndim == 3:  # (n_samples, n_features, n_classes) for multiclass
        importance = np.abs(values).mean(axis=(0, 2))
    else:
        importance = np.abs(values).mean(axis=0)

    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)
