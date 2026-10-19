from __future__ import annotations

import numpy as np
import pandas as pd


def predict_with_probability(pipeline, X: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """'Tipo de falha' + 'probabilidade da falha' from any sklearn-compatible
    classifier that exposes predict_proba (RF/XGBoost/MLP/kNN/LogReg do by
    default; SVC needs `probability=True`, which slows down fitting)."""

    if not hasattr(pipeline, "predict_proba"):
        raise AttributeError("Este modelo não expõe predict_proba (ex.: SVC precisa de probability=True).")

    proba = pipeline.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in class_names])
    out.insert(0, "probability", proba.max(axis=1))
    out.insert(0, "predicted_label", [class_names[i] for i in pred_idx])
    return out


def predict_with_probability_keras(model, X: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Same as `predict_with_probability` but for a Keras classifier whose
    output layer is already a softmax (every architecture in module 11)."""

    proba = model.predict(X, verbose=0)
    pred_idx = proba.argmax(axis=1)
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in class_names])
    out.insert(0, "probability", proba.max(axis=1))
    out.insert(0, "predicted_label", [class_names[i] for i in pred_idx])
    return out
