from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def default_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Six classical models spanning trees, margin-based, distance-based, neural
    and linear methods. Scale-sensitive models (SVM/MLP/kNN/LogReg) get a
    StandardScaler in-pipeline; tree ensembles don't need one."""

    return {
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1,
                eval_metric="mlogloss", tree_method="hist",
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=random_state)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=random_state)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
    }


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    f1_macro: float
    fit_seconds: float
    predict_seconds: float


def compare_models(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    *, models: dict[str, Pipeline] | None = None,
) -> tuple[pd.DataFrame, dict[str, ModelResult]]:
    """Fit every model on the same split and report accuracy/F1/timing side by
    side — 'comparar automaticamente todos os modelos'."""

    models = models or default_models()
    encoder = LabelEncoder().fit(y_train)
    y_train_enc = encoder.transform(y_train)
    y_test_enc = encoder.transform(y_test)

    results: dict[str, ModelResult] = {}
    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train_enc)
        fit_seconds = time.time() - t0

        t0 = time.time()
        y_pred = pipeline.predict(X_test)
        predict_seconds = time.time() - t0

        results[name] = ModelResult(
            name=name, pipeline=pipeline,
            accuracy=accuracy_score(y_test_enc, y_pred),
            f1_macro=f1_score(y_test_enc, y_pred, average="macro"),
            fit_seconds=fit_seconds, predict_seconds=predict_seconds,
        )

    summary = pd.DataFrame([
        {"model": r.name, "accuracy": r.accuracy, "f1_macro": r.f1_macro,
         "fit_seconds": r.fit_seconds, "predict_seconds": r.predict_seconds}
        for r in results.values()
    ]).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    return summary, results
