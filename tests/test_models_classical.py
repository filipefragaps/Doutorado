import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from harmonic_signatures.models.classical import compare_models, default_models


def test_default_models_returns_six_pipelines():
    models = default_models()
    assert set(models) == {"random_forest", "xgboost", "svm", "mlp", "knn", "logistic_regression"}


def test_compare_models_runs_and_ranks_by_f1(tmp_path):
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=6, n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    labels = np.array(["A", "B", "C"])[y]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=0, stratify=labels)

    summary, results = compare_models(X_train, y_train, X_test, y_test)

    assert len(summary) == 6
    assert (summary["f1_macro"] >= 0).all() and (summary["f1_macro"] <= 1).all()
    assert list(summary["f1_macro"]) == sorted(summary["f1_macro"], reverse=True)
    assert all(r.accuracy > 0.5 for r in results.values())
