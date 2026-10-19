import numpy as np
import pandas as pd

from harmonic_signatures.models.classical import default_models
from harmonic_signatures.models.evaluation import evaluate_cv, evaluate_holdout


def _synthetic_matrix(n_per_label: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["A", "B", "C"]
    rows = []
    for li, label in enumerate(labels):
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=li)
        for i in range(n_per_label):
            informative = li * 5.0 + rng.normal(scale=0.5)
            rows.append({
                "label": label,
                "timestamp": base + pd.Timedelta(seconds=10 * i),
                "informative": informative,
                "duplicate": informative + rng.normal(scale=0.01),
                "noise": rng.normal(),
            })
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _fast_models():
    models = default_models()
    return {name: models[name] for name in ("logistic_regression", "knn")}


def test_evaluate_holdout_selects_only_from_training_rows():
    df = _synthetic_matrix()
    summary, results, sel = evaluate_holdout(
        df, ["informative", "duplicate", "noise"], top_k=2, models=_fast_models(),
    )

    # "informative" and "duplicate" are near-identical; redundancy elimination
    # keeps exactly one of them (whichever ranks first), never both.
    assert len({"informative", "duplicate"} & set(sel.selected_features)) == 1
    assert len(summary) == 2
    assert (summary["accuracy"] > 0.7).all()


def test_evaluate_cv_returns_one_row_per_fold_per_model():
    df = _synthetic_matrix()
    result = evaluate_cv(df, ["informative", "duplicate", "noise"], n_splits=5, top_k=2, models=_fast_models())

    assert set(result["fold"]) == {0, 1, 2, 3, 4}
    assert len(result) == 5 * 2
    agg = result.groupby("model")[["accuracy", "f1_macro"]].agg(["mean", "std"])
    assert (agg[("accuracy", "mean")] > 0.5).all()
