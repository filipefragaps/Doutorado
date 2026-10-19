import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from harmonic_signatures.disaggregation.dataset import build_disaggregation_dataset
from harmonic_signatures.disaggregation.model import evaluate_disaggregation, fit_disaggregation_models, per_channel_r2
from harmonic_signatures.disaggregation.selection import select_features_for_disaggregation


def _synthetic_matrix():
    rng = np.random.default_rng(0)
    rows = []

    def add(label, base, n=60):
        for _ in range(n):
            rows.append({"label": label, "f1": base[0] + rng.normal(scale=0.1),
                         "f2": base[1] + rng.normal(scale=0.1)})

    add("MAQUINA", [1.0, 2.0])
    add("DIODO_X", [5.0, 6.0])
    add("DIODO_Y", [9.0, 1.0])
    # combined ~ additive-ish mix of machine + source, plus its own noise
    for _ in range(60):
        rows.append({"label": "MAQUINA+DIODO_X", "f1": 1.0 + 5.0 + rng.normal(scale=0.1),
                     "f2": 2.0 + 6.0 + rng.normal(scale=0.1)})
    for _ in range(60):
        rows.append({"label": "MAQUINA+DIODO_Y", "f1": 1.0 + 9.0 + rng.normal(scale=0.1),
                     "f2": 2.0 + 1.0 + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


_PAIRS = {
    "MAQUINA+DIODO_X": ("MAQUINA", "DIODO_X"),
    "MAQUINA+DIODO_Y": ("MAQUINA", "DIODO_Y"),
}


def test_build_disaggregation_dataset_shapes_and_targets():
    df = _synthetic_matrix()
    ds = build_disaggregation_dataset(df, ["f1", "f2"], pairs=_PAIRS)
    assert ds.X.shape == (120, 2)
    assert ds.y_machine.shape == ds.y_source.shape == (120, 2)
    assert set(ds.groups) == {"MAQUINA+DIODO_X", "MAQUINA+DIODO_Y"}
    # machine target should be close to the true machine baseline mean [1, 2]
    assert np.allclose(ds.y_machine.mean(axis=0), [1.0, 2.0], atol=0.2)


def test_evaluate_disaggregation_returns_one_row_per_fold_and_target():
    # With only 2 combined conditions, each leave-one-out fold trains on a
    # single remaining group — too little to expect good generalization (this
    # mirrors the real dataset's small-n caveat, documented in model.py). Here
    # we only verify the evaluation harness runs and shapes out correctly, not
    # that it hits a quality bar.
    df = _synthetic_matrix()
    ds = build_disaggregation_dataset(df, ["f1", "f2"], pairs=_PAIRS)
    report = evaluate_disaggregation(ds, model_factory=lambda: RandomForestRegressor(n_estimators=50, random_state=0))
    assert len(report) == 4  # 2 held-out folds x 2 targets
    assert set(report["target"]) == {"machine", "source"}
    assert report["mae"].notna().all()
    # the machine target is (near-)constant across groups here, so its MAE
    # should be tiny regardless of which condition is held out — R2 itself is
    # not a reliable check for a ~zero-variance target (see model.py caveat)
    assert (report.loc[report["target"] == "machine", "mae"] < 0.5).all()


def test_per_channel_r2_returns_one_row_per_feature():
    df = _synthetic_matrix()
    ds = build_disaggregation_dataset(df, ["f1", "f2"], pairs=_PAIRS)
    report = per_channel_r2(ds, target="source", model_factory=lambda: RandomForestRegressor(n_estimators=50, random_state=0))
    assert set(report["feature"]) == {"f1", "f2"}
    assert report["r2"].notna().all()


def test_fit_disaggregation_models_predicts_close_to_targets():
    df = _synthetic_matrix()
    ds = build_disaggregation_dataset(df, ["f1", "f2"], pairs=_PAIRS)
    machine_model, source_model = fit_disaggregation_models(
        ds, model_factory=lambda: RandomForestRegressor(n_estimators=50, random_state=0)
    )
    machine_pred = machine_model.predict(ds.X)
    assert np.allclose(machine_pred.mean(axis=0), [1.0, 2.0], atol=0.5)


def test_select_features_for_disaggregation_excludes_combined_rows_from_selection():
    df = _synthetic_matrix()
    # f2 only separates the pure labels (MAQUINA/DIODO_X/DIODO_Y) well; f3 is
    # pure noise for the pure labels but happens to separate the combined rows
    # from everything else — a selector that leaks combined-row information
    # would rank f3 highly, one that doesn't should not.
    rng = np.random.default_rng(1)
    df = df.copy()
    is_combined = df["label"].str.contains(r"\+")
    df["f3"] = rng.normal(size=len(df))
    df.loc[is_combined, "f3"] += 10.0

    sel = select_features_for_disaggregation(
        df, ["f1", "f2", "f3"], combined_labels=["MAQUINA+DIODO_X", "MAQUINA+DIODO_Y"], top_k=2,
    )

    assert "f3" not in sel.selected_features
    assert set(sel.selected_features) == {"f1", "f2"}
