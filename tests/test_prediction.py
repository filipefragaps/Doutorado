import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from harmonic_signatures.prediction.fault_classifier import predict_with_probability
from harmonic_signatures.prediction.forecasting import FORECAST_ARCHITECTURES, train_forecaster


def _synthetic_classification_data():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (50, 2)), rng.normal(5, 1, (50, 2))])
    y = np.array(["A"] * 50 + ["B"] * 50)
    return X, y


def test_predict_with_probability_shape_and_sums_to_one():
    X, y = _synthetic_classification_data()
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    out = predict_with_probability(clf, X[:10], list(clf.classes_))
    assert set(out.columns) >= {"predicted_label", "probability", "p_A", "p_B"}
    assert np.allclose(out["p_A"] + out["p_B"], 1.0)
    assert (out["probability"] >= 0.5).all()


def test_predict_with_probability_raises_without_predict_proba():
    X, y = _synthetic_classification_data()
    clf = SVC(probability=False).fit(X, y)
    with pytest.raises(AttributeError):
        predict_with_probability(clf, X[:5], list(clf.classes_))


@pytest.fixture
def tiny_forecast_data():
    rng = np.random.default_rng(0)
    n, window, n_features = 80, 4, 3
    X = rng.normal(size=(n, window, n_features)).astype("float32")
    y = rng.normal(size=(n, n_features)).astype("float32")
    return X[:60], y[:60], X[60:], y[60:]


@pytest.mark.parametrize("name", list(FORECAST_ARCHITECTURES))
def test_forecast_architecture_trains_and_predicts_right_shape(name, tiny_forecast_data):
    X_train, y_train, X_val, y_val = tiny_forecast_data
    builder = FORECAST_ARCHITECTURES[name]
    result = train_forecaster(builder, {}, X_train, y_train, X_val, y_val, epochs=1, batch_size=8)
    preds = result.model.predict(X_val, verbose=0)
    assert preds.shape == y_val.shape
    assert result.val_loss >= 0.0
