import numpy as np
import pytest

from harmonic_signatures.deep_learning.architectures import ARCHITECTURES, build_lstm
from harmonic_signatures.deep_learning.training import train_classifier
from harmonic_signatures.deep_learning.tuning import random_search


@pytest.fixture
def tiny_sequence_data():
    rng = np.random.default_rng(0)
    n, window, channels = 60, 4, 3
    X = rng.normal(size=(n, window, channels)).astype("float32")
    y = np.array(["A", "B"])[rng.integers(0, 2, size=n)]
    return X[:45], y[:45], X[45:], y[45:]


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_architecture_trains_one_epoch(name, tiny_sequence_data):
    X_train, y_train, X_val, y_val = tiny_sequence_data
    builder = ARCHITECTURES[name]
    result = train_classifier(builder, {}, X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    assert 0.0 <= result.val_accuracy <= 1.0
    assert result.val_loss >= 0.0


def test_random_search_returns_best_params(tiny_sequence_data):
    X_train, y_train, X_val, y_val = tiny_sequence_data
    grid = {"units": [8, 16], "dropout": [0.1, 0.2]}
    trials_df, best_params, best_model = random_search(
        build_lstm, grid, X_train, y_train, X_val, y_val, n_trials=3, epochs=1,
    )
    assert len(trials_df) == 3
    assert set(best_params) == {"units", "dropout"}
    assert list(trials_df["val_accuracy"]) == sorted(trials_df["val_accuracy"], reverse=True)
