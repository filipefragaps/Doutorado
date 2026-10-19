from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from keras import layers


def _compile_regressor(model: keras.Model) -> keras.Model:
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_forecast_lstm(input_shape: tuple[int, int], n_features: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.LSTM(units, dropout=dropout),
        layers.Dense(n_features),
    ])
    return _compile_regressor(model)


def build_forecast_gru(input_shape: tuple[int, int], n_features: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.GRU(units, dropout=dropout),
        layers.Dense(n_features),
    ])
    return _compile_regressor(model)


def build_forecast_bilstm(input_shape: tuple[int, int], n_features: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(units, dropout=dropout)),
        layers.Dense(n_features),
    ])
    return _compile_regressor(model)


FORECAST_ARCHITECTURES = {
    "lstm": build_forecast_lstm,
    "gru": build_forecast_gru,
    "bilstm": build_forecast_bilstm,
}


@dataclass
class ForecastResult:
    model: keras.Model
    history: dict
    val_loss: float
    val_mae: float


def train_forecaster(
    model_builder, build_kwargs: dict,
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    *, epochs: int = 20, batch_size: int = 64, patience: int = 5, verbose: int = 0,
) -> ForecastResult:
    """Train a sequence-to-vector forecaster: given a window of recent
    acquisitions, predict the feature vector `forecast_horizon` steps ahead
    (the 'assinatura futura das harmônicas' the user asked for). Rows with a
    NaN forecast target (windows too close to the end of a run) must be
    filtered out by the caller before this is called."""

    model = model_builder(input_shape=X_train.shape[1:], n_features=y_train.shape[-1], **build_kwargs)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=verbose,
    )

    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    return ForecastResult(model=model, history=history.history, val_loss=val_loss, val_mae=val_mae)
