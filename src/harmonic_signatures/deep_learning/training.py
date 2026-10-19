from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


@dataclass
class TrainResult:
    model: keras.Model
    history: dict
    val_accuracy: float
    val_loss: float


def train_classifier(
    model_builder,
    build_kwargs: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, epochs: int = 20, batch_size: int = 64, patience: int = 5, verbose: int = 0,
    label_encoder: LabelEncoder | None = None,
) -> TrainResult:
    encoder = label_encoder or LabelEncoder().fit(np.concatenate([y_train, y_val]))
    y_train_enc = encoder.transform(y_train)
    y_val_enc = encoder.transform(y_val)

    model = model_builder(input_shape=X_train.shape[1:], n_classes=len(encoder.classes_), **build_kwargs)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train, y_train_enc, validation_data=(X_val, y_val_enc),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=verbose,
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val_enc, verbose=0)
    return TrainResult(model=model, history=history.history, val_accuracy=val_accuracy, val_loss=val_loss)
