from __future__ import annotations

from tensorflow import keras
from keras import layers


def _compile(model: keras.Model) -> keras.Model:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_simple_rnn(input_shape: tuple[int, int], n_classes: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.SimpleRNN(units, dropout=dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return _compile(model)


def build_lstm(input_shape: tuple[int, int], n_classes: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.LSTM(units, dropout=dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return _compile(model)


def build_gru(input_shape: tuple[int, int], n_classes: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.GRU(units, dropout=dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return _compile(model)


def build_bilstm(input_shape: tuple[int, int], n_classes: int, *, units: int = 64, dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(units, dropout=dropout)),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return _compile(model)


def build_cnn_lstm(
    input_shape: tuple[int, int], n_classes: int, *, filters: int = 32, kernel_size: int = 3,
    lstm_units: int = 64, dropout: float = 0.2,
) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(filters, kernel_size, padding="causal", activation="relu"),
        layers.LSTM(lstm_units, dropout=dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return _compile(model)


def build_transformer(
    input_shape: tuple[int, int], n_classes: int, *, num_heads: int = 4, key_dim: int = 32,
    ff_dim: int = 64, dropout: float = 0.1,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(key_dim)(inputs)  # project channels so attention has a stable head dim
    x = x + layers.Embedding(input_dim=input_shape[0], output_dim=key_dim)(
        keras.ops.arange(input_shape[0])
    )
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn_out)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(key_dim)(ff)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inputs, outputs))


ARCHITECTURES = {
    "simple_rnn": build_simple_rnn,
    "lstm": build_lstm,
    "gru": build_gru,
    "bilstm": build_bilstm,
    "cnn_lstm": build_cnn_lstm,
    "transformer": build_transformer,
}
