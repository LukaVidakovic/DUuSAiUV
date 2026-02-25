"""
CNN+LSTM model for steering angle prediction.

Architecture:
  - TimeDistributed CNN: extracts spatial features from each frame in the sequence.
  - LSTM: models temporal dependencies across the feature sequence.
  - Dense head: regresses to a single steering angle in [-1, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_extractor(image_shape: tuple) -> Model:
    """Small CNN feature extractor applied to each frame (NVIDIA-inspired).

    Args:
        image_shape: (H, W, C) of a single input frame.

    Returns:
        A Keras Model that maps (H, W, C) → flat feature vector.
    """
    inputs = layers.Input(shape=image_shape)
    # Normalise to [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)

    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="elu", padding="valid")(x)
    x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation="elu", padding="valid")(x)
    x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation="elu", padding="valid")(x)
    x = layers.Conv2D(64, (3, 3), activation="elu", padding="valid")(x)
    x = layers.Conv2D(64, (3, 3), activation="elu", padding="valid")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="elu")(x)

    return Model(inputs, x, name="cnn_extractor")


def build_cnn_lstm_model(
    sequence_length: int = 5,
    image_shape: tuple = (66, 200, 3),
) -> Model:
    """Full CNN+LSTM steering angle prediction model.

    Args:
        sequence_length: Number of consecutive frames fed as one input sample.
        image_shape:     (H, W, C) of each frame.

    Returns:
        Uncompiled Keras Model.  Input shape: (batch, seq_len, H, W, C).
        Output shape: (batch, 1) — predicted steering angle.
    """
    inputs = layers.Input(shape=(sequence_length, *image_shape), name="frame_sequence")

    cnn = build_cnn_extractor(image_shape)
    x = layers.TimeDistributed(cnn, name="time_distributed_cnn")(inputs)

    x = layers.LSTM(64, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="elu")(x)
    x = layers.Dense(10, activation="elu")(x)
    outputs = layers.Dense(1, name="steering_angle")(x)

    model = Model(inputs, outputs, name="cnn_lstm_steering")
    return model
