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
    """Small CNN feature extractor applied to each frame (VGG-inspired).

    Args:
        image_shape: (H, W, C) of a single input frame.

    Returns:
        A Keras Model that maps (H, W, C) → flat feature vector.
    """
    inputs = layers.Input(shape=image_shape)
    
    # Data augmentation
    x = layers.RandomContrast(0.3)(inputs)
    x = layers.Rescaling(scale=1.0 / 255.0)(x)

    # Conv block 1
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Conv block 2
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Conv block 3
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Conv block 4
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)

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

    # Simplified LSTM - single layer with less dropout
    x = layers.LSTM(64, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.2)(x)  # Reduced from 0.4
    
    # Simplified dense head
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, name="steering_angle")(x)

    model = Model(inputs, outputs, name="cnn_lstm_steering")
    return model
