"""
Hybrid CNN architecture: larger kernels at start, smaller at end.
"""
from tensorflow.keras import layers, Model


def build_cnn_extractor_hybrid(image_shape: tuple) -> Model:
    """Hybrid CNN: 5×5 → 3×3 kernels with progressive channel increase."""
    inputs = layers.Input(shape=image_shape)
    
    # Data augmentation
    x = layers.RandomContrast(0.3)(inputs)
    x = layers.Rescaling(scale=1.0 / 255.0)(x)

    # Large kernels at start (capture broader features)
    x = layers.Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Medium kernels
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Small kernels at end (fine details)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    
    return Model(inputs, x, name="cnn_extractor_hybrid")


def build_cnn_lstm_model_hybrid(
    sequence_length: int = 5,
    image_shape: tuple = (66, 200, 3),
) -> Model:
    """CNN+LSTM with hybrid kernel sizes."""
    inputs = layers.Input(shape=(sequence_length, *image_shape), name="frame_sequence")

    cnn = build_cnn_extractor_hybrid(image_shape)
    x = layers.TimeDistributed(cnn, name="time_distributed_cnn")(inputs)

    x = layers.LSTM(64, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, name="steering_angle")(x)

    model = Model(inputs, outputs, name="cnn_lstm_steering_hybrid")
    return model


if __name__ == "__main__":
    model = build_cnn_lstm_model_hybrid(sequence_length=3)
    model.summary()
