"""
Deep CNN+LSTM model with stacked LSTM layers for better temporal modeling.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_extractor(image_shape: tuple) -> Model:
    """VGG-inspired CNN feature extractor."""
    inputs = layers.Input(shape=image_shape)
    
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
    """Deep CNN+LSTM with stacked LSTM layers."""
    inputs = layers.Input(shape=(sequence_length, *image_shape), name="frame_sequence")

    cnn = build_cnn_extractor(image_shape)
    x = layers.TimeDistributed(cnn, name="time_distributed_cnn")(inputs)

    # Stacked LSTM layers for better temporal modeling
    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, name="steering_angle")(x)

    model = Model(inputs, outputs, name="cnn_lstm_steering_deep")
    return model


if __name__ == "__main__":
    model = build_cnn_lstm_model(sequence_length=3)
    model.summary()
