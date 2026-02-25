"""
Training script for the CNN+LSTM steering angle model.

Usage example
-------------
    python train.py \\
        --csv  /data/driving_log.csv \\
        --data_dir /data/IMG \\
        --model steering_model.keras \\
        --epochs 15 \\
        --batch_size 32 \\
        --seq_len 5 \\
        --camera center
"""

import argparse
import os

import tensorflow as tf

from data_loader import make_generators
from model import build_cnn_lstm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNN+LSTM steering angle model."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the dataset CSV file (driving_log.csv).",
    )
    parser.add_argument(
        "--data_dir", default=None,
        help=(
            "Base directory for image files referenced in the CSV.  "
            "Defaults to the directory containing the CSV file."
        ),
    )
    parser.add_argument(
        "--model", default="steering_model.keras",
        help="Output path for the saved model (default: steering_model.keras).",
    )
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--seq_len",     type=int,   default=5,
                        help="Number of consecutive frames per sample.")
    parser.add_argument(
        "--image_height", type=int, default=66,
        help="Height to resize input images to (default: 66).",
    )
    parser.add_argument(
        "--image_width", type=int, default=200,
        help="Width to resize input images to (default: 200).",
    )
    parser.add_argument(
        "--camera", default="center",
        choices=["center", "left", "right", "all"],
        help="Which camera(s) to use for training.",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate for the Adam optimiser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_size = (args.image_height, args.image_width)

    # Adjust width if using all three cameras side-by-side
    if args.camera == "all":
        model_image_shape = (args.image_height, args.image_width * 3, 3)
    else:
        model_image_shape = (args.image_height, args.image_width, 3)

    print("Building model …")
    model = build_cnn_lstm_model(
        sequence_length=args.seq_len,
        image_shape=model_image_shape,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    print("Loading data …")
    train_gen, val_gen = make_generators(
        csv_path=args.csv,
        data_dir=args.data_dir,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        image_size=image_size,
        camera=args.camera,
        val_split=args.val_split,
    )
    print(f"  Training batches : {len(train_gen)}")
    print(f"  Validation batches: {len(val_gen)}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training …")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the best model during training
    print(f"Best model saved to: {os.path.abspath(args.model)}")

    # Print training summary
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))
    print(f"Best validation loss (MSE): {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
