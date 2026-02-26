"""
Improved training with data augmentation and balancing.
"""
import argparse
import pandas as pd
import numpy as np
import os
from tensorflow import keras
from model import build_cnn_lstm_model
from data_loader import SteeringSequence, load_dataframe

def balance_dataset(df: pd.DataFrame, zero_fraction: float = 0.1) -> pd.DataFrame:
    """Balance dataset by reducing zero steering angles."""
    zero_df = df[df["steering_angle"] == 0.0].sample(frac=zero_fraction, random_state=42)
    non_zero_df = df[df["steering_angle"] != 0.0]
    
    # Add flipped images for non-zero angles
    flipped_rows = []
    for _, row in non_zero_df.iterrows():
        flipped_row = row.copy()
        # Mark as flipped (will be handled in data loader)
        flipped_row["centercam"] = row["centercam"] + "|FLIP"
        flipped_row["steering_angle"] = -row["steering_angle"]
        flipped_rows.append(flipped_row)
    
    flipped_df = pd.DataFrame(flipped_rows)
    balanced_df = pd.concat([zero_df, non_zero_df, flipped_df], ignore_index=True)
    
    print(f"Original dataset: {len(df)} samples")
    print(f"  Zero angles: {len(df[df['steering_angle'] == 0.0])}")
    print(f"  Non-zero angles: {len(df[df['steering_angle'] != 0.0])}")
    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(f"  Zero angles: {len(balanced_df[balanced_df['steering_angle'] == 0.0])}")
    print(f"  Non-zero angles: {len(balanced_df[balanced_df['steering_angle'] != 0.0])}")
    
    return balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", default="steering_model_balanced.keras")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--zero_fraction", type=float, default=0.1)
    args = parser.parse_args()

    print("Loading and balancing data...")
    df = load_dataframe(args.csv)
    df = balance_dataset(df, args.zero_fraction)
    
    # Split
    n_val = int(len(df) * args.val_split)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)
    
    # Create generators with augmentation
    train_gen = SteeringSequence(
        train_df, args.data_dir, args.seq_len, args.batch_size,
        image_size=(66, 200), camera="center", augment=True
    )
    val_gen = SteeringSequence(
        val_df, args.data_dir, args.seq_len, args.batch_size,
        image_size=(66, 200), camera="center", augment=False
    )
    
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    
    # Build model
    print("Building model...")
    model = build_cnn_lstm_model(sequence_length=args.seq_len)
    
    # Use Huber loss - more sensitive to large errors
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.Huber(delta=0.1),  # Aggressive on large errors
        metrics=["mae"]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model, save_best_only=True, monitor="val_loss", verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
        ),
    ]
    
    # Train
    print("Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    
    print(f"Best model saved to: {os.path.abspath(args.model)}")

if __name__ == "__main__":
    main()
