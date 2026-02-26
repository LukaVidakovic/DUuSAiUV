"""
Data loading utilities for the Udacity Self-Driving Car – Behavioural Cloning dataset.

CSV column layout expected:
  centercam  leftcam  rightcam  steering_angle  throttle  reverse  speed

Images can be referenced by a path relative to ``data_dir`` (default: same
directory as the CSV file) or by an absolute path already stored in the CSV.
"""

import os
import warnings
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(raw: str, data_dir: str) -> str:
    """Return an absolute image path.

    If *raw* is already absolute and exists, it is returned as-is.
    Otherwise it is joined with *data_dir*.
    """
    raw = raw.strip()
    # Remove flip marker if present
    raw = raw.replace("|FLIP", "")
    
    # Extract just the filename from Windows paths
    if '\\' in raw or 'self_driving_car_dataset' in raw:
        # Split by backslash and get the last part (filename)
        raw = raw.replace('\\', '/').split('/')[-1]
    
    return os.path.join(data_dir, raw)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Read the dataset CSV and return a cleaned DataFrame."""
    df = pd.read_csv(csv_path, header=None, names=["centercam", "leftcam", "rightcam", "steering_angle", "throttle", "reverse", "speed"])
    df = df.dropna(subset=["steering_angle"]).reset_index(drop=True)
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_split: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into training and validation DataFrames.

    The split preserves the original temporal ordering of *df* by assigning
    the first portion to the training set and the last portion to the
    validation set, which helps avoid temporal data leakage when using
    sequences of consecutive frames.
    """
    n_samples = len(df)
    n_val = int(n_samples * val_split)

    if n_val == 0:
        train_df = df
        val_df = df.iloc[0:0]
    else:
        train_df = df.iloc[:-n_val]
        val_df = df.iloc[-n_val:]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Keras Sequence (data generator)
# ---------------------------------------------------------------------------

class SteeringSequence(Sequence):
    """Keras data generator that yields (X, y) batches.

    Each sample X has shape ``(sequence_length, H, W, C)``; y is a scalar
    steering angle.  Within a single call to ``__getitem__`` the sequence is
    built from *consecutive rows* of the (optionally shuffled) DataFrame.
    If the CSV contains multiple driving sessions/episodes, and no explicit
    episode information is used to segment the data, a single sequence may
    span an episode boundary and mix frames from different contexts.  If this
    is undesirable, callers should pre-split the CSV by episode or construct
    separate generators per session.

    Args:
        df:              Pandas DataFrame (already split into train/val).
        data_dir:        Base directory used to resolve relative image paths.
        sequence_length: Number of consecutive frames per sample.
        batch_size:      Samples per batch.
        image_size:      ``(H, W)`` to resize every frame to.
        camera:          Which camera to use: ``"center"``, ``"left"``, or
                         ``"right"``.  For multi-camera input pass
                         ``"all"`` to concatenate the three images
                         side-by-side.
        augment:         If True apply random brightness jitter.
    """

    _CAMERA_COL = {
        "center": "centercam",
        "left":   "leftcam",
        "right":  "rightcam",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        sequence_length: int = 5,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (66, 200),
        camera: str = "center",
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.image_size = image_size  # (H, W)
        self.camera = camera
        self.augment = augment

        if camera == "all":
            self.cols = ["centercam", "leftcam", "rightcam"]
        else:
            col = self._CAMERA_COL.get(camera, "centercam")
            self.cols = [col]

        # Valid end-indices: we need at least `sequence_length` preceding rows
        if len(self.df) < sequence_length:
            raise ValueError(
                f"Dataset has only {len(self.df)} rows, but sequence_length={sequence_length}. "
                "Provide more data or reduce sequence_length."
            )
        self._end_indices = list(range(sequence_length - 1, len(self.df)))

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return max(1, len(self._end_indices) // self.batch_size)

    def __getitem__(self, idx: int):
        batch_ends = self._end_indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        X, y = [], []
        for end in batch_ends:
            frames = []
            for row_idx in range(end - self.sequence_length + 1, end + 1):
                frames.append(self._load_frame(row_idx))
            X.append(frames)
            y.append(float(self.df.loc[end, "steering_angle"]))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def on_epoch_end(self) -> None:
        """Reshuffle end-indices after every epoch."""
        np.random.shuffle(self._end_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_frame(self, row_idx: int) -> np.ndarray:
        """Load, resize (and optionally augment) the frame at *row_idx*."""
        # Check if this is a flipped image
        cam_path = str(self.df.loc[row_idx, self.cols[0]])
        should_flip = "|FLIP" in cam_path
        
        if self.camera == "all":
            imgs = [self._read_image(row_idx, col) for col in self.cols]
            img = np.concatenate(imgs, axis=1)  # side-by-side → (H, 3*W, C)
        else:
            img = self._read_image(row_idx, self.cols[0])
        
        # Apply flip if marked
        if should_flip:
            img = cv2.flip(img, 1)
        
        if self.augment:
            img = self._random_brightness(img)
        return img

    def _read_image(self, row_idx: int, col: str) -> np.ndarray:
        raw_path = str(self.df.loc[row_idx, col])
        abs_path = _resolve_path(raw_path, self.data_dir)
        img = cv2.imread(abs_path)
        if img is None:
            warnings.warn(f"Could not read image: {abs_path}; using blank frame.")
            h, w = self.image_size
            return np.zeros((h, w, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))  # (W, H)
        return img

    @staticmethod
    def _random_brightness(img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        ratio = 0.4 + np.random.random() * 1.2  # [0.4, 1.6]
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# Convenience loader used by train.py
# ---------------------------------------------------------------------------

def make_generators(
    csv_path: str,
    data_dir: Optional[str] = None,
    sequence_length: int = 5,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (66, 200),
    camera: str = "center",
    val_split: float = 0.2,
) -> Tuple["SteeringSequence", "SteeringSequence"]:
    """Return (train_gen, val_gen) ready for ``model.fit``."""
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(csv_path))

    df = load_dataframe(csv_path)
    train_df, val_df = split_dataframe(df, val_split=val_split)

    train_gen = SteeringSequence(
        train_df, data_dir, sequence_length, batch_size, image_size,
        camera=camera, augment=True,
    )
    val_gen = SteeringSequence(
        val_df, data_dir, sequence_length, batch_size, image_size,
        camera=camera, augment=False,
    )
    return train_gen, val_gen
