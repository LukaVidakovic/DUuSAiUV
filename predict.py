"""
Inference script: load a trained CNN+LSTM model, predict steering angles for
a sequence of images (provided via a CSV or a folder), overlay the predicted
angle on each frame, detect lane changes and display visual warnings.

Usage examples
--------------
Run on the full dataset CSV (saves annotated frames to --output_dir):
    python predict.py \\
        --model steering_model.keras \\
        --csv   /data/driving_log.csv \\
        --data_dir /data/IMG \\
        --output_dir ./output_frames

Run without saving, just display frames:
    python predict.py \\
        --model steering_model.keras \\
        --csv   /data/driving_log.csv \\
        --show

Run on a plain image folder (alphabetical order):
    python predict.py \\
        --model steering_model.keras \\
        --image_dir /data/frames/ \\
        --show
"""

import argparse
import math
import os
import sys
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from lane_change_detector import LaneChangeDetector


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_steering_angle(
    frame: np.ndarray,
    angle: float,
    lane_change: bool = False,
) -> np.ndarray:
    """Return a copy of *frame* with steering indicator and optional warning.

    Visual elements
    ---------------
    * Numeric steering angle (top-left corner).
    * Steering-wheel icon (circle + needle) at the bottom-centre.
    * Red border + "⚠ LANE CHANGE" banner when *lane_change* is True.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # ------------------------------------------------------------------
    # 1. Numeric label
    # ------------------------------------------------------------------
    label = f"Steering: {angle:+.3f}"
    cv2.putText(
        out, label, (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2, cv2.LINE_AA,
    )

    # ------------------------------------------------------------------
    # 2. Steering-wheel icon
    # ------------------------------------------------------------------
    cx, cy = w // 2, h - 50
    radius = 35
    cv2.circle(out, (cx, cy), radius, (220, 220, 220), 2, cv2.LINE_AA)
    # Needle points in the direction of the steering angle
    # angle = 0 → needle points up; angle = ±1 → needle at ±90°
    needle_rad = angle * math.pi / 2.0
    nx = int(cx + radius * math.sin(needle_rad))
    ny = int(cy - radius * math.cos(needle_rad))
    cv2.line(out, (cx, cy), (nx, ny), (0, 230, 0), 3, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 4, (0, 230, 0), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 3. Lane-change warning (visual)
    # ------------------------------------------------------------------
    if lane_change:
        # Red border
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        # Warning banner
        banner_text = "! LANE CHANGE !"
        font_scale = max(0.6, w / 640)
        (tw, th), _ = cv2.getTextSize(
            banner_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2
        )
        bx = (w - tw) // 2
        by = 70
        # Semi-transparent background for readability
        overlay = out.copy()
        cv2.rectangle(overlay, (bx - 8, by - th - 8), (bx + tw + 8, by + 8),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
        cv2.putText(
            out, banner_text, (bx, by),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 50, 255), 2, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Image sources
# ---------------------------------------------------------------------------

def _image_paths_from_csv(
    csv_path: str,
    data_dir: str,
    camera: str = "center",
) -> List[Tuple[str, float]]:
    """Return list of (image_path, ground_truth_angle) from a CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {"center": "centercam", "left": "leftcam", "right": "rightcam"}
    img_col = col_map.get(camera, "centercam")

    pairs = []
    for _, row in df.iterrows():
        raw = str(row[img_col]).strip()
        path = raw if (os.path.isabs(raw) and os.path.exists(raw)) \
               else os.path.join(data_dir, raw)
        gt = float(row["steering_angle"])
        pairs.append((path, gt))
    return pairs


def _image_paths_from_folder(folder: str) -> List[Tuple[str, float]]:
    """Return list of (image_path, 0.0) sorted alphabetically."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(
        p for p in (os.path.join(folder, f) for f in os.listdir(folder))
        if os.path.splitext(p)[1].lower() in exts
    )
    return [(p, 0.0) for p in paths]


# ---------------------------------------------------------------------------
# Frame loader / resizer
# ---------------------------------------------------------------------------

def load_and_resize(path: str, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load an image and resize to (H, W).  Returns None on failure."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    return img


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    pairs: List[Tuple[str, float]],
    image_size: Tuple[int, int],
    sequence_length: int = 5,
    output_dir: Optional[str] = None,
    show: bool = False,
    detector_threshold: float = 0.2,
) -> None:
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    detector = LaneChangeDetector(threshold=detector_threshold)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Rolling buffer for the sequence window
    frame_buffer: List[np.ndarray] = []
    lane_change_frames = 0  # how many frames to keep the warning visible

    for frame_idx, (img_path, gt_angle) in enumerate(pairs):
        img = load_and_resize(img_path, image_size)
        if img is None:
            print(f"  WARNING: could not read {img_path}, skipping.")
            continue

        frame_buffer.append(img)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)

        # Only predict once we have a full sequence
        if len(frame_buffer) < sequence_length:
            continue

        seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis]  # (1, T, H, W, 3)
        pred_angle = float(model.predict(seq, verbose=0)[0, 0])
        pred_angle = float(np.clip(pred_angle, -1.0, 1.0))

        new_event = detector.update(pred_angle)
        if new_event:
            lane_change_frames = 30  # show warning for 30 frames

        lane_change_active = lane_change_frames > 0
        if lane_change_frames > 0:
            lane_change_frames -= 1

        annotated = draw_steering_angle(img, pred_angle, lane_change=lane_change_active)

        # Print ground-truth comparison when available
        if gt_angle != 0.0:
            print(
                f"Frame {frame_idx:05d} | "
                f"GT: {gt_angle:+.4f}  Pred: {pred_angle:+.4f}  "
                f"Err: {abs(pred_angle - gt_angle):.4f}"
                + ("  ⚠ LANE CHANGE" if new_event else "")
            )
        else:
            print(
                f"Frame {frame_idx:05d} | Pred: {pred_angle:+.4f}"
                + ("  ⚠ LANE CHANGE" if new_event else "")
            )

        if output_dir:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            # Convert back to BGR for OpenCV
            cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        if show:
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow("Steering Prediction", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Interrupted by user.")
                break

    if show:
        cv2.destroyAllWindows()

    print("Inference complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the CNN+LSTM steering model."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the saved Keras model (steering_model.keras).",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv",
        help="Path to dataset CSV (centercam/steering_angle columns used).",
    )
    src.add_argument(
        "--image_dir",
        help="Directory of images to process in alphabetical order.",
    )

    parser.add_argument(
        "--data_dir", default=None,
        help=(
            "Base directory for image paths in the CSV.  "
            "Defaults to the CSV directory."
        ),
    )
    parser.add_argument(
        "--camera", default="center",
        choices=["center", "left", "right"],
        help="Camera to use when reading from a CSV (default: center).",
    )
    parser.add_argument(
        "--seq_len", type=int, default=5,
        help="Sequence length (must match the trained model).",
    )
    parser.add_argument(
        "--image_height", type=int, default=66,
        help="Frame height expected by the model.",
    )
    parser.add_argument(
        "--image_width", type=int, default=200,
        help="Frame width expected by the model.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save annotated frames (optional).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display frames in a window while running.",
    )
    parser.add_argument(
        "--lane_threshold", type=float, default=0.2,
        help="Steering-angle threshold for lane-change detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_size = (args.image_height, args.image_width)

    if args.csv:
        data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.csv))
        pairs = _image_paths_from_csv(args.csv, data_dir, camera=args.camera)
    else:
        pairs = _image_paths_from_folder(args.image_dir)

    if not pairs:
        print("ERROR: no images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(pairs)} frames …")
    run_inference(
        model_path=args.model,
        pairs=pairs,
        image_size=image_size,
        sequence_length=args.seq_len,
        output_dir=args.output_dir,
        show=args.show,
        detector_threshold=args.lane_threshold,
    )


if __name__ == "__main__":
    main()
