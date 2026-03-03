"""
Offline evaluation for steering-angle prediction and lane-change events.

Outputs:
  - Steering regression metrics (MAE/RMSE/Max error/Correlation/R2).
  - Pseudo lane-change event metrics by comparing event extraction from
    ground-truth steering vs. predicted steering using the same detector.
  - Optional per-frame CSV and JSON report for submission artifacts.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from lane_change_detector import LaneChangeDetector


CSV_COLUMNS = [
    "centercam",
    "leftcam",
    "rightcam",
    "steering_angle",
    "throttle",
    "reverse",
    "speed",
]


def _resolve_csv_image_path(raw_path: str, data_dir: str) -> str:
    raw = str(raw_path).strip()
    if "\\" in raw or "self_driving_car_dataset" in raw:
        raw = raw.replace("\\", "/").split("/")[-1]
    return os.path.join(data_dir, raw)


def image_pairs_from_csv(
    csv_path: str,
    data_dir: str,
    camera: str = "center",
) -> List[Tuple[str, float]]:
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    col_map = {"center": "centercam", "left": "leftcam", "right": "rightcam"}
    image_col = col_map[camera]

    pairs: List[Tuple[str, float]] = []
    for _, row in df.iterrows():
        path = _resolve_csv_image_path(row[image_col], data_dir)
        try:
            gt_angle = float(row["steering_angle"])
        except (TypeError, ValueError):
            continue
        pairs.append((path, gt_angle))
    return pairs


def load_and_resize(path: str, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (image_size[1], image_size[0]))


def binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    yt = np.array(y_true, dtype=bool)
    yp = np.array(y_pred, dtype=bool)

    tp = int(np.sum(yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / max(1, len(yt))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "support_positive": int(np.sum(yt)),
        "predicted_positive": int(np.sum(yp)),
    }


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def evaluate(
    model_path: str,
    pairs: List[Tuple[str, float]],
    image_size: Tuple[int, int],
    sequence_length: int,
    lane_threshold: float,
    lane_min_hold_frames: int,
    lane_settle_threshold: float,
    lane_max_settle_frames: int,
    lane_cooldown_frames: int,
    max_frames: Optional[int] = None,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    model = tf.keras.models.load_model(model_path)

    gt_detector = LaneChangeDetector(
        threshold=lane_threshold,
        min_hold_frames=lane_min_hold_frames,
        settle_threshold=lane_settle_threshold,
        max_settle_frames=lane_max_settle_frames,
        cooldown_frames=lane_cooldown_frames,
    )
    pred_detector = LaneChangeDetector(
        threshold=lane_threshold,
        min_hold_frames=lane_min_hold_frames,
        settle_threshold=lane_settle_threshold,
        max_settle_frames=lane_max_settle_frames,
        cooldown_frames=lane_cooldown_frames,
    )

    frame_buffer: List[np.ndarray] = []

    rows: List[Dict[str, object]] = []
    gt_vals: List[float] = []
    pred_vals: List[float] = []
    gt_events: List[bool] = []
    pred_events: List[bool] = []

    for idx, (img_path, gt_angle) in enumerate(pairs):
        if max_frames is not None and idx >= max_frames:
            break

        img = load_and_resize(img_path, image_size)
        if img is None:
            continue

        frame_buffer.append(img)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)

        if len(frame_buffer) < sequence_length:
            continue

        seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis]
        pred_angle = float(model.predict(seq, verbose=0)[0, 0])
        pred_angle = float(np.clip(pred_angle, -1.0, 1.0))

        gt_event = bool(gt_detector.update(gt_angle))
        pred_event = bool(pred_detector.update(pred_angle))

        err = abs(pred_angle - gt_angle)
        rows.append(
            {
                "frame_idx": idx,
                "image_path": img_path,
                "gt_angle": float(gt_angle),
                "pred_angle": pred_angle,
                "abs_error": float(err),
                "gt_lane_change": gt_event,
                "pred_lane_change": pred_event,
            }
        )

        gt_vals.append(float(gt_angle))
        pred_vals.append(pred_angle)
        gt_events.append(gt_event)
        pred_events.append(pred_event)

    if not rows:
        raise RuntimeError(
            "No valid evaluation frames were produced. "
            "Check paths, model input size, and sequence length."
        )

    y_true = np.array(gt_vals, dtype=np.float32)
    y_pred = np.array(pred_vals, dtype=np.float32)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    max_abs_error = float(np.max(np.abs(y_true - y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    r2 = float(r2_score(y_true, y_pred))

    lane_stats = binary_metrics(gt_events, pred_events)

    report: Dict[str, object] = {
        "num_scored_frames": len(rows),
        "sequence_length": sequence_length,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "steering_metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "max_abs_error": max_abs_error,
            "pearson_corr": corr,
            "r2": r2,
        },
        "lane_change_metrics_pseudo_label": lane_stats,
        "lane_detector_config": {
            "threshold": lane_threshold,
            "min_hold_frames": lane_min_hold_frames,
            "settle_threshold": lane_settle_threshold,
            "max_settle_frames": lane_max_settle_frames,
            "cooldown_frames": lane_cooldown_frames,
        },
    }
    return report, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate steering model and export metrics/report artifacts."
    )
    parser.add_argument("--model", required=True, help="Path to saved Keras model.")
    parser.add_argument("--csv", required=True, help="Path to driving_log.csv.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Image directory for CSV paths. Defaults to CSV directory.",
    )
    parser.add_argument(
        "--camera",
        default="center",
        choices=["center", "left", "right"],
        help="Camera column used from CSV.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=5,
        help="Sequence length expected by the model.",
    )
    parser.add_argument("--image_height", type=int, default=66)
    parser.add_argument("--image_width", type=int, default=200)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap for faster checks (e.g., 1000).",
    )

    parser.add_argument("--lane_threshold", type=float, default=0.2)
    parser.add_argument("--lane_min_hold_frames", type=int, default=5)
    parser.add_argument("--lane_settle_threshold", type=float, default=0.08)
    parser.add_argument("--lane_max_settle_frames", type=int, default=25)
    parser.add_argument("--lane_cooldown_frames", type=int, default=20)

    parser.add_argument(
        "--output_json",
        default="artifacts/evaluation_metrics.json",
        help="Where to write aggregated metrics JSON.",
    )
    parser.add_argument(
        "--output_csv",
        default="artifacts/frame_predictions.csv",
        help="Where to write per-frame predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.csv))
    pairs = image_pairs_from_csv(args.csv, data_dir=data_dir, camera=args.camera)
    if not pairs:
        print("ERROR: no image/angle rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    report, frame_df = evaluate(
        model_path=args.model,
        pairs=pairs,
        image_size=(args.image_height, args.image_width),
        sequence_length=args.seq_len,
        lane_threshold=args.lane_threshold,
        lane_min_hold_frames=args.lane_min_hold_frames,
        lane_settle_threshold=args.lane_settle_threshold,
        lane_max_settle_frames=args.lane_max_settle_frames,
        lane_cooldown_frames=args.lane_cooldown_frames,
        max_frames=args.max_frames,
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    frame_df.to_csv(args.output_csv, index=False)

    sm = report["steering_metrics"]
    lm = report["lane_change_metrics_pseudo_label"]
    print("Evaluation complete.")
    print(f"Scored frames: {report['num_scored_frames']}")
    print(f"MAE:  {sm['mae']:.6f}")
    print(f"RMSE: {sm['rmse']:.6f}")
    print(f"Corr: {sm['pearson_corr']:.6f}")
    print(f"R2:   {sm['r2']:.6f}")
    print(
        "Lane event (pseudo-label) P/R/F1: "
        f"{lm['precision']:.3f} / {lm['recall']:.3f} / {lm['f1']:.3f}"
    )
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
