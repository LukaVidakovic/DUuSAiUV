# Steering Angle Prediction & Lane Change Detection

Predicts the vehicle **steering angle** from a sequence of front-camera
images using a **CNN+LSTM** deep neural network, and **detects lane changes**
from the predicted angle history with a visual warning overlay.

**University Project (DUuSAiUV)** - Autonomous Driving Course

---

## Project structure

```
.
├── model.py               # VGG-like CNN + LSTM architecture
├── data_loader.py         # Keras Sequence dataset generator with augmentation
├── lane_change_detector.py# Stateful lane-change detector
├── train.py               # Training script with data balancing (CLI)
├── predict.py             # Inference + visualisation script (CLI)
├── evaluate.py            # Offline evaluation + metrics export (CLI)
├── run_all.sh             # One-command pipeline (train+evaluate+predict)
├── tests/                 # Unit tests for lane-change logic
│   └── test_lane_change_detector.py
├── artifacts/             # Generated evaluation artifacts (JSON/CSV/frames)
├── requirements.txt       # Python dependencies
├── data/                  # Dataset directory (not in git)
│   ├── self_driving_car_dataset_make/
│   │   ├── driving_log.csv
│   │   └── IMG/
│   └── self_driving_car_dataset_jungle/
│       ├── driving_log.csv
│       └── IMG/
└── README.md
```

---

## Dataset Setup

[Udacity Self-Driving Car – Behavioural Cloning](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning)

### Download and Extract

1. Download the dataset from Kaggle
2. Extract to `data/` directory:
   ```
   data/
   ├── self_driving_car_dataset_make/
   │   ├── driving_log.csv
   │   └── IMG/           # Contains all .jpg images
   └── self_driving_car_dataset_jungle/
       ├── driving_log.csv
       └── IMG/           # Contains all .jpg images
   ```

The dataset CSV (`driving_log.csv`) has the following columns:

| centercam | leftcam | rightcam | steering_angle | throttle | reverse | speed |
|-----------|---------|----------|----------------|----------|---------|-------|
| path/to/jpg | … | … | −1 … 1 | … | … | … |

**Note:** The `data/` directory is excluded from git (.gitignore).

---

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Python ≥ 3.9 and TensorFlow ≥ 2.12 are recommended.

---

## Training

```bash
python train.py \
    --csv data/self_driving_car_dataset_make/driving_log.csv \
    --data_dir data/self_driving_car_dataset_make/IMG \
    --model steering_model.keras \
    --epochs 20 \
    --batch_size 32 \
    --seq_len 3 \
    --zero_fraction 0.1
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | *(required)* | Path to `driving_log.csv` |
| `--data_dir` | *(required)* | Directory containing images (IMG folder) |
| `--model` | `steering_model_balanced.keras` | Output model path |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--seq_len` | `5` | Frames per temporal sequence (recommended: 3 for faster response) |
| `--val_split` | `0.2` | Validation split ratio |
| `--zero_fraction` | `0.1` | Fraction of zero steering angles to keep (data balancing) |

### Data Balancing & Augmentation

The training script automatically:
- **Balances the dataset** by keeping only 10% of zero steering angles
- **Applies horizontal flip augmentation** to non-zero angles (doubles the data)
- **Uses RandomContrast** augmentation during training
- **Uses Huber loss** (delta=0.1) for better handling of large steering changes

### Training Results

**Make dataset** (recommended):
- Training samples: ~2000 (after balancing)
- Best validation MAE: **0.122** (~7.0° error)
- Training time: ~7s per epoch on M-series Mac
- Configuration: seq_len=3, Huber loss (delta=0.1), dropout=0.2

**Jungle dataset**:
- Training samples: ~4000 (after balancing)
- Best validation MAE: **0.231** (~13.2° error)
- Configuration: seq_len=5, MAE loss

The best checkpoint (lowest validation loss) is automatically saved.
Training stops early if validation loss does not improve for 5 epochs.

---

## One-command run

Run full pipeline (train + evaluate + render annotated frames):

```bash
bash run_all.sh --dataset make
```

Quick checks:

```bash
bash run_all.sh --dataset make --epochs 5 --eval_max_frames 1500 --pred_max_frames 400
```

Reuse existing model (skip train):

```bash
bash run_all.sh --dataset make --skip_train --model artifacts/steering_model_make.keras
```

All outputs are written under `artifacts/run_<timestamp>_<dataset>/`.

---

## Inference & visualisation

```bash
python predict.py \
    --model    steering_model.keras \
    --csv      data/self_driving_car_dataset_make/driving_log.csv \
    --data_dir data/self_driving_car_dataset_make/IMG \
    --output_dir ./output_frames \
    --show
```

Or, using a folder of images in alphabetical order:

```bash
python predict.py \
    --model     steering_model.keras \
    --image_dir /path/to/frames/ \
    --show
```

Each annotated frame includes:

* **Predicted steering angle** (top-left text)
* **Steering-wheel icon** with a needle that rotates proportionally to the
  predicted angle
* **Red border + "! LANE CHANGE !" banner** when a lane change is detected

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to saved model |
| `--csv` / `--image_dir` | *(one required)* | Image source |
| `--seq_len` | `5` | Must match the trained model (use 3 for best model) |
| `--output_dir` | `None` | Save annotated frames here |
| `--max_frames` | `None` | Optional cap on source frames to process |
| `--show` | `False` | Display frames in a window (press **Q** to quit) |
| `--lane_threshold` | `0.2` | Abs-angle threshold for lane-change detection |
| `--lane_min_hold_frames` | `5` | Consecutive frames over threshold for candidate start |
| `--lane_settle_threshold` | `0.08` | Angle considered settled (near straight) |
| `--lane_max_settle_frames` | `25` | Max frames to confirm candidate |
| `--lane_cooldown_frames` | `20` | Frames to suppress repeated events |

---

## Architecture

```
Input: (batch, seq_len, H, W, 3)
         │
TimeDistributed
  └─ CNN (VGG-inspired)
       RandomContrast(0.3)
       Rescaling(1/255)
       Conv2D 16×3×3 → ReLU → MaxPool → BatchNorm
       Conv2D 32×3×3 → ReLU → MaxPool → BatchNorm
       Conv2D 64×3×3 → ReLU → MaxPool → BatchNorm
       Conv2D 128×3×3 → ReLU → MaxPool → BatchNorm
       Flatten → Dropout(0.4)
       Dense(128) → ReLU
         │
LSTM(64)
  └─ Dropout(0.4)
     Dense(128) → ReLU
     Dense(1)  → steering angle ∈ [−1, 1]
```

**Total parameters:** ~943K (3.6 MB)

---

## Lane-change detection

`LaneChangeDetector` keeps a rolling window of recent steering angles.
A **lane-change event** is fired using a two-stage heuristic:

1. **Candidate start:** `abs(angle)` exceeds `threshold` for at least
   `min_hold_frames` consecutive frames with the same steering sign.
2. **Confirmation:** within `max_settle_frames`, steering either returns near
   straight (`abs(angle) <= settle_threshold`) or shows opposite-sign
   counter-steer.

This reduces false positives on long constant curves compared to a pure
threshold detector. A `cooldown_frames` counter prevents repeated triggers
during the same manoeuvre.

The visual warning (red border + text banner) is displayed for 30 frames
after the event fires.

---

## Evaluation & Artifacts

Generate objective metrics and submission artifacts:

```bash
python evaluate.py \
    --model steering_model.keras \
    --csv data/self_driving_car_dataset_make/driving_log.csv \
    --data_dir data/self_driving_car_dataset_make/IMG \
    --seq_len 3 \
    --output_json artifacts/evaluation_metrics.json \
    --output_csv artifacts/frame_predictions.csv
```

Output files:
- `artifacts/evaluation_metrics.json` (aggregate steering + lane metrics)
- `artifacts/frame_predictions.csv` (per-frame GT/prediction/error/events)

To generate visual artifact frames for review:

```bash
python predict.py \
    --model steering_model.keras \
    --csv data/self_driving_car_dataset_make/driving_log.csv \
    --data_dir data/self_driving_car_dataset_make/IMG \
    --seq_len 3 \
    --output_dir artifacts/output_frames
```

---

## Testing

Run unit tests for lane-change detection:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
