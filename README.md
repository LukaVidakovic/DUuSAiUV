# Steering Angle Prediction & Lane Change Detection

Predicts the vehicle **steering angle** from a sequence of front-camera
images using a **CNN+LSTM** deep neural network, and **detects lane changes**
from the predicted angle history with a visual warning overlay.

---

## Project structure

```
.
├── model.py               # CNN+LSTM architecture definition
├── data_loader.py         # Keras Sequence dataset generator
├── lane_change_detector.py# Stateful lane-change detector
├── train.py               # Training script (CLI)
├── predict.py             # Inference + visualisation script (CLI)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Dataset

[Udacity Self-Driving Car – Behavioural Cloning](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning)

The dataset CSV (`driving_log.csv`) has the following columns:

| centercam | leftcam | rightcam | steering_angle | throttle | reverse | speed |
|-----------|---------|----------|----------------|----------|---------|-------|
| path/to/jpg | … | … | −1 … 1 | … | … | … |

Download and extract the dataset so that the CSV file and the `IMG/`
directory containing the JPEG images are available locally.

---

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 and TensorFlow ≥ 2.12 are recommended.

---

## Training

```bash
python train.py \
    --csv      /path/to/driving_log.csv \
    --data_dir /path/to/IMG \
    --model    steering_model.keras \
    --epochs   15 \
    --batch_size 32 \
    --seq_len  5 \
    --camera   center
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | *(required)* | Path to `driving_log.csv` |
| `--data_dir` | CSV directory | Base directory for image paths |
| `--model` | `steering_model.keras` | Output model path |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--seq_len` | `5` | Frames per temporal sequence |
| `--camera` | `center` | Camera: `center`, `left`, `right`, or `all` |
| `--image_height` | `66` | Resize height |
| `--image_width` | `200` | Resize width |
| `--lr` | `1e-3` | Initial Adam learning rate |

The best checkpoint (lowest validation MSE) is automatically saved to
`--model`.  Training stops early if validation loss does not improve for
5 epochs.

### Example training time

A model with ~100 K parameters trained on a subset of the Udacity dataset
takes roughly **1–2 minutes per epoch** on an Intel Core i7 CPU.

---

## Inference & visualisation

```bash
python predict.py \
    --model    steering_model.keras \
    --csv      /path/to/driving_log.csv \
    --data_dir /path/to/IMG \
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
| `--seq_len` | `5` | Must match the trained model |
| `--output_dir` | `None` | Save annotated frames here |
| `--show` | `False` | Display frames in a window (press **Q** to quit) |
| `--lane_threshold` | `0.2` | Abs-angle threshold for lane-change detection |

---

## Architecture

```
Input: (batch, seq_len, H, W, 3)
         │
TimeDistributed
  └─ CNN (NVIDIA-inspired)
       Conv2D 24×5×5 stride 2  → ELU
       Conv2D 36×5×5 stride 2  → ELU
       Conv2D 48×5×5 stride 2  → ELU
       Conv2D 64×3×3            → ELU
       Conv2D 64×3×3            → ELU
       Flatten → Dense(100)    → ELU
         │
LSTM(64)
  └─ Dropout(0.2)
     Dense(50) → ELU
     Dense(10) → ELU
     Dense(1)  → steering angle ∈ [−1, 1]
```

---

## Lane-change detection

`LaneChangeDetector` keeps a rolling window of recent steering angles.
A **lane-change event** is fired when the absolute angle exceeds
`threshold` (default 0.2) for at least `min_hold_frames` (default 5)
consecutive frames.  A `cooldown_frames` (default 20) counter prevents
repeated triggers during the same manoeuvre.

The visual warning (red border + text banner) is displayed for 30 frames
after the event fires.