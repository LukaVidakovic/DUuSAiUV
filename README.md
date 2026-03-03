# Steering Angle Prediction & Lane Change Detection

Predicts vehicle **steering angle** from front-camera image sequences using a **CNN+LSTM** deep neural network, and **detects lane changes** from predicted angle history with visual warnings.

**University Project (DUuSAiUV)** - Autonomous Driving Course

---

## Quick Start (Easiest Way)

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download dataset (see Dataset Setup section)

# 3. Run complete pipeline (train + evaluate + visualize)
./run_all.sh --dataset make --show
```

This single command will:

- Train the model (~2 minutes)
- Evaluate on full dataset
- Generate annotated frames with visualization
- Display results in window (press Q to exit)

All results saved to `artifacts/run_<timestamp>_make/`

---

## Project Structure

```
.
├── model.py               # VGG-like CNN + LSTM architecture (production)
├── model_hybrid.py        # Hybrid CNN (5×5→3×3 kernels, faster training)
├── data_loader.py         # Keras Sequence dataset generator with augmentation
├── lane_change_detector.py# Two-stage stateful lane-change detector
├── train.py               # Training script with data balancing (CLI)
├── train_hybrid.py        # Training script for hybrid model
├── predict.py             # Inference + visualization script (CLI)
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

## Python Files Overview

### Core Model Files

**`model.py`** - Production CNN+LSTM architecture

- VGG-inspired CNN with 3×3 kernels (16→32→64→128 filters)
- Single LSTM layer (64 units) for temporal modeling
- Best accuracy: 0.100 MAE (~5.7° error)
- Training time: ~7s/epoch on M-series Mac
- **Use for**: Production deployment, best accuracy

**`model_hybrid.py`** - Fast prototyping architecture

- Progressive kernel sizes (5×5 → 3×3)
- Larger receptive field in early layers
- Accuracy: 0.110 MAE (~6.3° error)
- Training time: ~5s/epoch (30% faster)
- **Use for**: Quick experiments, faster iterations

### Training Scripts

**`train.py`** - Main training script

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

Features:

- Automatic data balancing (keeps only 10% of zero angles)
- Horizontal flip augmentation for non-zero angles
- Huber loss (delta=0.1) for better large error handling
- Early stopping + learning rate reduction
- Saves best model checkpoint

**`train_hybrid.py`** - Training for hybrid model (same interface as `train.py`)

### Inference & Visualization

**`predict.py`** - Real-time inference with visualization

```bash
# Run on dataset CSV
python predict.py \
    --model steering_model.keras \
    --csv data/self_driving_car_dataset_make/driving_log.csv \
    --data_dir data/self_driving_car_dataset_make/IMG \
    --seq_len 3 \
    --show

# Run on image folder (alphabetical order)
python predict.py \
    --model steering_model.keras \
    --image_dir /path/to/frames/ \
    --show
```

Visualization features:

- **Info panel**: Prediction, Ground Truth, Error (when GT available)
- **Steering wheel**: Dual needles (green=prediction, yellow=GT)
- **Lane change warning**: Red border + "! LANE CHANGE !" banner
- **Save frames**: Use `--output_dir` to save annotated images

Key options:

- `--show`: Display frames in window (press Q to quit)
- `--max_frames N`: Limit number of frames to process
- `--output_dir PATH`: Save annotated frames
- `--lane_threshold 0.2`: Steering angle threshold for lane change detection
- `--lane_min_hold_frames 5`: Consecutive frames needed for candidate start
- `--lane_cooldown_frames 20`: Suppress repeated triggers

### Evaluation & Metrics

**`evaluate.py`** - Comprehensive offline evaluation

```bash
python evaluate.py \
    --model steering_model.keras \
    --csv data/self_driving_car_dataset_make/driving_log.csv \
    --data_dir data/self_driving_car_dataset_make/IMG \
    --seq_len 3 \
    --output_json artifacts/metrics.json \
    --output_csv artifacts/predictions.csv
```

Outputs:

- **JSON metrics**: MAE, RMSE, Max Error, Correlation, R², lane change stats
- **CSV predictions**: Per-frame GT/prediction/error/lane events
- **Use for**: Submission artifacts, model comparison, debugging

### Data Loading

**`data_loader.py`** - Keras Sequence generator

- Handles CSV parsing and image loading
- Applies horizontal flip augmentation (marked with `|FLIP` suffix)
- Creates temporal sequences (sliding window)
- Robust path resolution for different CSV formats
- **Used internally** by train/evaluate/predict scripts

### Lane Change Detection

**`lane_change_detector.py`** - Two-stage heuristic detector

- **Stage 1 (Candidate)**: Sustained steering over threshold (5+ frames)
- **Stage 2 (Confirmation)**: Settling near straight or counter-steer
- Cooldown period to prevent repeated triggers
- **Used internally** by predict.py for visual warnings

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

| centercam   | leftcam | rightcam | steering_angle | throttle | reverse | speed |
| ----------- | ------- | -------- | -------------- | -------- | ------- | ----- |
| path/to/jpg | …       | …        | −1 … 1         | …        | …       | …     |

**Note:** The `data/` directory is excluded from git (.gitignore).

---

## Complete Pipeline with `run_all.sh`

The **easiest and recommended way** to run the entire project:

### Basic Usage

```bash
# Train + evaluate + visualize (make dataset)
./run_all.sh --dataset make --show

# Train + evaluate + visualize (jungle dataset)
./run_all.sh --dataset jungle --show

# Quick test (5 epochs, limited frames)
./run_all.sh --dataset make --epochs 5 --eval_max_frames 1500 --pred_max_frames 400

# Skip training, use existing model
./run_all.sh --dataset make --skip_train --model steering_model_make_huber.keras --show
```

### What It Does

1. **Training Phase** (unless `--skip_train`)

   - Loads and balances dataset
   - Trains CNN+LSTM model
   - Saves best checkpoint to `artifacts/run_<timestamp>_<dataset>/steering_model.keras`
2. **Evaluation Phase**

   - Runs model on full dataset (or `--eval_max_frames`)
   - Calculates MAE, RMSE, Correlation, R²
   - Saves metrics to `evaluation_metrics.json`
   - Saves per-frame predictions to `frame_predictions.csv`
3. **Visualization Phase** (unless `--skip_predict`)

   - Generates annotated frames (default: 800 frames, or `--pred_max_frames`)
   - Saves to `output_frames/`
   - Displays in window if `--show` flag used

### All Options

```bash
./run_all.sh [options]

Options:
  --dataset make|jungle      Dataset preset (default: make)
  --csv PATH                 Override CSV path
  --data_dir PATH            Override IMG directory path
  --run_name NAME            Artifact run folder name (default: timestamp_dataset)
  --model PATH               Model path (output if training; input if --skip_train)
  --python PATH              Python executable (default: auto-detect venv)
  --epochs N                 Training epochs (default: 20)
  --batch_size N             Training batch size (default: 32)
  --seq_len N                Sequence length (default: 3, recommended for best model)
  --zero_fraction F          Fraction of zero-angle samples kept (default: 0.1)
  --eval_max_frames N        Limit frames in evaluate.py (default: all)
  --pred_max_frames N        Limit frames in predict.py (default: 800)
  --show                     Show frames during prediction
  --skip_train               Skip training (requires --model existing)
  --skip_predict             Skip frame rendering step
  -h, --help                 Show this help
```

### Output Structure

```
artifacts/run_<timestamp>_<dataset>/
├── steering_model.keras           # Trained model
├── evaluation_metrics.json        # Aggregate metrics
├── frame_predictions.csv          # Per-frame predictions
└── output_frames/                 # Annotated images
    ├── frame_0000.jpg
    ├── frame_0001.jpg
    └── ...
```

---

## Architecture

### VGG-like CNN + LSTM (Production Model)

```
Input: (batch, seq_len, 66, 200, 3)
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
  └─ Dropout(0.2)
     Dense(128) → ReLU
     Dense(1)  → steering angle ∈ [−1, 1]
```

**Total parameters:** ~943K (3.6 MB)

### Hybrid CNN + LSTM (Fast Prototyping)

```
Input: (batch, seq_len, 66, 200, 3)
         │
TimeDistributed
  └─ CNN (Progressive kernels)
       RandomContrast(0.3)
       Rescaling(1/255)
       Conv2D 16×5×5, stride=2 → ReLU → BatchNorm
       Conv2D 32×5×5, stride=2 → ReLU → BatchNorm
       Conv2D 64×3×3 → ReLU → MaxPool → BatchNorm
       Conv2D 128×3×3 → ReLU → MaxPool → BatchNorm
       Flatten → Dropout(0.4)
       Dense(128) → ReLU
         │
LSTM(64)
  └─ Dropout(0.2)
     Dense(128) → ReLU
     Dense(1)  → steering angle ∈ [−1, 1]
```

**Total parameters:** ~952K (3.6 MB)

---

## Training Results

### Make Dataset (Recommended)

**VGG-like architecture (best accuracy):**

- Training samples: ~2000 (after balancing)
- Best validation MAE: **0.122** (~7.0° error)
- Full dataset MAE: **0.100** (~5.7° error)
- Training time: ~7s per epoch on M-series Mac
- Configuration: seq_len=3, Huber loss (delta=0.1), dropout=0.2

**Hybrid architecture (faster training):**

- Best validation MAE: **0.125** (~7.2° error)
- Full dataset MAE: **0.110** (~6.3° error)
- Training time: ~5s per epoch (30% faster)
- Configuration: seq_len=3, Huber loss (delta=0.1), dropout=0.2

### Jungle Dataset

**VGG-like architecture (best accuracy):**
- Training samples: ~4000 (after balancing)
- Best validation MAE: **0.174** (~10.0° error)
- Full dataset MAE: **0.209** (~12.0° error)
- Training time: ~14s per epoch on M-series Mac
- Configuration: seq_len=3, Huber loss (delta=0.1), dropout=0.2
- Correlation: **0.812** (excellent temporal modeling)
- R²: **0.642** (good predictive power)

**Hybrid architecture (faster training):**
- Best validation MAE: **0.172** (~9.9° error)
- Full dataset MAE: **0.238** (~13.6° error)
- Training time: ~11s per epoch (21% faster)
- Configuration: seq_len=3, Huber loss (delta=0.1), dropout=0.2
- Correlation: **0.788** (good temporal modeling)
- R²: **0.560** (decent predictive power)

### Model Comparison

| Model                | Val MAE   | Full MAE  | Improvement  | Speed      |
| -------------------- | --------- | --------- | ------------ | ---------- |
| Baseline (original)  | 0.289     | -         | -            | -          |
| Improved v2          | 0.259     | -         | +10.4%       | -          |
| Balanced             | 0.231     | -         | +20.1%       | -          |
| **VGG-like (Final)** | **0.122** | **0.100** | **+57.8%** ✅ | 7s/epoch   |
| Hybrid (5×5→3×3)     | 0.125     | 0.110     | +56.7%       | 5s/epoch ⚡ |

### When to Use Which Model

**VGG-like:**

- ✅ Production deployment
- ✅ Best accuracy (0.100 MAE)
- ✅ Best generalization
- ✅ Most stable results

**Hybrid:**

- ✅ Fast prototyping and iterations
- ✅ 30% faster training
- ✅ Similar performance (difference ~1°)
- ✅ Larger receptive field (5×5 kernels)

---

## Lane Change Detection

`LaneChangeDetector` uses a two-stage heuristic to detect lane changes:

### Stage 1: Candidate Start

- `abs(angle)` exceeds `threshold` (default: 0.2)
- Sustained for at least `min_hold_frames` (default: 5) consecutive frames
- Same steering direction (no sign changes)

### Stage 2: Confirmation

- Within `max_settle_frames` (default: 25), steering either:
  - Returns near straight (`abs(angle) <= settle_threshold`, default: 0.08)
  - Shows opposite-sign counter-steer

### Cooldown

- After event fires, suppresses repeated triggers for `cooldown_frames` (default: 20)

### Visual Warning

- Red border around frame
- "! LANE CHANGE !" banner at top
- Displayed for 30 frames after event

This approach reduces false positives on long constant curves compared to simple threshold detection.

---

## Testing

Run unit tests for lane-change detection:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover:

- Small angles (no trigger)
- Sustained angle with settling (trigger)
- Cooldown period (no repeated trigger)
- Constant curve (no trigger)
- Counter-steer confirmation (trigger)

---

## Simulation & Visualization Details

### How Simulation Works

1. **Model Loading**: Loads trained CNN+LSTM model (.keras file)
2. **Sequence Building**: Creates sliding window of `seq_len` frames (default: 3)
3. **Prediction**: Model predicts steering angle for each sequence
4. **Lane Detection**: Analyzes prediction history to detect lane changes
5. **Visualization**: Overlays predictions, GT (if available), and warnings on frames

### Visualization Components

**Info Panel :**

- Prediction: Current predicted steering angle
- Ground Truth: Actual angle from dataset (if available)
- Error: Absolute difference (if GT available)

**Steering Wheel):**

- Large circular wheel (radius 100px)
- Green needle: Predicted angle
- Yellow needle: Ground truth (if available)
- Rotates proportionally to angle (-1 to +1 maps to -90° to +90°)

**Lane Change Warning:**

- Red border (20px thick) around entire frame
- Red banner at top with "! LANE CHANGE !" text
- Triggered by two-stage lane change detector
- Persists for 30 frames after event

### Display Scaling

- All frames scaled 3× for better visibility
- Original frame: 200×66 pixels
- Display size: 600×198 pixels
- Press **Q** to quit visualization window

---

## Key Features

### Data Balancing & Augmentation

**The Problem:**
The original Udacity dataset is heavily imbalanced:

- **78% zero steering angles** (straight driving)
- **22% non-zero angles** (turns and lane changes)
  - Only 2.9% positive (right turns)
  - 19.2% negative (left turns)

This imbalance causes the model to:

- Overfit to straight driving
- Poorly predict turns and lane changes
- Have high bias toward zero predictions

**The Solution:**
Implemented in `train.py` via `balance_dataset()` function:

1. **Downsample zero angles**: Keep only 10% of zero-angle samples

   ```python
   zero_df = df[df["steering_angle"] == 0.0].sample(frac=0.1)
   ```
2. **Keep all non-zero angles**: Preserve all turning samples

   ```python
   non_zero_df = df[df["steering_angle"] != 0.0]
   ```
3. **Horizontal flip augmentation**: Mirror all non-zero samples

   ```python
   # For each non-zero sample, create flipped version
   flipped_row["centercam"] = row["centercam"] + "|FLIP"
   flipped_row["steering_angle"] = -row["steering_angle"]
   ```

   - Doubles non-zero samples
   - Balances left/right turn distribution
   - Handled automatically by `data_loader.py` during training

**The Result:**
Balanced dataset composition:

- **15% zero angles** (306 samples)
- **42.5% original non-zero** (869 samples)
- **42.5% flipped non-zero** (869 samples)
- **Total: 2044 samples** (from original 3930)

**Impact on Model Performance:**

- Baseline (imbalanced): 0.289 MAE
- Balanced dataset: 0.231 MAE → **20% improvement**
- Final model (balanced + other improvements): 0.100 MAE → **57.8% improvement**

### Loss Function

- **Huber loss** (delta=0.1) instead of MAE
- Better handling of large steering changes (lane changes, sharp turns)
- More robust to outliers than MSE

### Sequence Length

- Reduced from 5 to 3 frames for quicker reactions
- Balances temporal context with responsiveness
- Improves performance on sharp maneuvers

### Dropout Strategy

- CNN: 0.4 dropout after feature extraction
- LSTM: 0.2 dropout for faster response
- Prevents overfitting while maintaining reactivity

---

## Requirements

```
tensorflow>=2.12.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
```

Python ≥ 3.9 recommended.

---

## License

MIT License - See LICENSE file for details.

---

## Tips & Troubleshooting

### Training Tips

- Use `seq_len=3` for best results on make dataset
- Increase `--epochs` if validation loss still decreasing
- Reduce `--batch_size` if running out of memory
- Use `--zero_fraction 0.1` for balanced dataset

### Visualization Tips

- Use `--max_frames 200` for quick preview
- Use `--show` to see results in real-time
- Save frames with `--output_dir` for later review
- Press Q to quit visualization window

### Performance Tips

- Use hybrid model for faster training (30% speedup)
- Reduce `--eval_max_frames` for quick evaluation
- Skip prediction phase with `--skip_predict` if only need metrics

### Common Issues

- **"CSV not found"**: Check dataset path in `data/` directory
- **"Model file not found"**: Train model first or use `--skip_train` with existing model
- **Slow training**: Reduce batch size or use hybrid model
- **Poor accuracy**: Ensure `seq_len` matches trained model, check data balancing

---

## Citation

Dataset: [Udacity Self-Driving Car – Behavioural Cloning](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning)

---

## Contact

University Project (DUuSAiUV) - Autonomous Driving Course
