#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
One-command pipeline: train -> evaluate -> (optional) annotated frames.

Usage:
  ./run_all.sh [options]

Options:
  --dataset make|jungle      Dataset preset (default: make)
  --csv PATH                 Override CSV path
  --data_dir PATH            Override IMG directory path
  --run_name NAME            Artifact run folder name (default: timestamp_dataset)
  --model PATH               Model path (output if training; input if --skip_train)
  --python PATH              Python executable (default: .venv/bin/python if present)
  --epochs N                 Training epochs (default: 20)
  --batch_size N             Training batch size (default: 32)
  --seq_len N                Sequence length (default: 3)
  --zero_fraction F          Fraction of zero-angle samples kept (default: 0.1)
  --eval_max_frames N        Limit frames in evaluate.py (default: all)
  --pred_max_frames N        Limit frames in predict.py (default: 800)
  --show                     Show frames during prediction
  --skip_train               Skip training (requires --model existing)
  --skip_predict             Skip frame rendering step
  -h, --help                Show this help
EOF
}

DATASET="make"
CSV_PATH=""
DATA_DIR=""
RUN_NAME=""
MODEL_PATH=""
PYTHON_BIN=""
EPOCHS=20
BATCH_SIZE=32
SEQ_LEN=3
ZERO_FRACTION=0.1
EVAL_MAX_FRAMES=""
PRED_MAX_FRAMES=800
SHOW=0
SKIP_TRAIN=0
SKIP_PREDICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --csv) CSV_PATH="${2:-}"; shift 2 ;;
    --data_dir) DATA_DIR="${2:-}"; shift 2 ;;
    --run_name) RUN_NAME="${2:-}"; shift 2 ;;
    --model) MODEL_PATH="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --batch_size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --seq_len) SEQ_LEN="${2:-}"; shift 2 ;;
    --zero_fraction) ZERO_FRACTION="${2:-}"; shift 2 ;;
    --eval_max_frames) EVAL_MAX_FRAMES="${2:-}"; shift 2 ;;
    --pred_max_frames) PRED_MAX_FRAMES="${2:-}"; shift 2 ;;
    --show) SHOW=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_predict) SKIP_PREDICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CSV_PATH" || -z "$DATA_DIR" ]]; then
  case "$DATASET" in
    make)
      CSV_PATH_DEFAULT="data/self_driving_car_dataset_make/driving_log.csv"
      DATA_DIR_DEFAULT="data/self_driving_car_dataset_make/IMG"
      ;;
    jungle)
      CSV_PATH_DEFAULT="data/self_driving_car_dataset_jungle/driving_log.csv"
      DATA_DIR_DEFAULT="data/self_driving_car_dataset_jungle/IMG"
      ;;
    *)
      echo "Unsupported --dataset '$DATASET'. Use make or jungle." >&2
      exit 1
      ;;
  esac
  CSV_PATH="${CSV_PATH:-$CSV_PATH_DEFAULT}"
  DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif [[ -x "venv/bin/python" ]]; then
    PYTHON_BIN="venv/bin/python"
  elif [[ -x "../venv/bin/python" ]]; then
    PYTHON_BIN="../venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found: $CSV_PATH" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Image directory not found: $DATA_DIR" >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(date +%Y%m%d_%H%M%S)_${DATASET}"
fi

RUN_DIR="artifacts/run_${RUN_NAME}"
mkdir -p "$RUN_DIR"

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="${RUN_DIR}/steering_model.keras"
fi

METRICS_JSON="${RUN_DIR}/evaluation_metrics.json"
PRED_CSV="${RUN_DIR}/frame_predictions.csv"
FRAMES_DIR="${RUN_DIR}/output_frames"

run_cmd() {
  echo "+ $*"
  "$@"
}

echo "============================================================"
echo "DUuSAiUV pipeline"
echo "Python:        $PYTHON_BIN"
echo "Dataset:       $DATASET"
echo "CSV:           $CSV_PATH"
echo "Data dir:      $DATA_DIR"
echo "Run dir:       $RUN_DIR"
echo "Model:         $MODEL_PATH"
echo "Sequence len:  $SEQ_LEN"
echo "============================================================"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  run_cmd "$PYTHON_BIN" train.py \
    --csv "$CSV_PATH" \
    --data_dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --zero_fraction "$ZERO_FRACTION"
else
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model file not found for --skip_train: $MODEL_PATH" >&2
    exit 1
  fi
fi

EVAL_ARGS=(
  --model "$MODEL_PATH"
  --csv "$CSV_PATH"
  --data_dir "$DATA_DIR"
  --seq_len "$SEQ_LEN"
  --output_json "$METRICS_JSON"
  --output_csv "$PRED_CSV"
)
if [[ -n "$EVAL_MAX_FRAMES" ]]; then
  EVAL_ARGS+=(--max_frames "$EVAL_MAX_FRAMES")
fi
run_cmd "$PYTHON_BIN" evaluate.py "${EVAL_ARGS[@]}"

if [[ "$SKIP_PREDICT" -eq 0 ]]; then
  PRED_ARGS=(
    --model "$MODEL_PATH"
    --csv "$CSV_PATH"
    --data_dir "$DATA_DIR"
    --seq_len "$SEQ_LEN"
    --output_dir "$FRAMES_DIR"
  )
  if [[ -n "$PRED_MAX_FRAMES" ]]; then
    PRED_ARGS+=(--max_frames "$PRED_MAX_FRAMES")
  fi
  if [[ "$SHOW" -eq 1 ]]; then
    PRED_ARGS+=(--show)
  fi
  run_cmd "$PYTHON_BIN" predict.py "${PRED_ARGS[@]}"
fi

echo "Done."
echo "Metrics JSON: $METRICS_JSON"
echo "Per-frame CSV: $PRED_CSV"
if [[ "$SKIP_PREDICT" -eq 0 ]]; then
  echo "Annotated frames: $FRAMES_DIR"
fi
