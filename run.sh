#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
DEVICE="${DEVICE:-cuda}"
MODELS_DEFAULT="unet,unetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,proposal_hf_unet"
MODELS="${MODELS:-$MODELS_DEFAULT}"

usage() {
  cat <<EOF
Usage:
  bash run.sh install
  bash run.sh prepare [--source-dir PATH | --zip-path PATH | --download-url URL]
  bash run.sh splits
  bash run.sh train-one MODEL
  bash run.sh train-all
  bash run.sh eval-one MODEL [SPLIT]
  bash run.sh eval-all [SPLIT]
  bash run.sh benchmark
  bash run.sh export

Environment overrides:
  PYTHON_BIN   Python executable (default: python)
  CONFIG_DIR   Config directory (default: configs)
  DATA_ROOT    Data root (default: data)
  OUTPUT_ROOT  Output root (default: outputs)
  IMAGE_SIZE   Image size (default: 352)
  DEVICE       Device string (default: cuda)
  MODELS       Comma-separated model list for train-all/eval-all/benchmark
EOF
}

cmd_install() {
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
}

cmd_prepare() {
  "$PYTHON_BIN" scripts/prepare_kvasir_seg.py \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    "$@"
}

cmd_splits() {
  "$PYTHON_BIN" scripts/make_splits.py \
    --data-root "$DATA_ROOT"
}

cmd_train_one() {
  local model="${1:?Missing model name}"
  "$PYTHON_BIN" scripts/train_one.py \
    --model "$model" \
    --config "$CONFIG_DIR/$model.yaml" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_train_all() {
  "$PYTHON_BIN" scripts/train_all.py \
    --models "$MODELS" \
    --config-dir "$CONFIG_DIR" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_eval_one() {
  local model="${1:?Missing model name}"
  local split="${2:-test}"
  "$PYTHON_BIN" scripts/eval_one.py \
    --model "$model" \
    --config "$CONFIG_DIR/$model.yaml" \
    --split "$split" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_eval_all() {
  local split="${1:-test}"
  "$PYTHON_BIN" scripts/eval_all.py \
    --models "$MODELS" \
    --config-dir "$CONFIG_DIR" \
    --split "$split" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_benchmark() {
  "$PYTHON_BIN" scripts/benchmark_all.py \
    --models "$MODELS" \
    --config-dir "$CONFIG_DIR" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_export() {
  "$PYTHON_BIN" scripts/export_results.py --output-root "$OUTPUT_ROOT"
}

main() {
  local action="${1:-}"
  shift || true

  case "$action" in
    install) cmd_install "$@" ;;
    prepare) cmd_prepare "$@" ;;
    splits) cmd_splits "$@" ;;
    train-one) cmd_train_one "$@" ;;
    train-all) cmd_train_all "$@" ;;
    eval-one) cmd_eval_one "$@" ;;
    eval-all) cmd_eval_all "$@" ;;
    benchmark) cmd_benchmark "$@" ;;
    export) cmd_export "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Unknown command: $action" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
