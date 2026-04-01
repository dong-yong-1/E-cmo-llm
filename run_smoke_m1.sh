#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "未找到 .env，请先配置 HF_TOKEN / DEEPSEEK_API_KEY。"
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  PY_RUN=(".venv/bin/python")
else
  echo "未找到 uv，也未找到 .venv/bin/python。"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_PATH="${DATA_PATH:-./ecommerce_sft_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/smoke_m1}"
REPORT_DIR="${REPORT_DIR:-./reports/smoke_m1}"

# M1 8G 友好默认配置
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
DEVICE_MAP="${DEVICE_MAP:-cpu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

echo "========== M1 小实验：训练 =========="
"${PY_RUN[@]}" scripts/train_sft.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length "$MAX_LENGTH" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --demo-samples 1 \
  --seed "$SEED" \
  --max-train-samples "$MAX_TRAIN_SAMPLES"

echo "========== M1 小实验：评估（仅规则准确率） =========="
"${PY_RUN[@]}" scripts/evaluate.py \
  --eval-data-path "$DATA_PATH" \
  --candidate-model-name "$MODEL_NAME" \
  --candidate-adapter-path "$OUTPUT_DIR" \
  --candidate-name "smoke_m1" \
  --judge-model "deepseek-chat" \
  --output-report "$REPORT_DIR/eval_report.json" \
  --output-details "$REPORT_DIR/eval_details.json" \
  --max-samples "$MAX_EVAL_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --seed "$SEED" \
  --skip-pairwise

echo "========== 小实验完成 =========="
echo "训练输出: $OUTPUT_DIR"
echo "评估报告: $REPORT_DIR/eval_report.json"
