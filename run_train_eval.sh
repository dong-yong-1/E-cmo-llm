#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "未找到 .env，请先参考 .env.example 配置 HF_TOKEN 和 API Key。"
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  PY_RUN=(".venv/bin/python")
else
  echo "未找到 uv，也未找到 .venv/bin/python，无法运行项目。"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_PATH="${DATA_PATH:-./ecommerce_sft_v1.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/sft_demo_run}"
REPORT_DIR="${REPORT_DIR:-./reports/default_run}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-cpu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEMO_SAMPLES="${DEMO_SAMPLES:-2}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-}"

CANDIDATE_NAME="${CANDIDATE_NAME:-$(basename "$OUTPUT_DIR")}"
BASELINE_NAME="${BASELINE_NAME:-base_model}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-100}"

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "训练数据不存在: $DATA_PATH"
  exit 1
fi

if [[ ! -f "$EVAL_DATA_PATH" ]]; then
  echo "评测数据不存在: $EVAL_DATA_PATH"
  exit 1
fi

echo "========== 训练开始 =========="
printf '工作目录: %s\n' "$ROOT_DIR"
printf '训练数据: %s\n' "$DATA_PATH"
printf '评测数据: %s\n' "$EVAL_DATA_PATH"
printf '输出目录: %s\n' "$OUTPUT_DIR"
printf '报告目录: %s\n' "$REPORT_DIR"

"${PY_RUN[@]}" scripts/train_sft.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length "$MAX_LENGTH" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --logging-steps "$LOGGING_STEPS" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --weight-decay "$WEIGHT_DECAY" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --target-modules "$TARGET_MODULES" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --demo-samples "$DEMO_SAMPLES" \
  --seed "$SEED" \
  --report-to "$REPORT_TO" \
  --run-name "$RUN_NAME"

if [[ ! -f "$OUTPUT_DIR/adapter_model.safetensors" ]]; then
  echo "训练已完成，但未在 $OUTPUT_DIR 下找到 adapter_model.safetensors"
  exit 1
fi

echo "========== 评估开始 =========="

"${PY_RUN[@]}" scripts/evaluate.py \
  --eval-data-path "$EVAL_DATA_PATH" \
  --candidate-model-name "$MODEL_NAME" \
  --candidate-adapter-path "$OUTPUT_DIR" \
  --candidate-name "$CANDIDATE_NAME" \
  --baseline-model-name "$MODEL_NAME" \
  --baseline-name "$BASELINE_NAME" \
  --judge-model "$JUDGE_MODEL" \
  --output-report "$REPORT_DIR/eval_report.json" \
  --output-details "$REPORT_DIR/eval_details.json" \
  --max-samples "$MAX_EVAL_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --seed "$SEED"

echo "========== 流程完成 =========="
printf '训练产物: %s\n' "$OUTPUT_DIR"
printf '评估报告: %s\n' "$REPORT_DIR/eval_report.json"
printf '评估明细: %s\n' "$REPORT_DIR/eval_details.json"
