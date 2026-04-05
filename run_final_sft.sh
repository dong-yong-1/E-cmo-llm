#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "未找到 .env，请先配置 HF_TOKEN / DEEPSEEK_API_KEY。"
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_RUN=("$PYTHON_BIN")
elif command -v python >/dev/null 2>&1; then
  PY_RUN=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY_RUN=(python3)
elif [[ -x ".venv/bin/python" ]]; then
  PY_RUN=(".venv/bin/python")
else
  echo "未找到可用的 Python 解释器。可通过 PYTHON_BIN 显式指定。"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./data/final/train.json}"
VAL_DATA_PATH="${VAL_DATA_PATH:-./data/final/val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/final_sft_r16_qkvo}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEMO_SAMPLES="${DEMO_SAMPLES:-2}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-final_sft_r16_qkvo}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
  echo "训练数据不存在: $TRAIN_DATA_PATH"
  exit 1
fi

if [[ ! -f "$VAL_DATA_PATH" ]]; then
  echo "提示: 当前训练脚本尚未接入独立 val 评估，但已检测到 val 路径: $VAL_DATA_PATH"
fi

echo "========== 正式 SFT 训练开始 =========="
printf '工作目录: %s\n' "$ROOT_DIR"
printf 'Python: %s\n' "${PY_RUN[*]}"
printf '模型: %s\n' "$MODEL_NAME"
printf '训练集: %s\n' "$TRAIN_DATA_PATH"
printf '验证集: %s\n' "$VAL_DATA_PATH"
printf '输出目录: %s\n' "$OUTPUT_DIR"
printf 'LoRA 配置: r=%s alpha=%s target=%s\n' "$LORA_R" "$LORA_ALPHA" "$TARGET_MODULES"

"${PY_RUN[@]}" scripts/train_sft.py \
  --model-name "$MODEL_NAME" \
  --data-path "$TRAIN_DATA_PATH" \
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

echo "========== 正式 SFT 训练完成 =========="
printf '模型输出: %s\n' "$OUTPUT_DIR"
