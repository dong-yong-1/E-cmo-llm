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
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-./models/final_v4_clean_sft_r16_qkvo}"
DPO_DATA_PATH="${DPO_DATA_PATH:-./data/dpo/small_dpo_pairs_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/small_dpo_v1}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-768}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
BETA="${BETA:-0.1}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-42}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-small_dpo_v1}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$SFT_ADAPTER_PATH" ]]; then
  echo "SFT adapter 不存在: $SFT_ADAPTER_PATH"
  exit 1
fi

if [[ ! -f "$DPO_DATA_PATH" ]]; then
  echo "DPO 数据不存在: $DPO_DATA_PATH"
  exit 1
fi

echo "========== 小规模 DPO 训练开始 =========="
printf '工作目录: %s\n' "$ROOT_DIR"
printf 'Python: %s\n' "${PY_RUN[*]}"
printf '基础模型: %s\n' "$MODEL_NAME"
printf 'SFT adapter: %s\n' "$SFT_ADAPTER_PATH"
printf 'DPO 数据: %s\n' "$DPO_DATA_PATH"
printf '输出目录: %s\n' "$OUTPUT_DIR"
printf '参数: batch=%s epochs=%s lr=%s beta=%s\n' "$BATCH_SIZE" "$EPOCHS" "$LEARNING_RATE" "$BETA"

"${PY_RUN[@]}" scripts/train_dpo.py \
  --model-name "$MODEL_NAME" \
  --sft-adapter-path "$SFT_ADAPTER_PATH" \
  --data-path "$DPO_DATA_PATH" \
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
  --beta "$BETA" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --seed "$SEED" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --report-to "$REPORT_TO" \
  --run-name "$RUN_NAME"

echo "========== 小规模 DPO 训练完成 =========="
printf '模型输出: %s\n' "$OUTPUT_DIR"
