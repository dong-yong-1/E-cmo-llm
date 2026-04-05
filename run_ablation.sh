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
DATA_PATH="${DATA_PATH:-./ecommerce_sft_v1.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_PATH}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./models/ablation}"
BASE_REPORT_DIR="${BASE_REPORT_DIR:-./reports/ablation}"

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
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEMO_SAMPLES="${DEMO_SAMPLES:-1}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-wandb}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

printf 'Python: %s\n' "${PY_RUN[*]}"

run_one() {
  local run_name="$1"
  local lora_r="$2"
  local lora_alpha="$3"
  local target_modules="$4"
  local output_dir="${BASE_OUTPUT_DIR}/${run_name}"
  local report_dir="${BASE_REPORT_DIR}/${run_name}"

  mkdir -p "$output_dir" "$report_dir"

  echo "========== 开始实验: ${run_name} =========="

  "${PY_RUN[@]}" scripts/train_sft.py \
    --model-name "$MODEL_NAME" \
    --data-path "$DATA_PATH" \
    --output-dir "$output_dir" \
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
    --lora-r "$lora_r" \
    --lora-alpha "$lora_alpha" \
    --target-modules "$target_modules" \
    --torch-dtype "$TORCH_DTYPE" \
    --device-map "$DEVICE_MAP" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --demo-samples "$DEMO_SAMPLES" \
    --seed "$SEED" \
    --report-to "$REPORT_TO" \
    --run-name "$run_name"

  "${PY_RUN[@]}" scripts/evaluate.py \
    --eval-data-path "$EVAL_DATA_PATH" \
    --candidate-model-name "$MODEL_NAME" \
    --candidate-adapter-path "$output_dir" \
    --candidate-name "$run_name" \
    --baseline-model-name "$MODEL_NAME" \
    --baseline-name "base_model" \
    --judge-model "$JUDGE_MODEL" \
    --output-report "$report_dir/eval_report.json" \
    --output-details "$report_dir/eval_details.json" \
    --max-samples "$MAX_EVAL_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --torch-dtype "$TORCH_DTYPE" \
    --device-map "$DEVICE_MAP" \
    --seed "$SEED"

  echo "========== 实验完成: ${run_name} =========="
}

run_one "lora_r8_qv" 8 32 "q_proj,v_proj"
run_one "lora_r16_qv" 16 32 "q_proj,v_proj"
run_one "lora_r16_qkvo" 16 32 "q_proj,k_proj,v_proj,o_proj"

echo "全部消融实验完成。"
