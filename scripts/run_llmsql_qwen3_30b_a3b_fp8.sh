#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/root/shared-nvme/rlvr/models/Qwen3-Coder-30B-A3B-Instruct-FP8}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/shared-nvme/rlvr/evals/qwen3_coder_30b_a3b_fp8_test}"
export TP_SIZE="${TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_FEWSHOTS="${NUM_FEWSHOTS:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export LIMIT="${LIMIT:-}"
export SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  /root/rl_project/scripts/run_llmsql_baseline.py
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --tensor-parallel-size "${TP_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-fewshots "${NUM_FEWSHOTS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "Running LLMSQL baseline with:"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_FEWSHOTS=${NUM_FEWSHOTS}"
echo "  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "  TEMPERATURE=${TEMPERATURE}"
echo "  LIMIT=${LIMIT:-full}"

conda run -n rlvr python "${ARGS[@]}"
