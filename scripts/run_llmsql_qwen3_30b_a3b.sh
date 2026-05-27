#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/root/shared-nvme/rlvr/models/Qwen3-Coder-30B-A3B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/shared-nvme/rlvr/evals/qwen3_coder_30b_a3b_full}"
export TP_SIZE="${TP_SIZE:-4}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_FEWSHOTS="${NUM_FEWSHOTS:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export RESUME="${RESUME:-1}"
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
echo "  ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  RESUME=${RESUME}"
echo "  LIMIT=${LIMIT:-full}"

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

conda run -n rlvr python "${ARGS[@]}"
