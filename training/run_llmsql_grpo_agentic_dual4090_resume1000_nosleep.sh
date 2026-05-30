#!/usr/bin/env bash
# Resume LLMSQL agentic GRPO from the saved step-1000 checkpoint on 2x4090.
# Disable vLLM sleep mode because long runs have shown EngineCore crashes in
# the sleep/unmap path (`cumem.py -> gpu_worker.py: sleep`).

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"

export DATA_DIR="${DATA_DIR:-/root/shared-nvme/rlvr/verl_data/llmsql_agent_0shot}"
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-/root/shared-nvme/rlvr/outputs/verl/llmsql_grpo_agentic/qwen25_coder_3b_agentic_dual4090_3000steps_noval/global_step_1000}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_coder_3b_agentic_dual4090_resume1000_nosleep}"
export LOGGER_BACKENDS="${LOGGER_BACKENDS:-[\"console\",\"tensorboard\"]}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/root/shared-nvme/rlvr/tensorboard/llmsql_grpo_agentic/${EXPERIMENT_NAME}}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-2}"

export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-2048}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
export ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-false}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3000}"
export SAVE_FREQ="${SAVE_FREQ:-500}"
export TEST_FREQ="${TEST_FREQ:--1}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/run_llmsql_grpo_agentic.sh" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    "$@"
