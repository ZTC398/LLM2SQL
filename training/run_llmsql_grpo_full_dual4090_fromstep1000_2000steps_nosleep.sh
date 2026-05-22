#!/usr/bin/env bash
# Continue from merged step1000 HF weights, disable vLLM sleep mode for stability.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"

export MODEL_PATH="${MODEL_PATH:-/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_full_3000steps_noval_step1000_hf}"
export DATA_DIR="${DATA_DIR:-/root/shared-nvme/rlvr/verl_data/llmsql_5shot}"
export VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_coder_3b_dual4090_fromstep1000_2000steps_nosleep}"
export LOGGER_BACKENDS="${LOGGER_BACKENDS:-[\"console\",\"tensorboard\"]}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/root/shared-nvme/rlvr/tensorboard/llmsql_grpo_full/${EXPERIMENT_NAME}}"

# Keep these aligned:
# - TRAIN_BATCH_SIZE should be divisible by PPO_MINI_BATCH_SIZE
# - TRAIN_BATCH_SIZE * ROLLOUT_N should be divisible by AGENT_NUM_WORKERS
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"

export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-1024}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2000}"
export SAVE_FREQ="${SAVE_FREQ:-500}"
export TEST_FREQ="${TEST_FREQ:--1}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/run_llmsql_grpo_full.sh" \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    "$@"
