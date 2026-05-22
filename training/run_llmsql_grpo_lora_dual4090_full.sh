#!/usr/bin/env bash
# Conservative 2x4090 full-data launch for LLMSQL GRPO + LoRA on verl.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"

export DATA_DIR="${DATA_DIR:-/root/shared-nvme/rlvr/verl_data/llmsql_5shot}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_coder_3b_dual4090_full}"

# Keep these aligned:
# - TRAIN_BATCH_SIZE should be divisible by PPO_MINI_BATCH_SIZE
# - TRAIN_BATCH_SIZE * ROLLOUT_N should be divisible by AGENT_NUM_WORKERS
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"

export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-1024}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-200}"
export TEST_FREQ="${TEST_FREQ:-200}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/run_llmsql_grpo_lora.sh" "$@"
