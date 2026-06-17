# LLMSQL-GRPO

Reinforcement learning experiments for text-to-SQL generation on the `LLMSQL 2.0` benchmark.

## Objective

This repository studies whether execution-based policy optimization improves execution-match accuracy relative to the base model on `LLMSQL 2.0`.

The primary configuration is single-turn `GRPO` training with `verl` and `vLLM` on `Qwen2.5-Coder-3B-Instruct`. An experimental agentic branch is retained for completeness but is not the recommended reproduction path.

## Method

- Base model: `Qwen2.5-Coder-3B-Instruct`
- Training framework: `verl`
- Optimization algorithm: `GRPO`
- Inference backend: `vLLM`
- Reward type: execution-based rule reward
- Database engine: `SQLite`
- Dataset: `LLMSQL 2.0`

## Results

Primary metric: execution-match accuracy.

| Setting | Accuracy |
| --- | ---: |
| Base model | `0.7471` |
| Best single-turn RL checkpoint | `0.8349` |
| Local `Qwen3-Coder-30B-A3B-Instruct` reference | `0.8523` |

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export LLMSQL_DATASET_DIR=/path/to/llmsql-2.0
export LLMSQL_DB_PATH=$LLMSQL_DATASET_DIR/sqlite_tables.db
export LLMSQL_OUTPUT_ROOT=/path/to/outputs
export MODEL_PATH=/path/to/Qwen2.5-Coder-3B-Instruct
```

Prepare parquet data for `verl`:

```bash
python data_prep/prepare_llmsql_verl_data.py \
  --dataset-dir "$LLMSQL_DATASET_DIR" \
  --output-dir /path/to/verl_data/llmsql_5shot \
  --num-fewshots 5
```

Run base-model evaluation:

```bash
python scripts/run_llmsql_baseline.py \
  --model-path "$MODEL_PATH" \
  --questions-path "$LLMSQL_DATASET_DIR/test_questions.jsonl" \
  --tables-path "$LLMSQL_DATASET_DIR/tables.jsonl" \
  --db-path "$LLMSQL_DB_PATH" \
  --output-dir "$LLMSQL_OUTPUT_ROOT/baseline_qwen25_coder_3b"
```

Run single-turn GRPO training:

```bash
bash training/run_llmsql_grpo_full.sh
```

Evaluate generated SQL:

```bash
python scripts/eval_llmsql_predictions.py \
  --predictions-path /path/to/preds.jsonl \
  --questions-path "$LLMSQL_DATASET_DIR/test_questions.jsonl" \
  --db-path "$LLMSQL_DB_PATH" \
  --output-dir "$LLMSQL_OUTPUT_ROOT/offline_eval" \
  --label run_name
```

## Configuration

The default training launcher is [training/run_llmsql_grpo_full.sh](training/run_llmsql_grpo_full.sh). The main configuration is:

- `TRAIN_BATCH_SIZE=4`
- `PPO_MINI_BATCH_SIZE=8`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=192`
- `PPO_MAX_TOKEN_LEN_PER_GPU=4096`
- `ROLLOUT_N=4`
- `ROLLOUT_GPU_MEM_UTIL=0.35`
- `ACTOR_LR=5e-6`
- `KL_LOSS_COEF=0.001`
- `TOTAL_EPOCHS=1`

## Repository Layout

- `data_prep/`: dataset conversion for `verl`
- `rewards/`: execution-based reward functions
- `scripts/`: baseline inference and offline evaluation
- `training/`: training launch scripts
- `llmsql_agentic/`: experimental multi-turn SQL agent components
- `configs/agent_loops/`: agent loop configuration
- `third_party/verl/`: vendored `verl` source

## References

- `LLMSQL` benchmark repository: <https://github.com/LLMSQL/llmsql-benchmark>
- `LLMSQL 2.0` dataset: <https://huggingface.co/datasets/llmsql-bench/llmsql-2.0>
- `verl` repository: <https://github.com/volcengine/verl>

This repository uses the `llmsql` benchmark package and includes a vendored copy of `verl` under `third_party/verl`. Upstream projects retain their respective licenses. See `THIRD_PARTY_NOTICES.md` for details.

## License

This repository is released under the Apache License 2.0. See `LICENSE`.
