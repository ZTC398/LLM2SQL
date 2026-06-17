# LLMSQL-RL

Reinforcement learning experiments for `LLMSQL` text-to-SQL generation with `verl + GRPO + Qwen2.5-Coder-3B-Instruct`.

## Experiment

This project asks a simple question: can execution-based RL improve text-to-SQL performance on `LLMSQL`?

The repository includes:

- a single-turn GRPO training pipeline
- execution-based reward functions
- offline evaluation scripts
- an experimental `tool-only` agentic RL branch

## Tech Stack

`Python` · `PyTorch` · `transformers` · `vLLM` · `verl` · `ray` · `datasets` · `SQLite`

## Results

Primary metric: execution-match accuracy.

| Setting | Accuracy |
| --- | --- |
| `Qwen2.5-Coder-3B-Instruct` base | `0.7471` |
| best single-turn RL checkpoint | `0.8349` |
| local `Qwen3-Coder-30B-A3B-Instruct` reference | `0.8523` |

The single-turn RL pipeline is the main result of this repository.  
The agentic branch is kept as an experiment, not as the recommended setup.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Set paths:

```bash
export LLMSQL_DATASET_DIR=/path/to/llmsql-2.0
export LLMSQL_DB_PATH=$LLMSQL_DATASET_DIR/sqlite_tables.db
export LLMSQL_OUTPUT_ROOT=/path/to/outputs
export MODEL_PATH=/path/to/Qwen2.5-Coder-3B-Instruct
```

Prepare training data:

```bash
python data_prep/prepare_llmsql_verl_data.py \
  --dataset-dir "$LLMSQL_DATASET_DIR" \
  --output-dir /path/to/verl_data/llmsql_5shot \
  --num-fewshots 5
```

Run baseline:

```bash
python scripts/run_llmsql_baseline.py \
  --model-path "$MODEL_PATH" \
  --questions-path "$LLMSQL_DATASET_DIR/test_questions.jsonl" \
  --tables-path "$LLMSQL_DATASET_DIR/tables.jsonl" \
  --db-path "$LLMSQL_DB_PATH" \
  --output-dir "$LLMSQL_OUTPUT_ROOT/baseline_qwen25_coder_3b"
```

Run training:

```bash
bash training/run_llmsql_grpo_full.sh
```

Evaluate predictions:

```bash
python scripts/eval_llmsql_predictions.py \
  --predictions-path /path/to/preds.jsonl \
  --questions-path "$LLMSQL_DATASET_DIR/test_questions.jsonl" \
  --db-path "$LLMSQL_DB_PATH" \
  --output-dir "$LLMSQL_OUTPUT_ROOT/offline_eval" \
  --label run_name
```

## Repository Layout

- `data_prep/` prepares parquet data for `verl`
- `rewards/` contains single-turn and agentic reward functions
- `scripts/` contains baseline, eval, and utility scripts
- `training/` contains GRPO launchers
- `llmsql_agentic/` contains the experimental SQL agent loop
- `configs/agent_loops/` contains agent loop config
- `third_party/verl/` vendors `verl`

## Agentic Branch

The repository also contains an experimental `tool-only` SQL agent:

- `data_prep/prepare_llmsql_agent_data.py`
- `llmsql_agentic/agent_loop.py`
- `llmsql_agentic/tool_env.py`
- `rewards/llmsql_agent_reward.py`
- `training/run_llmsql_grpo_agentic.sh`

It worked as an engineering prototype, but it was more memory-hungry and less stable than the single-turn mainline.
