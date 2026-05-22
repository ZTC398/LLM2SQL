# rl_project

`LLMSQL` 上的 text-to-SQL RL 项目，当前使用 `verl + GRPO + Qwen2.5-Coder-3B-Instruct`。

## Current Status

当前这条线已经跑通：

- 基模官方 baseline
- `LLMSQL -> parquet` 数据整理
- `verl` 训练脚本
- 自定义 SQL reward
- checkpoint 合并到 Hugging Face 格式
- 官方 evaluate 对齐评测

当前最好结果：

- 基模：`0.7471`
- best model：`0.8349`
- best checkpoint：`/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep_step1500_hf`

详细实验记录见：

- [outputs/2026-05-22/llmsql_verl_experiment_report.md](/root/rl_project/outputs/2026-05-22/llmsql_verl_experiment_report.md)

## Layout

- `data_prep/`: LLMSQL 数据转换脚本
- `rewards/`: 自定义 reward
- `scripts/`: baseline、评测、数据检查、reward smoke test
- `training/`: `verl` 训练启动脚本
- `third_party/verl/`: 本地 vendor 的 `verl` 源码

## Environment

- code root: `/root/rl_project`
- data/model/output root: `/root/shared-nvme/rlvr`
- conda env: `rlvr`
- base model: `/root/shared-nvme/rlvr/models/Qwen2.5-Coder-3B-Instruct`
- dataset: `https://huggingface.co/datasets/llmsql-bench/llmsql-2.0`

## Data

官方数据主要使用：

- questions: `/root/shared-nvme/rlvr/datasets/llmsql-2.0/test_questions.jsonl`
- tables: `/root/shared-nvme/rlvr/datasets/llmsql-2.0/tables.jsonl`
- sqlite db: `/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db`

训练数据当前使用整理后的 5-shot parquet：

- train: `/root/shared-nvme/rlvr/verl_data/llmsql_5shot/train.parquet`
- test: `/root/shared-nvme/rlvr/verl_data/llmsql_5shot/test.parquet`

构建命令：

```bash
conda run -n rlvr python /root/rl_project/data_prep/prepare_llmsql_verl_data.py
```

## Quick Start

1. 跑基模 baseline

```bash
conda run -n rlvr python /root/rl_project/scripts/run_llmsql_baseline.py \
  --model-path /root/shared-nvme/rlvr/models/Qwen2.5-Coder-3B-Instruct \
  --output-dir /root/shared-nvme/rlvr/outputs/baseline_qwen25_coder_3b_test
```

2. 跑 500-step GRPO

```bash
conda run -n rlvr bash /root/rl_project/training/run_llmsql_grpo_full_dual4090_500steps_noval.sh
```

3. 跑更长训练

```bash
conda run -n rlvr bash /root/rl_project/training/run_llmsql_grpo_full_dual4090_3000steps_noval.sh
```

4. 查看 parquet 结构

```bash
conda run -n rlvr python /root/rl_project/scripts/inspect_llmsql_parquet.py
```

## Important Notes

- 当前 `step1500` 最佳结果不是从零连续训练得到的。
- 它是先拿到 `step1000` 的 HF 权重，再新开一轮训练得到的“权重继续训练”结果，不是真正的 optimizer state resume。
- 当前训练的主要问题不是 reward 无效，而是 `verl + vLLM` 在长时间训练下的 `sleep/cumem` 稳定性。
- 为避免 checkpoint 存盘报错，actor checkpoint 当前只保存 `model`，不保存 optimizer。

## Main Scripts

- baseline eval:
  - `/root/rl_project/scripts/run_llmsql_baseline.py`
- reward:
  - `/root/rl_project/rewards/llmsql_sql_reward.py`
- main launcher:
  - `/root/rl_project/training/run_llmsql_grpo_full.sh`
- 500-step run:
  - `/root/rl_project/training/run_llmsql_grpo_full_dual4090_500steps_noval.sh`
- 3000-step run:
  - `/root/rl_project/training/run_llmsql_grpo_full_dual4090_3000steps_noval.sh`
- continue-from-step1000 run:
  - `/root/rl_project/training/run_llmsql_grpo_full_dual4090_fromstep1000_2000steps_sleep.sh`
