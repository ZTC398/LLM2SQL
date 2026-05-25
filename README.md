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

最新离线评测总表见：

- [outputs/2026-05-25/llmsql_weight_progression_table.md](/root/rl_project/outputs/2026-05-25/llmsql_weight_progression_table.md)

详细实验记录见：

- [outputs/2026-05-22/llmsql_verl_experiment_report.md](/root/rl_project/outputs/2026-05-22/llmsql_verl_experiment_report.md)

## Offline Eval Snapshot

当前统一使用执行结果口径，而不是 exact SQL string match。

| Weight | first_exec_match | first_sql_exec | sample_sql_errors | candidate_sql_errors | agg_acc | where_acc |
| --- | --- | --- | --- | --- | --- | --- |
| `base` | `0.7471` | `0.9963` | `58` | `106` | `0.6976` | `0.7464` |
| `full_500` | `0.8048` | `0.9944` | `89` | `105` | `0.7702` | `0.8038` |
| `full_1000` | `0.8080` | `0.9970` | `47` | `58` | `0.6825` | `0.8074` |
| `effective_1500` | `0.8161` | `0.9974` | `41` | `52` | `0.7305` | `0.8153` |
| `effective_2000` | `0.8087` | `0.9974` | `41` | `60` | `0.6833` | `0.8080` |
| `effective_2500` | `0.8349` | `0.9979` | `34` | `50` | `0.7716` | `0.8344` |

说明：

- `1500/2000/2500` 这里是 effective total steps，不是一次 uninterrupted optimizer resume。
- 映射关系是：
  - `effective_1500` = 从 `step1000` HF 权重继续训练那条线的 `global_step_500`
  - `effective_2000` = 同一条线的 `global_step_1000`
  - `effective_2500` = 同一条线的 `global_step_1500`
- 当前最靠谱的主指标是 `first_exec_match`。
- `exact_sql_match_rate` 已移除，不再作为主指标。

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

5. 跑 richer offline eval

```bash
conda run -n rlvr python /root/rl_project/scripts/eval_llmsql_predictions.py \
  --predictions-path /root/shared-nvme/rlvr/evals/qwen25_coder_3b_dual4090_full_500steps_noval_hf/preds_5shot_full.jsonl \
  --output-dir /root/shared-nvme/rlvr/analysis/llmsql_offline_eval_v2/qwen25_coder_3b_dual4090_full_500steps_noval_hf \
  --label full_500steps
```

6. 汇总多份 offline eval

```bash
conda run -n rlvr python /root/rl_project/scripts/summarize_llmsql_offline_eval.py \
  --summary-json \
    /root/shared-nvme/rlvr/analysis/llmsql_offline_eval_v2/baseline_qwen25_coder_3b_test/summary.json \
    /root/shared-nvme/rlvr/analysis/llmsql_offline_eval_v2/qwen25_coder_3b_dual4090_full_500steps_noval_hf/summary.json \
    /root/shared-nvme/rlvr/analysis/llmsql_offline_eval_v2/qwen25_coder_3b_dual4090_full_3000steps_noval_step1000_hf/summary.json \
  --output-path /root/rl_project/outputs/2026-05-25/llmsql_weight_progression_table.md
```

## OpenRouter Eval

为了探索任务上限，现在支持直接调用 OpenRouter 上的大模型 API 跑 `LLMSQL` test。

脚本：

- `/root/rl_project/scripts/run_llmsql_openrouter.py`

特点：

- 复用官方 `LLMSQL` 0/1/5-shot prompt
- 输出格式和本地 baseline 一致，可直接接官方 evaluate 和本仓库 richer offline eval
- 支持 `resume`
- 支持 `429/5xx` 重试和指数退避
- 支持低速限流，适合 free model

先设置 key：

```bash
export OPENROUTER_API_KEY='your_key_here'
```

小样本 smoke test：

```bash
conda run -n rlvr python /root/rl_project/scripts/run_llmsql_openrouter.py \
  --model 'openrouter/owl-alpha' \
  --output-dir /root/shared-nvme/rlvr/evals/openrouter_owl_alpha_smoke_limit1 \
  --limit 1 \
  --max-concurrency 1 \
  --min-request-interval 2 \
  --max-retries 6 \
  --retry-base-delay 3 \
  --temperature 0.0
```

全量 test：

```bash
PYTHONUNBUFFERED=1 conda run -n rlvr python /root/rl_project/scripts/run_llmsql_openrouter.py \
  --model 'openrouter/owl-alpha' \
  --output-dir /root/shared-nvme/rlvr/evals/openrouter_owl_alpha_full \
  --max-concurrency 1 \
  --min-request-interval 2 \
  --max-retries 12 \
  --retry-base-delay 3 \
  --temperature 0.0 \
  --resume
```

跑完后接 richer offline eval：

```bash
conda run -n rlvr python /root/rl_project/scripts/eval_llmsql_predictions.py \
  --predictions-path /root/shared-nvme/rlvr/evals/openrouter_owl_alpha_full/preds_5shot_full.jsonl \
  --output-dir /root/shared-nvme/rlvr/analysis/llmsql_offline_eval_v2/openrouter_owl_alpha_full \
  --label openrouter_owl_alpha_full
```

监控预测进度：

```bash
watch -n 5 "wc -l /root/shared-nvme/rlvr/evals/openrouter_owl_alpha_full/preds_5shot_full.jsonl"
```

注意：

- free model 可能会被上游 provider rate limit。
- 如果遇到 `429 temporarily rate-limited upstream`，优先保留 `resume`，不要从头重跑。
- `openrouter/owl-alpha` 已完成 `limit=1` smoke test，链路是通的；全量吞吐是否稳定，取决于当时的 OpenRouter/provider 限流状态。

## Important Notes

- 当前 `step1500` 最佳结果不是从零连续训练得到的。
- 它是先拿到 `step1000` 的 HF 权重，再新开一轮训练得到的“权重继续训练”结果，不是真正的 optimizer state resume。
- 当前训练的主要问题不是 reward 无效，而是 `verl + vLLM` 在长时间训练下的 `sleep/cumem` 稳定性。
- 为避免 checkpoint 存盘报错，actor checkpoint 当前只保存 `model`，不保存 optimizer。

## Main Scripts

- baseline eval:
  - `/root/rl_project/scripts/run_llmsql_baseline.py`
- richer offline eval:
  - `/root/rl_project/scripts/eval_llmsql_predictions.py`
- offline eval summary:
  - `/root/rl_project/scripts/summarize_llmsql_offline_eval.py`
- OpenRouter eval:
  - `/root/rl_project/scripts/run_llmsql_openrouter.py`
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
