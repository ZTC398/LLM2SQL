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
- local 30B upper bound：`0.8523`
- local 30B eval dir：`/root/shared-nvme/rlvr/evals/qwen3_coder_30b_a3b_full`

当前新增方向：

- `tool-only` 多轮 SQL agent 设计
- 目标是让模型自己决定是否继续 loop，而不是由环境显式告诉它当前答案对错
- 长期目标是接成一个与 `Search-R1` 思路一致的 `agentic RL` 训练版本

当前原型结果：

- `effective_1000` on `probe200`: `0.7500 -> 0.8000`，`+0.0500`
- `effective_2500` on `probe200`: `0.7950 -> 0.8300`，`+0.0350`
- `agentic_step1000` on full official eval: `0.7845`
- 两组里 `correct_to_wrong_rate` 都是 `0.0`
- 当前收益主要来自 `verification_incorrect -> correct`，不是 execution error repair
- `2x4090` 上的 full agentic train 已经跑通到 `step1000` checkpoint，但连续长跑在 `step1447` 左右遇到 `vLLM sleep/unmap` 路径崩溃；当前恢复策略是 `resume_from_path + disable_sleep_mode`

最新离线评测总表见：

- [outputs/2026-05-25/llmsql_weight_progression_table.md](/root/rl_project/outputs/2026-05-25/llmsql_weight_progression_table.md)

详细实验记录见：

- [outputs/2026-05-22/llmsql_verl_experiment_report.md](/root/rl_project/outputs/2026-05-22/llmsql_verl_experiment_report.md)
- [docs/llmsql_agentic_resume_project_plan.md](/root/rl_project/docs/llmsql_agentic_resume_project_plan.md)

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
| `qwen3_30b_a3b_full` | `0.8523` | `0.9987` | `21` | `87` | `0.8034` | `0.8525` |

说明：

- `1500/2000/2500` 这里是 effective total steps，不是一次 uninterrupted optimizer resume。
- 映射关系是：
  - `effective_1500` = 从 `step1000` HF 权重继续训练那条线的 `global_step_500`
  - `effective_2000` = 同一条线的 `global_step_1000`
  - `effective_2500` = 同一条线的 `global_step_1500`
- 当前最靠谱的主指标是 `first_exec_match`。
- `exact_sql_match_rate` 已移除，不再作为主指标。

## Agent Loop Snapshot

当前同时保留两类结果：

- `probe200`：快速看 loop 是否有正向收益
- `full official eval`：把整段 trajectory 导出最后的 `<final_sql>`，再接官方 `evaluate`

| Model | Split | first_acc | final_acc | gain | repair_attempt_rate | repair_success_rate | verification_repair_gain | correct_to_wrong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `effective_1000` | `probe200` | `0.7500` | `0.8000` | `+0.0500` | `0.2500` | `0.2000` | `0.2083` | `0.0000` |
| `effective_2500` | `probe200` | `0.7950` | `0.8300` | `+0.0350` | `0.2050` | `0.1707` | `0.1842` | `0.0000` |

说明：

- `agentic_step1000 full` 里的 `first_acc=0.0` 是预期现象，不代表模型完全失效。
- 原因是当前 tool-only agent 的 first turn 通常输出 exploratory `<sql>`，不是最终答案。
- 因此 full official 口径的主结果应看 `final_acc=0.7845`，也就是“导出最后 `<final_sql>` 后再跑官方 evaluate”的分数。
- 上表里的 `repair_attempt_rate / repair_success_rate / verification_repair_gain / correct_to_wrong_rate` 是旧 `verify_incorrect` 原型 loop 的指标，不适合直接复用到当前 tool-only agent full eval。

当前 tool-only agent 的 full official eval 单独记：

| Model | Split | final_acc | final_matches | final_sql_errors | final_sql_extract_rate |
| --- | --- | --- | --- | --- | --- |
| `agentic_step1000` | `full` | `0.7845` | `12421 / 15834` | `816` | `0.9997` |

本地结果目录：

- `effective_1000`:
  `/root/shared-nvme/rlvr/evals/qwen25_coder_3b_agent_loop_effective1000_probe200`
- `effective_2500`:
  `/root/shared-nvme/rlvr/evals/qwen25_coder_3b_agent_loop_effective2500_probe200`
- `agentic_step1000`:
  `/root/shared-nvme/rlvr/evals/qwen25_coder_3b_agentic_step1000_full`

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
- local large model: `/root/shared-nvme/rlvr/models/Qwen3-Coder-30B-A3B-Instruct`
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

4. 跑本地 30B 上限测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n rlvr bash /root/rl_project/scripts/run_llmsql_qwen3_30b_a3b.sh
```

监控进度：

```bash
watch -n 5 'python /root/rl_project/scripts/count_llmsql_predictions.py --output-dir /root/shared-nvme/rlvr/evals/qwen3_coder_30b_a3b_full --total 15834'
```

5. Agentic RL 设计方向

说明：

- 当前更推荐的方向，不是 verifier-guided repair，而是 `tool-only` agent loop。
- 环境只返回 SQL 工具原始 observation：
  - 执行报错
  - 或执行结果预览
- 环境不返回 `correct / incorrect` 判断。
- 模型自己决定：
  - 继续输出 `<sql> ... </sql>` 再查一次
  - 或输出 `<final_sql> ... </final_sql>` 结束轨迹
- 最终 reward 只在轨迹结束时计算，依赖最后的 `<final_sql>`。

这版设计更接近 `Search-R1`：

- `Search-R1`: `<search> -> <information> -> <search|answer>`
- `LLMSQL tool loop`: `<sql> -> <execution_result> -> <sql|final_sql>`

当前建议的实现顺序：

1. 新建 agent 数据准备脚本，生成新的 tool-use prompt，而不是复用当前 `only SQL` 的 5-shot prompt
2. 实现 SQL tool environment，只返回工具 observation
3. 实现 Search-R1 风格的多轮 generation manager
4. 再接到 `verl` 的 rollout/trainer

详细设计见：

- [docs/llmsql_tool_only_agent_rl.md](/root/rl_project/docs/llmsql_tool_only_agent_rl.md)
- [docs/llmsql_agentic_sql_loop.md](/root/rl_project/docs/llmsql_agentic_sql_loop.md)
- [docs/llmsql_agentic_resume_project_plan.md](/root/rl_project/docs/llmsql_agentic_resume_project_plan.md)
- [docs/llmsql_agentic_current_direction.md](/root/rl_project/docs/llmsql_agentic_current_direction.md)

6. 查看 parquet 结构

```bash
conda run -n rlvr python /root/rl_project/scripts/inspect_llmsql_parquet.py
```

7. 跑 richer offline eval

```bash
conda run -n rlvr python /root/rl_project/scripts/eval_llmsql_predictions.py \
  --predictions-path /root/shared-nvme/rlvr/evals/qwen25_coder_3b_dual4090_full_500steps_noval_hf/preds_5shot_full.jsonl \
  --output-dir /root/shared-nvme/rlvr/analysis/llmsql_offline_eval_v2/qwen25_coder_3b_dual4090_full_500steps_noval_hf \
  --label full_500steps
```

8. 汇总多份 offline eval

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
- 当前 `agent loop` 的设计方向已经调整为 `tool-only`。
- 也就是说，中间 observation 只暴露工具原始返回，不暴露 verifier-style `incorrect` 信号。
- 最终目标是让模型自己学会“何时继续查、何时停止”，而不是等环境告诉它“你错了再修”。

## Main Scripts

- baseline eval:
  - `/root/rl_project/scripts/run_llmsql_baseline.py`
- agent loop prototype:
  - `/root/rl_project/scripts/run_llmsql_agent_loop.py`
- tool-only agent design:
  - `/root/rl_project/docs/llmsql_tool_only_agent_rl.md`
- richer offline eval:
  - `/root/rl_project/scripts/eval_llmsql_predictions.py`
- offline eval summary:
  - `/root/rl_project/scripts/summarize_llmsql_offline_eval.py`
- OpenRouter eval:
  - `/root/rl_project/scripts/run_llmsql_openrouter.py`
- agent loop design:
  - `/root/rl_project/docs/llmsql_agentic_sql_loop.md`
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
