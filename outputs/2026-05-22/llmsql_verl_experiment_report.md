# LLMSQL + verl 实验进展报告

日期：2026-05-22

## 1. 目标与当前结论

当前工作目标是基于 `verl` 在 `LLMSQL` 任务上对 `Qwen2.5-Coder-3B-Instruct` 做 GRPO 强化训练，并用官方 `LLMSQL` evaluate 流程做可对齐评测。

截至目前，结论比较明确：

- 任务本身是可学的，且提升幅度明显。
- 在当前配置下，模型从基模 `74.71%` 提升到了当前最佳 `83.49%`。
- 训练收益是真实存在的，当前主要瓶颈不是算法无效，而是 `verl + vLLM` 的训练稳定性问题。
- 后期反复崩溃的核心原因不是 reward 设计错误，也不是 loss 发散，而是 `vLLM sleep/cumem` 相关基础设施问题。

## 2. 环境与代码位置

- 代码根目录：`/root/rl_project`
- 数据/模型/实验输出根目录：`/root/shared-nvme/rlvr`
- Conda 环境：`rlvr`
- 基础模型：`/root/shared-nvme/rlvr/models/Qwen2.5-Coder-3B-Instruct`
- 主训练脚本：
  - `/root/rl_project/training/run_llmsql_grpo_full.sh`
  - `/root/rl_project/training/run_llmsql_grpo_full_dual4090_500steps_noval.sh`
  - `/root/rl_project/training/run_llmsql_grpo_full_dual4090_3000steps_noval.sh`
  - `/root/rl_project/training/run_llmsql_grpo_full_dual4090_fromstep1000_2000steps_sleep.sh`
- 官方 baseline/evaluate 脚本：
  - `/root/rl_project/scripts/run_llmsql_baseline.py`

## 3. 数据与训练设置

当前训练使用的是整理后的 `LLMSQL 5-shot parquet` 数据：

- 训练集：`/root/shared-nvme/rlvr/verl_data/llmsql_5shot/train.parquet`
- 测试集：`/root/shared-nvme/rlvr/verl_data/llmsql_5shot/test.parquet`
- 官方评测问题文件：`/root/shared-nvme/rlvr/datasets/llmsql-2.0/test_questions.jsonl`
- 官方表结构文件：`/root/shared-nvme/rlvr/datasets/llmsql-2.0/tables.jsonl`
- 官方 SQLite DB：`/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db`

当前主训练超参大致为：

- `TRAIN_BATCH_SIZE=4`
- `PPO_MINI_BATCH_SIZE=2`
- `ROLLOUT_N=4`
- `AGENT_NUM_WORKERS=4`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=192`
- `PPO_MAX_TOKEN_LEN_PER_GPU=3072`
- `ROLLOUT_GPU_MEM_UTIL=0.45`
- `ROLLOUT_MAX_MODEL_LEN=1024`
- `ROLLOUT_MAX_NUM_BATCHED_TOKENS=1024`
- `ROLLOUT_MAX_NUM_SEQS=4`
- `actor_rollout_ref.rollout.layered_summon=True`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `actor_rollout_ref.actor.checkpoint.save_contents=["model"]`

## 4. 已完成实验

### 4.1 基模官方 baseline

模型：

- `/root/shared-nvme/rlvr/models/Qwen2.5-Coder-3B-Instruct`

结果文件：

- `/root/shared-nvme/rlvr/outputs/baseline_qwen25_coder_3b_test/meta_5shot_full.json`

结果：

- accuracy = `0.7470632815460402`
- matches = `11829 / 15834`
- sql_errors = `106`

说明：

- 这是当前所有训练结果的参考起点。
- 这次 baseline 使用的是官方 `LLMSQL` 推理+评测脚本。

### 4.2 从零开始训练：500 steps

实验名：

- `qwen25_coder_3b_dual4090_full_500steps_noval`

训练脚本：

- `/root/rl_project/training/run_llmsql_grpo_full_dual4090_500steps_noval.sh`

合并后模型：

- `/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_full_500steps_noval_hf`

评测结果：

- `/root/shared-nvme/rlvr/evals/qwen25_coder_3b_dual4090_full_500steps_noval_hf/meta_5shot_full.json`

结果：

- accuracy = `0.8047871668561324`
- matches = `12743 / 15834`
- sql_errors = `105`

现象：

- 只训练 `500` 步，准确率已经从 `74.71%` 提升到 `80.48%`。
- 说明任务与 reward 设计整体是通的。
- 这次训练结束时出现过 checkpoint 保存问题，错误表现是保存 optimizer 状态时报 `basic_ios::clear: iostream error`。
- 后续已将 actor checkpoint 保存内容改为 `["model"]`，避免继续保存 optimizer 状态。

### 4.3 从零开始训练：3000-step 计划，实际保留到 step1000

实验名：

- `qwen25_coder_3b_dual4090_full_3000steps_noval`

训练脚本：

- `/root/rl_project/training/run_llmsql_grpo_full_dual4090_3000steps_noval.sh`

关键情况：

- 该实验是从基模重新开始训练，不是接着 `500-step` 继续。
- 计划训练 `3000` 步，但在大约 `step 1800+` 附近崩溃。
- 当前有效保留并完成评测的 checkpoint 是 `global_step_1000`。

合并后模型：

- `/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_full_3000steps_noval_step1000_hf`

评测结果：

- `/root/shared-nvme/rlvr/evals/qwen25_coder_3b_dual4090_full_3000steps_noval_step1000_hf/meta_5shot_full.json`

结果：

- accuracy = `0.8080080838701529`
- matches = `12794 / 15834`
- sql_errors = `58`

现象：

- 相比 `500-step`，准确率只小幅提升，但 `sql_errors` 从 `105` 大幅下降到 `58`。
- 说明模型的 SQL 语法/执行稳定性明显变好了。
- 该实验后期的崩溃不是 reward 发散，而是 `vLLM` worker 在 sleep 路径上异常退出。

### 4.4 载入 step1000 权重后继续训练：sleep 版本，保留到 step1500

实验名：

- `qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep`

训练脚本：

- `/root/rl_project/training/run_llmsql_grpo_full_dual4090_fromstep1000_2000steps_sleep.sh`

该脚本最关键的一行是：

- `MODEL_PATH=/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_full_3000steps_noval_step1000_hf`

这意味着：

- 这条实验线不是从基模开始。
- 它是加载了 `step1000` 的 HF 模型权重，然后重新开一轮新的训练。
- 它不是严格意义上的“断点续训/true resume”。

更准确地说，这条线属于：

- `权重继续训练`

而不是：

- `保留 optimizer / trainer / checkpoint state 的原地续训`

这点非常重要，因为它带来两个后果：

1. actor 初始权重来自 `step1000`，不是基模。
2. KL 的 reference policy 也会围绕这个 `MODEL_PATH` 初始化，因此这一阶段的 KL 约束锚点也已经变成了 `step1000`，不再是最初的基模。

所以：

- 这个实验目录下的 `step1500`，不能简单等同于“从零连续训练 2500 步”。
- 更准确的理解是：“以 `step1000` 模型为新起点，再做了 1500 步新的 RL 训练”。

该实验保存的有效 checkpoint：

- `global_step_500`
- `global_step_1000`
- `global_step_1500`

对应输出目录：

- `/root/shared-nvme/rlvr/outputs/verl/llmsql_grpo_full/qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep`

其中 `step1500` 合并后模型：

- `/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep_step1500_hf`

最终评测结果：

- `/root/shared-nvme/rlvr/evals/qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep_step1500_hf/meta_5shot_full.json`

结果：

- accuracy = `0.834912214222559`
- matches = `13220 / 15834`
- sql_errors = `50`

现象：

- 这是当前最好结果。
- 相比基模，准确率提升了约 `8.78` 个百分点。
- 相比 `500-step`，准确率提升了约 `3.01` 个百分点。
- 相比 `step1000`，准确率提升了约 `2.69` 个百分点。
- `sql_errors` 继续下降到 `50`。

## 5. 当前结果总表

| 实验 | 起始权重 | 评测模型 | accuracy | matches | sql_errors |
| --- | --- | --- | --- | --- | --- |
| 基模 baseline | 基模 | base | 0.747063 | 11829 / 15834 | 106 |
| full_500steps_noval | 基模 | step500 | 0.804787 | 12743 / 15834 | 105 |
| full_3000steps_noval_step1000 | 基模 | step1000 | 0.808008 | 12794 / 15834 | 58 |
| fromstep1000_2000steps_sleep_step1500 | `step1000` HF 权重 | 新实验 step1500 | 0.834912 | 13220 / 15834 | 50 |

## 6. 关键实验现象

### 6.1 任务本身偏“好学”

当前最重要的正面现象是：

- 即使只做相对便宜的 3B 规模训练，LLMSQL 也能在短步数内获得明显收益。
- 从 `74.71%` 到 `80%+` 很快。
- 再往上到 `83.49%` 也是真实可达的。

这说明：

- 任务难度对当前模型容量是友好的。
- reward 至少在方向上是对的。
- prompt/schema 形式对模型也比较友好。

### 6.2 后期崩溃与训练质量无强相关

多次崩溃的关键模式比较一致：

- 训练前中期可以稳定推进。
- 能持续产生更好的 checkpoint。
- 崩溃多发生在 `step 1800+` 左右。

当前最明确的一次日志证据来自：

- `/root/shared-nvme/rlvr/logs/qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep_train.log`

其中关键报错为：

- `Fatal Python error: none_dealloc: deallocating None`
- `Worker proc VllmWorker-0 died unexpectedly`
- `Exception: Call to collective_rpc method failed: cancelled`

这说明主要问题在：

- `verl` 训练循环里的 `sleep_replicas()`
- `vLLM` 的 `sleep/wake`
- `cumem` / worker lifecycle

而不是：

- loss 爆炸
- reward 明显失效
- 训练样本本身不可用

### 6.3 sleep on / sleep off 都有问题

当前观察到的是：

- `sleep on`：能训得更远，但在 `1800+` 左右会遇到 `vLLM sleep/cumem` 崩溃。
- `sleep off`：会因为 rollout 常驻显存，导致 actor 优化阶段更容易 OOM。

所以当前真正卡住继续扩步数的，不是算法收益，而是工程稳定性权衡。

### 6.4 `layered_summon=True` 可能在放大 sleep 路径脆弱性

当前主训练脚本里显式设置了：

- `actor_rollout_ref.rollout.layered_summon=True`

而 `verl` 默认配置里该项是 `False`。

这不等于它一定是唯一根因，但它会让训练更依赖当前这套 `sleep/wake` 的混合路径，因此它是后续最值得优先试验的稳定性开关之一。

## 7. 当前最佳模型与应如何理解

当前最佳模型是：

- `/root/shared-nvme/rlvr/merged_models/qwen25_coder_3b_dual4090_fromstep1000_2000steps_sleep_step1500_hf`

它不是“从零连续训到 1500 步”的模型，也不是“从零连续训到 2500 步”的严格等价物。

更准确的定义是：

- 先得到一个 `from-scratch step1000` 模型。
- 再把它 merge 成 HF。
- 再把这个 HF 模型当作新的 `MODEL_PATH`，开一个新的 GRPO 训练任务。
- 在这个新任务里走到 `global_step_1500`。

因此：

- 如果只看权重轨迹，它确实继承了之前 `step1000` 的知识。
- 但如果看优化器状态、训练器状态、KL 锚点和训练进程语义，它不是同一条 uninterrupted training run。

## 8. 当前判断

从研究角度看，当前已经可以下几个比较有把握的判断：

- `LLMSQL + Qwen2.5-Coder-3B + GRPO + 当前 reward` 这条线是成立的。
- 现阶段最值得优化的不是“是否有效”，而是“如何稳定把更长训练跑完”。
- 当前最好结果已经达到 `83.49%`，因此后续可以把更多精力放在：
  - 训练稳定性
  - 更规范的 resume 语义
  - 更长训练与收益曲线
  - 是否需要引入更难任务，而不是只盯着最初的 pipeline 打通

## 9. 下一步建议

如果后续继续做，我建议优先级如下：

1. 保留当前最佳 `step1500` 作为阶段性 best model，不要覆盖。
2. 真正梳理“resume 训练”和“load 权重再训”的差别，避免后续混淆 step 含义。
3. 训练稳定性上优先尝试：
   - `layered_summon=False`
   - 审查 `enable_sleep_mode`
   - 在不动 `ROLLOUT_N=4` 和主 batch 设计的前提下，只调 rollout 侧显存占用策略
4. 如果目标转向研究，而不是纯刷 LLMSQL 分数，可以考虑更难的任务，避免在过于容易的数据集上过早饱和。

