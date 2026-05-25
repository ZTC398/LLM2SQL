# LLMSQL RL Next Steps

## Goal

当前项目已经证明两件事：

- `LLMSQL + verl + GRPO` 这条线是能学到东西的。
- 当前最好结果相对基模有明显提升，不是纯噪声。

下一阶段的目标，不再是“继续盲目拉长训练”，而是把项目补成一个更完整、更能写进简历的 RL 实验工程：

1. 解释模型到底提升了什么能力。
2. 识别当前 reward 和训练曲线为什么看起来不显著。
3. 做一个低资源但更接近真实系统的增强版本。
4. 产出一套可以稳定复现、展示、对比的结果表。

## Strategy

当前不建议立刻更换数据集。

原因：

- `LLMSQL` 已经证明 reward 有效，继续深挖的性价比最高。
- 现在最大问题不是“任务太差”，而是缺少拆解指标和误差分析。
- 如果在这个阶段直接换任务，容易重新掉回“只会跑通，不知道发生了什么”。

因此下一步优先做：

- 补离线评测
- 补错误分类
- 补轻量 repair loop
- 最后做一轮小规模 ablation

## Phase 1: Offline Evaluation

### Objective

把当前单一的 `mean score` 或官方 accuracy，拆成更可解释的多指标视图。

### Why

现在训练结果已经显示：

- accuracy 在提升
- `sql_errors` 在下降

但 TensorBoard 里的很多值变化很小，不足以说明模型到底是：

- SQL 格式更稳定了
- 执行成功率更高了
- 还是结果匹配率更高了

### Metrics

建议至少补这几项：

- `sql_extract_rate`
  - 能否从模型输出中抽出 SQL。
- `sql_exec_rate`
  - SQL 能否在 sqlite 上成功执行。
- `exec_match_rate`
  - 执行结果是否与标准答案一致。
- `exact_sql_match_rate`
  - 生成 SQL 是否与标准 SQL 完全一致。
- `sql_error_count`
  - 执行报错条数。
- `sql_error_rate`
  - 执行报错占比。

建议额外补两类更有解释力的结构化指标：

- `agg_func_acc`
  - 对包含 `COUNT / AVG / SUM / MAX / MIN` 的题，单独看准确率。
- `where_clause_acc`
  - 对带条件过滤的题，单独看准确率。

### Deliverables

- `scripts/eval_llmsql_predictions.py`
- 统一格式的评测输出 `json`
- 一份简明结果表，至少包含：
  - baseline
  - 当前 best RL checkpoint
  - 后续 repair 版本

### Done Criteria

完成后，应该可以直接回答：

- 当前 RL 的收益主要来自哪里？
- 是 SQL 可执行性提升更明显，还是结果匹配率提升更明显？
- 聚合题、条件题、简单 lookup 题，哪类收益最大？

## Phase 2: Error Analysis

### Objective

把错误从“答错了”拆成若干具体类型。

### Why

如果没有错误分类，后续所有改进都只能靠猜。

### Suggested Error Types

- `no_sql_extracted`
  - 没抽到合法 SQL。
- `execution_error`
  - SQL 抽出来了，但执行失败。
- `wrong_column`
  - 选错列。
- `wrong_aggregation`
  - 聚合函数错，比如该 `COUNT` 却写成普通 `SELECT`。
- `wrong_condition_column`
  - `WHERE` 用错列。
- `wrong_condition_value`
  - `WHERE` 值错。
- `wrong_result_despite_executable`
  - 能执行，但结果不对。

### Deliverables

- `scripts/analyze_llmsql_errors.py`
- 一个错误分布报告，至少输出：
  - 各类错误占比
  - baseline vs RL 的对比

### Done Criteria

完成后，应该能回答：

- RL 到底修复了哪些错误？
- 剩余瓶颈集中在哪一类？
- 下一步应该改 reward、改 prompt，还是加 repair loop？

## Phase 3: Lightweight Repair Loop

### Objective

在不显著增加训练成本的前提下，做一个更接近真实系统的 SQL 修复流程。

### Why

这是当前最容易把项目从“toy 训练实验”推进到“更像真实 agent / production flow”的一步。

而且它对资源要求低，不需要重新做大规模训练。

### Minimal Design

单轮 repair 即可：

1. 模型先生成一次 SQL。
2. 如果无法执行，收集 sqlite 错误信息。
3. 把原问题、schema、第一次 SQL、错误信息一起喂回模型。
4. 让模型再生成一次修复版 SQL。
5. 统计修复前后执行成功率和最终正确率。

### Deliverables

- `scripts/run_llmsql_repair_eval.py`
- 一份 single-pass vs repair 的对比结果

### Key Metrics

- `repair_attempt_rate`
- `repair_exec_recovery_rate`
- `repair_final_gain`

### Done Criteria

完成后，应该能回答：

- repair 是否真的提升最终正确率？
- repair 的收益主要来自“修语法错”还是“修条件错”？
- 当前 best RL checkpoint 和 repair 组合后，是否比单纯继续训练更划算？

## Phase 4: Small Ablation Study

### Objective

做一轮规模不大但结论清晰的对比实验。

### Why

项目最后写进简历，不需要十几组实验，但至少需要 3 到 4 个有信息量的比较对象。

### Recommended Comparison Set

- baseline
- current best RL checkpoint
- best RL checkpoint + repair
- optional: reward shaping variant

### Optional Reward Shaping

如果要继续动训练，优先考虑小改动而不是重做整套数据：

- 当前 exact execution match = `1.0`
- SQL 可执行但结果不对 = 小正奖励
- 能抽出 SQL 但执行失败 = 更小正奖励或 `0`

这样可以测试：

- shaping 是否能提升早期训练稳定性
- shaping 是否能进一步降低 `sql_errors`

### Deliverables

- 一张总表
- 一张错误类型对比表
- 一段简短结论

## Recommended Execution Order

建议按下面顺序推进：

1. 先做 `Phase 1` 离线评测。
2. 基于评测结果做 `Phase 2` 错误分类。
3. 如果剩余错误里执行错误较多，先做 `Phase 3` repair loop。
4. 最后决定是否值得做 reward shaping 或继续长训。

这个顺序的原因很简单：

- 没有评测，后面所有改进都缺解释。
- 没有错误分类，repair 和 reward shaping 都容易拍脑袋。
- repair loop 的实现成本低，且最可能带来项目层面的“非 toy 化”收益。

## Practical Milestone Plan

### Milestone A

补齐评测与错误分析。

交付物：

- `eval_llmsql_predictions.py`
- `analyze_llmsql_errors.py`
- baseline vs best RL 的对比结果

### Milestone B

加单轮 repair。

交付物：

- `run_llmsql_repair_eval.py`
- repair 前后对比表

### Milestone C

决定是否继续动训练。

判断标准：

- 如果 repair 收益明显，优先把它作为项目亮点。
- 如果错误主要集中在语法/执行层面，再考虑 reward shaping。
- 如果错误已经主要是语义理解偏差，继续长训的边际收益可能有限。

## Resume-Oriented Framing

如果以上阶段做完，这个项目可以从“跑了一个 GRPO 训练”升级成下面这种表述：

- 在 `LLMSQL` 上基于 `verl + GRPO` 构建 text-to-SQL 强化学习训练与评测流程，推动模型从 baseline `74.7%` 提升到 `83%+`
- 自建执行式 reward、离线多指标评测与错误分类工具，定位 SQL 生成中的执行错误、条件错误与聚合错误
- 设计基于执行反馈的 SQL repair loop，比较单次生成与双阶段修复流程在最终执行正确率上的收益

这个版本比单纯写“做了 RL 训练”更完整，也更像真实工程项目。

## Current Recommendation

下一步直接开始 `Phase 1 + Phase 2`。

原因：

- 成本最低
- 对当前项目价值最高
- 是后续 repair 和训练决策的前置条件

在这之前，不建议继续单纯拉长训练步数。
