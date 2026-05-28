# LLMSQL Agentic SQL Loop

## Goal

把当前单轮 `text-to-SQL + execution reward` 项目，推进成一个更明确的 agentic RL 方向：

- 模型先生成 SQL
- 环境执行 SQL
- 环境返回执行或验证反馈
- 模型基于反馈修正 SQL
- 最终 reward 绑定整个两步轨迹

这里的重点不是做一个复杂 agent，而是做一个最小可行的 `execution + verifier feedback` SQL agent。

## Why This Direction

当前项目已经接近这个 benchmark 的单轮 ceiling：

- 3B RL best: `0.8349`
- local 30B upper bound: `0.8523`

继续拉长单轮训练步数，边际收益已经变小。

如果还想继续推进项目复杂度和结果上限，更合理的方向不是继续堆 step，而是让模型在失败后获得环境反馈并尝试修正。

## Task Framing

第一阶段只做一个两步 agent：

1. Step 1: generate SQL
2. Execute SQL in sqlite
3. Environment returns one of:
   - `execution_error`
   - `verification_incorrect`
   - `correct`
4. If not `correct`, Step 2 generates a revised SQL
5. Final score is computed on the last SQL

这已经满足 agentic 化的核心条件：

- 有 action
- 有 environment feedback
- 有 second action
- 有 trajectory-level outcome

## Feedback Design

### Allowed Feedback

只给 verifier-style feedback，不直接泄露 gold answer。

可用两类信号：

- `execution_error`
  - SQL 无法执行时返回 sqlite error
- `verification_incorrect`
  - SQL 可执行，但执行结果与 gold answer 不一致

### Not Allowed

第一阶段不返回：

- gold SQL
- gold result content
- exact correct row/value

否则任务会退化成带答案提示的修复，而不是 agentic feedback learning。

## Loop Policy

第一阶段推荐默认策略：

- Step 1 correct:
  - stop
- Step 1 no SQL / execution error:
  - trigger repair
- Step 1 executable but incorrect:
  - trigger repair with verifier feedback

也就是说，repair 不只修 syntax error，也修 `executed but incorrect` 的语义错误。

## Prototype First

在接 `verl` 训练前，先做一个推理原型脚本验证交互信号：

- `scripts/run_llmsql_agent_loop.py`

这个脚本的目标不是最终项目结论，而是验证：

1. second turn 是否真的有 lift
2. lift 主要来自哪类反馈
3. second turn 是否会破坏原本正确的样本

## Outputs

原型脚本至少输出：

- `preds_first_*.jsonl`
- `preds_final_*.jsonl`
- `agent_traces_*.jsonl`
- `meta_*.json`

其中 `agent_traces` 记录每条样本：

- first completion
- first extracted SQL
- first status
- repair triggered or not
- feedback type
- second completion
- second extracted SQL
- final status

## Metrics

除了已有的官方 accuracy / offline eval 外，新增这些 loop-specific 指标：

- `repair_attempt_rate`
- `repair_success_rate`
- `verification_repair_gain`
- `execution_repair_gain`
- `correct_to_wrong_rate`
- `first_to_final_exec_match_gain`

如果 `correct_to_wrong_rate` 太高，说明 repair prompt 或 accept policy 有问题。

## Success Criteria For Prototype

原型值得继续接训练，至少要满足一条：

- `effective_1000 + loop` 接近 `effective_2500`
- `effective_2500 + loop` 明显优于 `effective_2500`
- `verification_incorrect` 类型样本能被 second turn 有效修复

如果 second turn 几乎没有提升，就不值得把 loop 接进训练。

## Training Path

如果原型验证有效，第二阶段再接到 `verl`：

1. 将单轮 prompt 改为多轮 interaction trajectory
2. 在 rollout 中执行 SQL
3. 将 environment feedback 作为 observation 注入下一轮
4. 最终 reward 绑定轨迹结果

## Reward Sketch

推荐的简单 reward：

- final correct: `+1.0`
- first-step correct bonus: `+0.1`
- second-step correct: `+1.0`
- each extra repair step penalty: `-0.05`
- final incorrect: `0.0`

这个设计会鼓励：

- 能一步做对就一步做对
- 一步做不对就尽量通过反馈修回来
- 不鼓励无意义多走一步

## Recommended Order

1. 先实现并跑 `run_llmsql_agent_loop.py`
2. 跑 `effective_1000` 和 `effective_2500`
3. 看 second-turn gain 是否足够
4. 再决定是否值得改 `verl` 训练
