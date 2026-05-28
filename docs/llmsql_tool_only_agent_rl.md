# LLMSQL Tool-Only Agentic RL Design

## Objective

把当前单轮 `text-to-SQL` 训练，升级为一个与 `Search-R1` 思路一致的多轮 agent：

- 模型可以自主决定是否调用 SQL 执行工具
- 环境只返回工具原始 observation
- 环境不返回任何 `correct / incorrect` 判断
- 模型自己决定是否继续 loop，或何时停止并提交最终 SQL
- 最终 reward 只在轨迹结束时计算

这个设计的核心目标是：

1. 让 `loop` 行为不是被环境显式触发，而是由策略自己学出来
2. 使同一套 agent 协议既能用于有监督数据，也能迁移到无监督或弱监督场景
3. 让项目真正具备 `agentic RL` 的形态，而不是单纯的 verifier-guided repair

## Why This Version

前一版 `execution_error + verification_incorrect` loop 虽然更容易出效果，但有一个根本问题：

- 中间 observation 含有显式 judge 信号

这会让策略依赖“环境告诉我错了再修”，而不是自己根据工具结果判断是否继续。

如果目标是做一个更标准的 agentic RL 项目，这种设计不够干净。

因此本版采用与 `Search-R1` 相同的哲学：

- 环境提供的是工具结果
- correctness 只在终局 reward 里体现

## Search-R1 Mapping

`Search-R1` 的多轮结构可以抽象成：

- Action 1: `<search> query </search>`
- Observation 1: `<information> search results </information>`
- Action 2: `<search>` or `<answer>`
- Final reward: exact-match on final answer

对应到 `LLMSQL`：

- Action 1: `<sql> query </sql>`
- Observation 1:
  - execution error, or
  - execution result preview
- Action 2: `<sql>` or `<final_sql>`
- Final reward: execution match on final SQL

这两者在结构上是一致的。

## Agent Protocol

### Allowed Actions

模型在每一轮只能输出一个 action：

- `<sql> ... </sql>`
  - 调用 SQL 执行工具，查看执行结果
- `<final_sql> ... </final_sql>`
  - 声明最终 SQL，结束轨迹

如果输出不符合这两种格式，视为 invalid action。

### Environment Observation

环境对 `<sql>` 的返回只包含工具原始信息：

1. 执行失败

```text
<observation>
<execution_error>OperationalError: no such column: ...</execution_error>
</observation>
```

2. 执行成功

```text
<observation>
<execution_result>
columns: ["col_a", "col_b"]
row_count: 3
rows:
["v1", "v2"]
["v3", "v4"]
["v5", "v6"]
</execution_result>
</observation>
```

环境不会返回：

- `correct`
- `incorrect`
- gold result
- gold SQL

### Invalid Action Observation

如果模型输出非法格式，环境只返回 action-format 提示：

```text
<observation>
<invalid_action>
You must output exactly one action:
1. <sql> SELECT ... </sql>
2. <final_sql> SELECT ... </final_sql>
</invalid_action>
</observation>
```

这与 `Search-R1` 的 invalid action 提示机制一致。

## Stop Policy

模型必须自己决定是否继续 loop：

- 如果觉得结果已经足够支持最终答案，就输出 `<final_sql>`
- 如果觉得还需要验证或修正，就继续输出 `<sql>`

这正是要通过 RL 学出来的行为。

### Max Turns

为了避免轨迹无限增长，需要设置：

- `max_tool_turns`
  - 最多允许多少次 `<sql>` tool call
- `final_force_turn`
  - 在达到 `max_tool_turns` 后，再给模型一次强制输出 `<final_sql>` 的机会

推荐第一版：

- `max_tool_turns = 2`
- `max_total_turns = 3`

也就是：

1. 工具查询
2. 可选再查一次
3. 最后一轮必须给最终 SQL

## Prompt Design

当前 `LLMSQL 5-shot` prompt 不能直接复用，因为它要求：

- 输出 only SQL

这与多轮 action protocol 冲突。

因此需要一套新的 agent prompt 数据。

### Initial Prompt

第一轮 prompt 建议为：

```text
You are solving a SQLite text-to-SQL task with access to a SQL execution tool.

At each step, output exactly one action:
1. <sql> SELECT ... </sql> to inspect the table by running a query.
2. <final_sql> SELECT ... </final_sql> when you are confident this is the final answer.

Rules:
- Use table name "Table"
- Use only valid SQLite SELECT queries
- Use double quotes around all column names and the table name
- You may call <sql> multiple times before <final_sql>
- Do not output explanations

Question: ...
Columns: ...
Types: ...
Sample row: ...
```

### Final Turn Prompt

在最后强制终止轮，可追加一句：

```text
This is your final turn. Output <final_sql> ... </final_sql> now.
```

## Data Preparation

建议新建一份 agent 数据，而不是复用当前 parquet：

- `data_prep/prepare_llmsql_agent_data.py`

输出结构仍与现有 `verl` 数据格式兼容：

- `prompt`
- `reward_model.ground_truth`
- `extra_info.table_id`
- `extra_info.question`
- `extra_info.gold_result_json`

区别只是：

- `prompt` 改成新的 tool-only agent instruction

建议数据目录：

- `.../verl_data/llmsql_agent_0shot/`

第一版建议用 `0-shot` agent prompt，不用当前 `5-shot` SQL 例子。

原因：

- 旧 few-shot 示例是单轮答题示范，会抑制工具使用
- 先把 agent 行为训出来更重要

## Environment Implementation

建议参考 `Search-R1` 新建：

- `agent/llmsql_tool_env.py`
- `agent/llmsql_generation.py`

### `llmsql_tool_env.py`

负责：

- 解析 `<sql>` / `<final_sql>`
- 把 `"Table"` 替换成真实 `table_id`
- 执行 sqlite
- 裁剪结果预览
- 构造 observation 字符串

关键接口：

- `step(action_text, table_id) -> observation, done, meta`

### `llmsql_generation.py`

负责：

- 复刻 `Search-R1` 的 `LLMGenerationManager`
- 在 rollout 中循环：
  - 生成 action
  - 调环境
  - 拼 observation
  - 更新 rolling prompt
- 最终生成完整 trajectory

建议保留的统计：

- `turns_stats`
- `valid_action_stats`
- `tool_call_stats`
- `forced_final_turn_stats`

## Reward Design

reward 只看轨迹结束时的最终 SQL。

### Final Reward

设最终 `<final_sql>` 为待评分 SQL：

- 执行结果与 gold result 一致：`+1.0`
- 否则：`0.0`

### Step Penalty

为了学出 stop policy，必须加轻微 step penalty：

- 每次 `<sql>` tool call：`-0.02` 到 `-0.05`

最终 reward 示例：

```text
final_reward = exact_exec_match_reward - step_penalty * num_sql_calls
```

其中：

- exact match 正确：`1.0`
- 错误：`0.0`

这样策略就会自然学到：

- 能直接结束就别多查
- 但如果多查一次能显著提高成功率，也值得付出 penalty

### Invalid Action Penalty

可选增加：

- invalid action: `-0.05`

用于稳定格式学习。

## Custom Reward Function

当前 `rewards/llmsql_sql_reward.py` 不适用于多轮 agent。

原因：

- 它会在整个字符串里抽 SQL
- 多轮轨迹里会有多个 `<sql>`
- 它可能错误地用中间 SQL 计分

因此要新建：

- `rewards/llmsql_agent_sql_reward.py`

这个 reward 需要：

1. 解析最后一个 `<final_sql>`
2. 替换真实 `table_id`
3. 执行并与 `gold_result_json` 对比
4. 减去 tool-call penalty

## veRL Integration Strategy

最稳妥的集成路径，是模仿 `Search-R1`，不要硬往当前单轮 `main_ppo` 里塞 patch。

建议新增一个本地 trainer 入口，而不是直接修改 upstream `third_party/verl`：

- `agent/main_ppo_tool_loop.py`
- `agent/ray_trainer_tool_loop.py`

### Why Not Patch Vanilla `main_ppo`

当前标准 `verl.trainer.main_ppo` 假设：

- 一次 rollout 直接得到最终 response

而 tool-only agent 需要：

- rollout 内部存在多轮 action-observation loop

这不是 reward function 能解决的，必须改 generation path。

### What To Copy From Search-R1

直接参考 `Search-R1` 的这两层：

- `search_r1/llm_agent/generation.py`
- `verl/trainer/ppo/ray_trainer.py`

对应在本项目中实现：

- `agent/llmsql_generation.py`
- `agent/ray_trainer_tool_loop.py`

## Trainer Changes

需要改的关键点只有两个：

1. Validation generation

像 `Search-R1` 一样，在 `_validate()` 中判断：

- 如果 `do_tool_loop=False`，走普通生成
- 如果 `do_tool_loop=True`，走 `generation_manager.run_llm_loop(...)`

2. Training generation

在训练 loop 中：

- 原始 `gen_batch = batch.pop(...)`
- 调用 `generation_manager.run_llm_loop(...)`
- 把最终 trajectory 重新 union 回 batch
- 再走 logprob、advantage、PPO update

也就是说，RL update 本身不用重写，主要改 rollout 生成路径。

## File-Level Plan

建议新增这些文件：

- `data_prep/prepare_llmsql_agent_data.py`
- `agent/llmsql_tool_env.py`
- `agent/llmsql_generation.py`
- `agent/ray_trainer_tool_loop.py`
- `agent/main_ppo_tool_loop.py`
- `rewards/llmsql_agent_sql_reward.py`
- `training/run_llmsql_grpo_tool_loop.sh`

## Training Launcher

建议新脚本：

- `training/run_llmsql_grpo_tool_loop.sh`

新增配置项：

- `DO_TOOL_LOOP=true`
- `MAX_TOOL_TURNS=2`
- `MAX_OBS_LENGTH=256`
- `SQL_RESULT_MAX_ROWS=5`
- `SQL_RESULT_MAX_CHARS=512`
- `TOOL_STEP_PENALTY=0.02`

## Validation Plan

先不要直接开训，按这个顺序做：

1. 实现 tool-only inference loop 原型
2. 跑 `limit=200` smoke test
3. 看模型是否真的会：
   - 输出 `<sql>`
   - 根据 observation 再输出下一步
   - 在某些 case 上主动继续、某些 case 上主动停止
4. 再接训练

## Success Criteria

这个方向值得继续，至少满足一条：

- `effective_1000 + tool-only loop` 接近 `effective_2500`
- 训练后模型能明显减少平均 tool calls，同时保持或提升最终正确率
- 模型在没有显式 verifier 的情况下学会“有些题直接收敛，有些题多查一次”

## Key Risk

最大风险不是 trainer 集成，而是：

- 模型只会机械地一律多查一次
- 或者一律不查，直接 `<final_sql>`

这也是为什么：

- step penalty 必须加
- action format 必须严格
- final reward 必须只看 `<final_sql>`

否则 stop policy 学不出来。
