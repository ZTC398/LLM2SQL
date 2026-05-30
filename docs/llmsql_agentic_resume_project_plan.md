# LLMSQL Agentic RL Resume Project Plan

## Objective

把当前 `LLMSQL + verl + GRPO` 单轮 SQL RL 项目，升级成一个更完整的 `agentic RL` 项目版本。

这条线的首要目标不是追求论文式结论，而是做成一个：

- 架构完整
- 训练可运行
- 评测可解释
- 能稳定写进简历和项目介绍

的工程项目。

## Project Positioning

当前项目已经证明：

- `LLMSQL` 任务本身可学
- `verl + custom reward` 训练链路可跑通
- 3B 模型从基模到 RL 后有明确收益

下一阶段不再把重点放在：

- 继续拉单轮训练步数
- 做很多研究型 ablation
- 反复优化旧的 verifier-guided repair prototype

而是转成一个更成体系的 `tool-only agentic RL` 项目。

## Core Decisions

### 1. 不手改 `verl` 框架

项目实现应尽量复用当前 vendored `verl` 已有能力：

- `experimental/agent_loop`
- `agent_name`
- `agent_loop_config_path`
- 多轮 rollout

原则：

- 项目代码写在 `/root/rl_project` 自己目录下
- `third_party/verl/` 不作为主战场
- 除非确认存在明确 blocker，否则不对 `verl` 本体做定制修改

### 2. 从 base 开始训练

agentic 版本的训练起点统一使用：

- `/root/shared-nvme/rlvr/models/Qwen2.5-Coder-3B-Instruct`

不使用：

- `step1000`
- `effective_1000`
- `effective_2500`

作为训练 warm start。

原因不是它们没价值，而是当前项目目标是做一个叙事更清楚的完整工程：

- base model
- agent data
- tool-only loop
- terminal reward
- agentic RL training
- eval and analysis

这样更适合作为简历项目表述。

### 3. 旧的 verifier loop 不再作为主线

现有：

- `scripts/run_llmsql_agent_loop.py`

对应的是旧的 `execution_error / verification_incorrect` repair prototype。

这条线可以保留作历史参考，但不再作为当前主实现路线，不继续扩展它。

### 4. 优先做“完整系统”，而不是先追最优分数

第一阶段成功标准不是：

- 必须超过当前 `0.8349`

而是：

- 数据管线完整
- agent loop 训练可跑
- reward 逻辑清楚
- rollout 协议闭环
- 有一套能展示项目价值的评测输出

如果这条线能在小规模训练上验证可行，再继续扩大训练。

## Target System

目标系统采用 `Search-R1` 风格的 tool-only 轨迹结构：

1. 用户问题进入 agent prompt
2. 模型输出一个 action
3. action 只能是两类之一：
   - `<sql> ... </sql>`
   - `<final_sql> ... </final_sql>`
4. 环境执行 `<sql>` 并返回原始 observation
5. 模型自己决定继续查还是提交最终 SQL
6. reward 只在终局计算，依据最后的 `<final_sql>`

### Environment Contract

环境只返回工具 observation，不返回显式 judge：

- `execution_error`
- `execution_result` preview
- `invalid_action`
- `final_turn` reminder

环境不返回：

- `correct`
- `incorrect`
- gold SQL
- gold result

### Reward Contract

reward 只看最终答案：

- 从整段轨迹中提取最后的 `<final_sql>`
- 替换 `"Table"` 为真实 `table_id`
- 执行 SQL
- 与 `gold_result_json` 比较

第一版默认采用简单 terminal reward：

- final execution match: `1.0`
- else: `0.0`

后续如果需要提升训练稳定性，再考虑轻量 shaping。

## Implementation Scope

第一阶段只做最小可行 agentic RL 版本。

### In Scope

- 新 agent parquet 数据
- 新 prompt 协议
- 自定义 SQL tool environment
- 自定义 `AgentLoopBase` 子类
- 新 terminal reward
- 新训练启动脚本
- 新 smoke train 路径
- 基础 trajectory / offline eval 指标

### Out of Scope

- 改 `verl` trainer 主体
- 远程 SQL 服务化
- 多工具混合系统
- 复杂 memory / planner
- 大规模 research ablation
- 旧 verifier loop 的继续维护

## Planned File Layout

第一阶段预期新增如下文件：

### Data

- `data_prep/prepare_llmsql_agent_data.py`

### Agent Runtime

- `llmsql_agentic/__init__.py`
- `llmsql_agentic/prompting.py`
- `llmsql_agentic/tool_env.py`
- `llmsql_agentic/agent_loop.py`

### Reward

- `rewards/llmsql_agent_reward.py`

### Config

- `configs/agent_loops/llmsql_tool_agent.yaml`

### Training

- `training/run_llmsql_grpo_agentic.sh`
- `training/run_llmsql_grpo_agentic_dual4090_smoke.sh`

### Docs / Analysis

- 当前文档

## Phase Plan

## Phase 0: Freeze Baseline

### Goal

冻结当前单轮项目作为 agentic 版本对照基线。

### Deliverables

- README 中明确：
  - 当前单轮 best
  - 当前 agentic 目标
  - 旧 repair prototype 不再是主线

### Done Criteria

完成后，后续 agentic 结果统一拿以下对象对比：

- base model
- current best single-turn RL
- local 30B upper bound

## Phase 1: Agent Data

### Goal

生成新的 tool-only agent 数据，而不是复用当前 `5-shot only SQL` prompt。

### Design

每条样本仍保持 veRL 兼容格式，但增加 agent 字段：

- `prompt`
- `reward_model.ground_truth`
- `extra_info.question`
- `extra_info.table_id`
- `extra_info.gold_result_json`
- `agent_name = llmsql_tool_agent`

prompt 采用：

- `0-shot`
- 明确 action protocol
- 不带旧 few-shot SQL 示例

### Output Directory

- `/root/shared-nvme/rlvr/verl_data/llmsql_agent_0shot`

### Done Criteria

- 成功生成 `train/val/test.parquet`
- `inspect_llmsql_parquet.py` 可正常查看
- 抽样确认 prompt 已切换为 agent 协议

## Phase 2: Agent Loop Runtime

### Goal

在不修改 `verl` 框架的前提下，接入一个可训练的多轮 SQL loop。

### Design

实现一个自定义 `AgentLoopBase` 子类，使用：

- `agent_name`
- `agent_loop_config_path`

接入 `verl` 的 rollout。

该 loop 负责：

1. 读取 `raw_prompt`
2. 生成 action
3. 解析 `<sql>` / `<final_sql>`
4. 执行 sqlite 或生成 invalid observation
5. 把 observation token 追加回轨迹
6. 在终局交给 reward 评分

### First-Version Constraints

- `max_tool_turns = 2`
- 强制 final turn
- 单工具，单数据库
- 不做复杂并行工具调用

### Done Criteria

- rollout 能在 `verl` agent loop 路径上跑通
- response 中正确包含：
  - assistant action token
  - environment observation token
- final trajectory 能进入 reward 计算

## Phase 3: Reward Wiring

### Goal

让 agent 轨迹的 terminal reward 能稳定工作。

### Design

新 reward 文件只做一件事：

- 从整段轨迹里提取最后的 `<final_sql>`

然后：

- 替换真实表名
- 执行 SQL
- 与 `gold_result_json` 对比

### Important Constraint

不再沿用单轮 reward 的“从任意位置找 SQL 即可评分”逻辑。

agent 版 reward 的语义必须是：

- 只有 `<final_sql>` 才算最终答案

### Done Criteria

- reward smoke test 通过
- 缺失 `<final_sql>` 时 reward 行为可解释
- 正确 final SQL 能拿到满分

## Phase 4: Training Framework

### Goal

把 agent 数据、agent loop、reward 全部接进现有训练启动脚本体系。

### Design

新训练脚本应明确覆盖这些差异：

- `DATA_DIR` 指向 agent parquet
- `default_agent_loop = llmsql_tool_agent`
- `agent_loop_config_path` 指向自定义 config
- `reward.custom_reward_function.path` 改为 agent reward
- `data.max_response_length` 适当放大
- 从 base model 开始训练

### First Training Milestone

先做：

- smoke train
- 小规模 step
- 只验证 agentic training 路径打通

再做：

- 正式训练

### Done Criteria

- smoke train 启动并完成
- 能保存 checkpoint
- 不依赖旧 verifier loop 脚本

## Phase 5: Evaluation and Project Story

### Goal

补齐 agentic 版本的工程叙事和指标。

### Metrics

除了已有主指标，还要补：

- `final_exec_match`
- `avg_tool_calls`
- `invalid_action_rate`
- `forced_final_turn_rate`
- `final_sql_extract_rate`
- `trajectory_length`

### Output

至少要有：

- 训练配置说明
- 轨迹协议说明
- 小规模训练结果
- 对比基线结果
- 错误案例

### Done Criteria

最后应能用简洁语言说明：

1. 项目做了什么
2. 为什么它是 agentic RL，而不是普通单轮 SQL RL
3. 工程上哪些部分是你自己补出来的
4. 跑通了哪些训练与评测链路

## Resume-Oriented Framing

项目最终对外表述建议强调这些点：

- 基于 `veRL` 的多轮 agentic RL 项目实现
- 为 `LLMSQL` 设计了 tool-only SQL interaction protocol
- 自定义 SQLite environment 与 terminal reward
- 在不改动底层 RL 框架的前提下完成 agent loop 接入
- 构建了训练数据、训练脚本、reward、评测分析的完整闭环

这比“我把一个 benchmark 跑到多少分”更适合作为简历项目亮点。

## Immediate Next Step

文档确认后，按以下顺序开始编码：

1. `prepare_llmsql_agent_data.py`
2. `llmsql_agent_reward.py`
3. `llmsql_agentic/tool_env.py`
4. `llmsql_agentic/agent_loop.py`
5. `configs/agent_loops/llmsql_tool_agent.yaml`
6. `training/run_llmsql_grpo_agentic.sh`

在这之后再做：

7. smoke train
8. agent eval / analysis
