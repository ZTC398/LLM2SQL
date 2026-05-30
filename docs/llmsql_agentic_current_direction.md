# LLMSQL Agentic Current Direction

当前主线已经明确切换为：

- `tool-only`
- `terminal reward`
- `custom agent loop on top of verl`

而不是：

- 旧的 verifier-guided repair loop

## Mainline

主实现目标：

- 利用 `verl` 已有的 `agent_loop` 能力
- 从 base model 开始做 `LLMSQL` agentic RL
- 做出一个更完整、可展示、可写进简历的工程项目

## Deprecated Path

以下内容不再作为当前主线：

- `scripts/run_llmsql_agent_loop.py`
- 基于 `verification_incorrect` 的 repair design
- 以 `effective_1000 / effective_2500` 作为 agent 训练起点

这些内容可以保留作历史实验参考，但不继续作为当前版本设计依据。
