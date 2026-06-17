# Third-Party Notices

This repository depends on external projects and datasets. Their code, data, and trademarks remain under the terms specified by the respective upstream maintainers.

## LLMSQL

- Upstream repository: <https://github.com/LLMSQL/llmsql-benchmark>
- Dataset: <https://huggingface.co/datasets/llmsql-bench/llmsql-2.0>
- Observed package license metadata: `MIT`
- Usage in this repository:
  - benchmark data format
  - prompt construction utilities through the `llmsql` Python package
  - evaluation conventions for text-to-SQL execution accuracy

## verl

- Upstream repository: <https://github.com/volcengine/verl>
- Vendored path: `third_party/verl`
- Upstream license in vendored copy: `Apache-2.0`
- Usage in this repository:
  - GRPO training framework
  - rollout and policy optimization infrastructure

## Qwen Models

- Model family reference: <https://github.com/QwenLM/Qwen>
- Usage in this repository:
  - base model and local reference model for text-to-SQL generation

Users are responsible for complying with the licenses and usage terms of all upstream dependencies, models, and datasets.
