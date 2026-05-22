#!/usr/bin/env python3
"""Quick local sanity check for the llmsql verl reward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

from rewards.llmsql_sql_reward import compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-path",
        default="/root/shared-nvme/rlvr/verl_data/llmsql_5shot/val.parquet",
    )
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet_path)
    dataset = load_dataset("parquet", data_files={"data": str(parquet_path)})["data"]
    sample = dataset[args.index]

    prompt_text = sample["prompt"][0]["content"]
    gold_sql = sample["reward_model"]["ground_truth"]
    extra_info = sample["extra_info"]

    wrong_sql = 'SELECT COUNT(*) FROM "Table";'
    exact_reward = compute_score(
        sample["data_source"],
        gold_sql,
        gold_sql,
        extra_info=extra_info,
    )
    wrong_reward = compute_score(
        sample["data_source"],
        wrong_sql,
        gold_sql,
        extra_info=extra_info,
    )

    print(
        json.dumps(
            {
                "parquet_path": str(parquet_path),
                "index": args.index,
                "question_id": extra_info["question_id"],
                "table_id": extra_info["table_id"],
                "gold_reward": exact_reward,
                "wrong_reward": wrong_reward,
                "gold_sql": gold_sql,
                "prompt_preview": prompt_text[:500],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
