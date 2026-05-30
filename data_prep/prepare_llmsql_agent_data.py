#!/usr/bin/env python3
"""Prepare LLMSQL parquet data for tool-only agentic RL training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
from llmsql.utils.evaluation_utils import connect_sqlite, execute_sql


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SPLIT_TO_FILE = {
    "train": "train_questions.jsonl",
    "val": "val_questions.jsonl",
    "test": "test_questions.jsonl",
}


DEFAULT_DATASET_DIR = "/root/shared-nvme/rlvr/datasets/llmsql-2.0"
DEFAULT_OUTPUT_DIR = "/root/shared-nvme/rlvr/verl_data/llmsql_agent_0shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LLMSQL agentic RL parquet files from llmsql-2.0 JSONL files.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory containing LLMSQL benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write agentic verl parquet files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=sorted(SPLIT_TO_FILE),
        help="Dataset splits to convert.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke data generation.",
    )
    parser.add_argument(
        "--data-source",
        default="llmsql-bench/llmsql-2.0:agent-0shot",
        help="Value written to the verl data_source field.",
    )
    parser.add_argument(
        "--agent-name",
        default="llmsql_tool_agent",
        help="Agent loop name written into each row.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_sql_result(result: list[tuple[Any, ...]] | None) -> list[list[Any]] | None:
    if result is None:
        return None
    return [list(row) for row in result]


def build_agent_prompt(
    *,
    question: str,
    columns: list[Any],
    types: list[Any],
    sample_row: list[Any],
) -> str:
    return (
        "You are solving a SQLite text-to-SQL task with access to a SQL execution tool.\n\n"
        "Your goal is to produce the correct final SQL for the question.\n\n"
        "At each step, output exactly one action in one of these two formats:\n\n"
        "1. <sql> SELECT ... </sql>\n"
        "   Use this when you want to inspect data or verify a hypothesis by executing a SQL query.\n\n"
        "2. <final_sql> SELECT ... </final_sql>\n"
        "   Use this when you are confident this is the final answer.\n\n"
        "Rules:\n"
        "- Use table name \"Table\".\n"
        "- Use only valid SQLite SELECT queries.\n"
        "- Allowed functions: ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']\n"
        "- Allowed condition operators: ['=', '>', '<', '!=']\n"
        "- Allowed SQL keywords: ['SELECT', 'WHERE', 'AND']\n"
        "- Always use double quotes around all column names and the table name, even for one-word names.\n"
        "- Do not output explanations.\n"
        "- Do not output markdown fences.\n"
        "- Do not output anything before or after the action tags.\n"
        "- If you are uncertain, use <sql> first.\n"
        "- End the trajectory by outputting <final_sql> ... </final_sql>.\n\n"
        "Environment behavior:\n"
        "- If you output <sql> ... </sql>, the environment will execute it and return either:\n"
        "  - an execution error, or\n"
        "  - a preview of the execution result\n"
        "- The environment will not tell you whether your query is correct or incorrect.\n"
        "- You must decide for yourself whether to continue or to output <final_sql>.\n\n"
        f"Question: {question}\n"
        f"Columns: {columns}\n"
        f"Types: {types}\n"
        f"Sample row: {sample_row}"
    )


def build_record(
    *,
    example: dict[str, Any],
    table: dict[str, Any],
    split: str,
    idx: int,
    data_source: str,
    agent_name: str,
    conn: Any,
) -> dict[str, Any]:
    sample_row = table["rows"][0] if table["rows"] else []
    prompt_text = build_agent_prompt(
        question=example["question"],
        columns=table["header"],
        types=table["types"],
        sample_row=sample_row,
    )
    gold_result = normalize_sql_result(execute_sql(conn, example["sql"]))
    if gold_result is None:
        raise ValueError(
            f"Gold SQL failed for question_id={example['question_id']}: {example['sql']}"
        )

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "text-to-sql",
        "agent_name": agent_name,
        "reward_model": {
            "style": "rule",
            "ground_truth": example["sql"],
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "question_id": example["question_id"],
            "table_id": example["table_id"],
            "question": example["question"],
            "gold_result_json": json.dumps(gold_result, ensure_ascii=False),
            "prompt_style": "agent-0shot",
            "agent_protocol": "tool-only",
        },
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        item["table_id"]: item for item in load_jsonl(dataset_dir / "tables.jsonl")
    }
    conn = connect_sqlite(str(dataset_dir / "sqlite_tables.db"))

    for split in args.splits:
        questions = load_jsonl(dataset_dir / SPLIT_TO_FILE[split])
        if args.max_samples_per_split is not None:
            questions = questions[: args.max_samples_per_split]

        rows = [
            build_record(
                example=example,
                table=tables[example["table_id"]],
                split=split,
                idx=idx,
                data_source=args.data_source,
                agent_name=args.agent_name,
                conn=conn,
            )
            for idx, example in enumerate(questions)
        ]

        out_path = output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(out_path))
        print(
            json.dumps(
                {
                    "split": split,
                    "rows": len(rows),
                    "output_path": str(out_path),
                    "data_source": args.data_source,
                    "agent_name": args.agent_name,
                },
                indent=2,
            )
        )

    conn.close()


if __name__ == "__main__":
    main()
