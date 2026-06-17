#!/usr/bin/env python3
"""Convert LLMSQL benchmark data into verl parquet format."""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset
from llmsql.utils.evaluation_utils import connect_sqlite, execute_sql
from llmsql.utils.utils import choose_prompt_builder


SPLIT_TO_FILE = {
    "train": "train_questions.jsonl",
    "val": "val_questions.jsonl",
    "test": "test_questions.jsonl",
}


def parse_args() -> argparse.Namespace:
    default_dataset_dir = os.environ.get(
        "LLMSQL_DATASET_DIR",
        "/root/shared-nvme/rlvr/datasets/llmsql-2.0",
    )
    default_output_dir = os.environ.get(
        "LLMSQL_VERL_DATA_DIR",
        "/root/shared-nvme/rlvr/verl_data/llmsql_5shot",
    )
    parser = argparse.ArgumentParser(
        description="Prepare verl parquet files from llmsql-2.0 JSONL files.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=default_dataset_dir,
        help="Directory containing LLMSQL benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory to write verl parquet files.",
    )
    parser.add_argument(
        "--num-fewshots",
        type=int,
        default=5,
        choices=[0, 1, 5],
        help="Official LLMSQL prompt style to use.",
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
        default=None,
        help="Optional override for the verl data_source field.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_sql_result(result: list[tuple[Any, ...]] | None) -> list[list[Any]] | None:
    if result is None:
        return None
    return [list(row) for row in result]


def build_record(
    *,
    example: dict[str, Any],
    table: dict[str, Any],
    split: str,
    idx: int,
    data_source: str,
    prompt_builder,
    conn,
) -> dict[str, Any]:
    sample_row = table["rows"][0] if table["rows"] else []
    prompt_text = prompt_builder(
        example["question"],
        table["header"],
        table["types"],
        sample_row,
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
            "prompt_style": f"{data_source.split(':')[-1]}",
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
    prompt_builder = choose_prompt_builder(args.num_fewshots)
    data_source = args.data_source or f"llmsql-bench/llmsql-2.0:{args.num_fewshots}shot"

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
                data_source=data_source,
                prompt_builder=prompt_builder,
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
                    "data_source": data_source,
                },
                indent=2,
            )
        )

    conn.close()


if __name__ == "__main__":
    main()
