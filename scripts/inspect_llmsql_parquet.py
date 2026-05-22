#!/usr/bin/env python3
"""Inspect LLMSQL parquet files used by verl."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_DIR = Path("/root/shared-nvme/rlvr/verl_data/llmsql_5shot")


def _maybe_parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def _preview(value: Any, max_chars: int) -> str:
    parsed = _maybe_parse(value)
    if isinstance(parsed, (dict, list, tuple)):
        rendered = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        rendered = str(parsed)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "\n...<truncated>..."
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing train/val/test parquet files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to inspect.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max rendered characters per field before truncation.",
    )
    args = parser.parse_args()

    parquet_path = args.data_dir / f"{args.split}.parquet"
    df = pd.read_parquet(parquet_path)

    print(f"file: {parquet_path}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print("dtypes:")
    for column, dtype in df.dtypes.astype(str).items():
        print(f"  - {column}: {dtype}")

    print("\nrow previews:")
    for idx, row in df.head(args.rows).iterrows():
        print(f"\n=== row {idx} ===")
        for column, value in row.items():
            print(f"[{column}]")
            print(_preview(value, args.max_chars))


if __name__ == "__main__":
    main()
