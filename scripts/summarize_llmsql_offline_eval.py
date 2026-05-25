#!/usr/bin/env python3
"""Summarize multiple LLMSQL offline-eval summary.json files into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ORDER = [
    "base",
    "full_500",
    "full_1000",
    "effective_1500",
    "effective_2000",
    "effective_2500",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect offline LLMSQL evaluation summaries into a markdown table."
    )
    parser.add_argument(
        "--summary-json",
        nargs="+",
        required=True,
        help="List of summary.json files to aggregate.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the markdown table.",
    )
    return parser.parse_args()


def infer_step_key(summary: dict) -> str:
    label = summary["label"]
    pred_path = summary["predictions_path"]

    if "baseline_qwen25_coder_3b_test" in label:
        return "base"
    if "full_500steps" in label:
        return "full_500"
    if "full_1000steps" in label:
        return "full_1000"
    if "effective1500" in pred_path:
        return "effective_1500"
    if "effective2000" in pred_path:
        return "effective_2000"
    if "step1500_hf" in pred_path:
        return "effective_2500"
    raise ValueError(f"Unable to infer step key from label={label}, pred_path={pred_path}")


def row_for(summary: dict) -> dict:
    return {
        "step_key": infer_step_key(summary),
        "label": summary["label"],
        "sql_extract_rate": summary["sql_extract_rate"],
        "first_sql_exec_rate": summary["first_sql_exec_rate"],
        "any_sql_exec_rate": summary["any_sql_exec_rate"],
        "first_exec_match_rate": summary["first_exec_match_rate"],
        "any_exec_match_rate": summary["any_exec_match_rate"],
        "sample_sql_error_count": summary["sample_sql_error_count"],
        "candidate_sql_error_count": summary["candidate_sql_error_count"],
        "agg_func_acc_first": summary["agg_func_acc_first"],
        "where_clause_acc_first": summary["where_clause_acc_first"],
    }


def fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in args.summary_json:
        with open(summary_path) as f:
            rows.append(row_for(json.load(f)))

    rows.sort(key=lambda row: DEFAULT_ORDER.index(row["step_key"]))

    lines = [
        "# LLMSQL Offline Eval Summary",
        "",
        "| Weight | first_exec_match | any_exec_match | first_sql_exec | any_sql_exec | sample_sql_errors | candidate_sql_errors | agg_acc | where_acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["step_key"],
                    fmt_float(row["first_exec_match_rate"]),
                    fmt_float(row["any_exec_match_rate"]),
                    fmt_float(row["first_sql_exec_rate"]),
                    fmt_float(row["any_sql_exec_rate"]),
                    str(row["sample_sql_error_count"]),
                    str(row["candidate_sql_error_count"]),
                    fmt_float(row["agg_func_acc_first"]),
                    fmt_float(row["where_clause_acc_first"]),
                ]
            )
            + " |"
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
