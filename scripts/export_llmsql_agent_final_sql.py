#!/usr/bin/env python3
"""Export agent trajectories into official-eval-compatible LLMSQL predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_SQL_RE = re.compile(r"<final_sql>\s*(.*?)\s*</final_sql>", re.DOTALL | re.IGNORECASE)
TEXT_KEYS = ("completion", "response", "prediction", "output", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the last <final_sql> from agent trajectories and export "
            "official-eval-compatible LLMSQL predictions."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the JSONL file containing agent trajectory predictions.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Path to write official-eval-compatible predictions JSONL. "
            "Defaults to <input_stem>.final_sql.jsonl next to the input file."
        ),
    )
    parser.add_argument(
        "--analysis-path",
        default=None,
        help=(
            "Optional path to write per-sample extraction analysis JSONL. "
            "Defaults to <input_stem>.final_sql_analysis.jsonl."
        ),
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help=(
            "Optional path to write summary JSON. "
            "Defaults to <input_stem>.final_sql_summary.json."
        ),
    )
    parser.add_argument(
        "--text-key",
        default=None,
        help=(
            "Optional explicit text field to read. "
            "If unset, the script tries completion/response/prediction/output/text."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_text_key(row: dict[str, Any], preferred_key: str | None) -> str:
    if preferred_key is not None:
        if preferred_key not in row:
            raise KeyError(f"Missing text key '{preferred_key}' in row keys={sorted(row.keys())}")
        return preferred_key

    for key in TEXT_KEYS:
        if key in row:
            return key
    raise KeyError(
        "Could not find a supported text key in row. "
        f"Tried: {TEXT_KEYS}. Available keys={sorted(row.keys())}"
    )


def extract_last_final_sql(text: str) -> str | None:
    matches = FINAL_SQL_RE.findall(text)
    if not matches:
        return None
    candidate = matches[-1].strip()
    return candidate or None


def build_default_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else build_default_path(
        input_path, ".final_sql.jsonl"
    )
    analysis_path = (
        Path(args.analysis_path)
        if args.analysis_path
        else build_default_path(input_path, ".final_sql_analysis.jsonl")
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else build_default_path(input_path, ".final_sql_summary.json")
    )

    rows = read_jsonl(input_path)
    counters: Counter[str] = Counter()
    detected_text_key: str | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out_f, analysis_path.open("w") as analysis_f:
        for idx, row in enumerate(rows, start=1):
            if "question_id" not in row:
                raise KeyError(
                    f"Missing question_id at row {idx}. Available keys={sorted(row.keys())}"
                )

            text_key = pick_text_key(row, args.text_key)
            if detected_text_key is None:
                detected_text_key = text_key

            raw_text = row[text_key]
            if not isinstance(raw_text, str):
                raise TypeError(
                    f"Row {idx} field '{text_key}' must be str, got {type(raw_text).__name__}"
                )

            final_sql = extract_last_final_sql(raw_text)
            has_final_sql = final_sql is not None

            counters["total"] += 1
            counters["has_final_sql"] += int(has_final_sql)
            counters["missing_final_sql"] += int(not has_final_sql)

            exported_record = {
                "question_id": row["question_id"],
                # Empty completion is intentional: it prevents official eval from
                # accidentally scoring intermediate <sql> actions as final answers.
                "completion": final_sql or "",
            }
            analysis_record = {
                "question_id": row["question_id"],
                "text_key": text_key,
                "has_final_sql": has_final_sql,
                "final_sql": final_sql,
                "original_text_preview": raw_text[:500],
            }

            out_f.write(json.dumps(exported_record, ensure_ascii=False) + "\n")
            analysis_f.write(json.dumps(analysis_record, ensure_ascii=False) + "\n")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "analysis_path": str(analysis_path),
        "text_key": detected_text_key,
        "total": counters["total"],
        "has_final_sql": counters["has_final_sql"],
        "missing_final_sql": counters["missing_final_sql"],
        "final_sql_extract_rate": (
            counters["has_final_sql"] / counters["total"] if counters["total"] else 0.0
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
