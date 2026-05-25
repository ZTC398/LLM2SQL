#!/usr/bin/env python3
"""Offline per-sample evaluation for LLMSQL predictions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from llmsql.utils.evaluation_utils import connect_sqlite, fix_table_name
from llmsql.utils.regex_extractor import find_sql


AGG_FUNCS = ("count(", "avg(", "sum(", "max(", "min(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run richer offline evaluation for LLMSQL prediction files."
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="Path to preds_*.jsonl produced by baseline or checkpoint eval.",
    )
    parser.add_argument(
        "--questions-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/test_questions.jsonl",
        help="Path to official LLMSQL questions jsonl.",
    )
    parser.add_argument(
        "--db-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db",
        help="Path to official LLMSQL sqlite database.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save per-sample and aggregate analysis outputs.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional experiment label stored in output metadata.",
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


def load_questions(path: Path) -> dict[int, dict[str, Any]]:
    questions: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            questions[item["question_id"]] = item
    return questions


def execute_sql_with_error(
    conn: sqlite3.Connection, sql: str
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return sorted(cur.fetchall()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def has_agg(sql: str) -> bool:
    normalized = sql.strip().lower()
    return any(func in normalized for func in AGG_FUNCS)


def has_where(sql: str) -> bool:
    return " where " in f" {sql.strip().lower()} "


def safe_preview(value: Any, max_len: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    for candidate in candidates:
        if candidate["exec_match"]:
            return candidate
    for candidate in candidates:
        if candidate["sql_executable"]:
            return candidate
    return candidates[0]


def evaluate_prediction_file(
    predictions_path: Path,
    questions_path: Path,
    db_path: Path,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)
    predictions = read_jsonl(predictions_path)
    conn = connect_sqlite(str(db_path))

    per_sample_path = output_dir / "per_sample_analysis.jsonl"
    summary_path = output_dir / "summary.json"

    totals = Counter()
    subset = {
        "agg": Counter(),
        "where": Counter(),
    }
    error_types_any = Counter()
    error_types_first = Counter()

    with per_sample_path.open("w") as out_f:
        for item in predictions:
            question_id = item["question_id"]
            question = questions[question_id]
            gold_sql = question["sql"]
            table_id = question["table_id"]
            completion = item["completion"]

            gold_result, gold_error = execute_sql_with_error(conn, gold_sql)
            if gold_error is not None:
                raise RuntimeError(f"Gold SQL failed for question_id={question_id}: {gold_error}")

            extracted_sqls = find_sql(completion)
            candidates: list[dict[str, Any]] = []
            for rank, raw_sql in enumerate(extracted_sqls, start=1):
                fixed_sql = fix_table_name(raw_sql, table_id)
                pred_result, pred_error = execute_sql_with_error(conn, fixed_sql)
                candidates.append(
                    {
                        "rank": rank,
                        "raw_sql": raw_sql,
                        "fixed_sql": fixed_sql,
                        "sql_executable": pred_error is None,
                        "exec_match": pred_error is None and pred_result == gold_result,
                        "sql_error_message": pred_error,
                        "pred_result_preview": safe_preview(pred_result),
                    }
                )

            best_candidate = pick_best_candidate(candidates)
            first_candidate = candidates[0] if candidates else None
            sql_extracted = bool(candidates)
            any_sql_exec = any(candidate["sql_executable"] for candidate in candidates)
            any_exec_match = any(candidate["exec_match"] for candidate in candidates)
            first_sql_exec = first_candidate["sql_executable"] if first_candidate else False
            first_exec_match = first_candidate["exec_match"] if first_candidate else False
            candidate_sql_error_count = sum(
                int(not candidate["sql_executable"]) for candidate in candidates
            )

            gold_has_agg = has_agg(gold_sql)
            gold_has_where = has_where(gold_sql)

            if not sql_extracted:
                error_type_any = "no_sql_extracted"
                error_type_first = "no_sql_extracted"
            elif not any_sql_exec:
                error_type_any = "execution_error"
                error_type_first = "execution_error"
            elif any_exec_match:
                error_type_any = "correct"
            else:
                error_type_any = "wrong_result_despite_executable"

            if not sql_extracted:
                error_type_first = "no_sql_extracted"
            elif not first_sql_exec:
                error_type_first = "execution_error"
            elif first_exec_match:
                error_type_first = "correct"
            else:
                error_type_first = "wrong_result_despite_executable"

            totals["total"] += 1
            totals["sql_extracted"] += int(sql_extracted)
            totals["any_sql_executable"] += int(any_sql_exec)
            totals["any_exec_match"] += int(any_exec_match)
            totals["first_sql_executable"] += int(first_sql_exec)
            totals["first_exec_match"] += int(first_exec_match)
            totals["sample_sql_error_count"] += int(sql_extracted and not any_sql_exec)
            totals["candidate_sql_error_count"] += candidate_sql_error_count
            error_types_any[error_type_any] += 1
            error_types_first[error_type_first] += 1

            if gold_has_agg:
                subset["agg"]["total"] += 1
                subset["agg"]["any_exec_match"] += int(any_exec_match)
                subset["agg"]["first_exec_match"] += int(first_exec_match)
            if gold_has_where:
                subset["where"]["total"] += 1
                subset["where"]["any_exec_match"] += int(any_exec_match)
                subset["where"]["first_exec_match"] += int(first_exec_match)

            record = {
                "question_id": question_id,
                "question": question["question"],
                "table_id": table_id,
                "gold_sql": gold_sql,
                "gold_result_preview": safe_preview(gold_result),
                "completion": completion,
                "sql_extract_rate_hit": sql_extracted,
                "first_sql_exec_rate_hit": first_sql_exec,
                "any_sql_exec_rate_hit": any_sql_exec,
                "first_exec_match": first_exec_match,
                "any_exec_match": any_exec_match,
                "gold_has_agg": gold_has_agg,
                "gold_has_where": gold_has_where,
                "error_type_first": error_type_first,
                "error_type_any": error_type_any,
                "num_extracted_sqls": len(candidates),
                "candidate_sql_error_count": candidate_sql_error_count,
                "first_candidate": first_candidate,
                "best_candidate": best_candidate,
                "candidates": candidates,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = totals["total"]
    summary = {
        "label": label,
        "predictions_path": str(predictions_path),
        "questions_path": str(questions_path),
        "db_path": str(db_path),
        "per_sample_path": str(per_sample_path),
        "total": total,
        "sql_extract_rate": totals["sql_extracted"] / total if total else 0.0,
        "first_sql_exec_rate": totals["first_sql_executable"] / total if total else 0.0,
        "any_sql_exec_rate": totals["any_sql_executable"] / total if total else 0.0,
        "first_exec_match_rate": totals["first_exec_match"] / total if total else 0.0,
        "any_exec_match_rate": totals["any_exec_match"] / total if total else 0.0,
        "sample_sql_error_count": totals["sample_sql_error_count"],
        "sample_sql_error_rate": totals["sample_sql_error_count"] / total if total else 0.0,
        "candidate_sql_error_count": totals["candidate_sql_error_count"],
        "candidate_sql_error_rate": totals["candidate_sql_error_count"] / total if total else 0.0,
        "agg_func_total": subset["agg"]["total"],
        "agg_func_acc_first": (
            subset["agg"]["first_exec_match"] / subset["agg"]["total"]
            if subset["agg"]["total"]
            else None
        ),
        "agg_func_acc_any": (
            subset["agg"]["any_exec_match"] / subset["agg"]["total"]
            if subset["agg"]["total"]
            else None
        ),
        "where_clause_total": subset["where"]["total"],
        "where_clause_acc_first": (
            subset["where"]["first_exec_match"] / subset["where"]["total"]
            if subset["where"]["total"]
            else None
        ),
        "where_clause_acc_any": (
            subset["where"]["any_exec_match"] / subset["where"]["total"]
            if subset["where"]["total"]
            else None
        ),
        "error_type_distribution_first": dict(error_types_first),
        "error_type_distribution_any": dict(error_types_any),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions_path)
    questions_path = Path(args.questions_path)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    label = args.label or predictions_path.stem

    summary = evaluate_prediction_file(
        predictions_path=predictions_path,
        questions_path=questions_path,
        db_path=db_path,
        output_dir=output_dir,
        label=label,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
