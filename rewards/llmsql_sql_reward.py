"""Execution-based reward for LLMSQL on verl."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from llmsql.utils.evaluation_utils import connect_sqlite, execute_sql, fix_table_name
from llmsql.utils.regex_extractor import find_sql


_CONNECTION_CACHE: dict[str, Any] = {}


def _get_conn(db_path: str):
    resolved = str(Path(db_path).expanduser().resolve())
    conn = _CONNECTION_CACHE.get(resolved)
    if conn is None:
        conn = connect_sqlite(resolved)
        _CONNECTION_CACHE[resolved] = conn
    return conn


def _normalize_result(result: Any) -> list[tuple[Any, ...]] | None:
    if result is None:
        return None
    if isinstance(result, str):
        result = json.loads(result)
    normalized: list[tuple[Any, ...]] = []
    for row in result:
        if isinstance(row, tuple):
            normalized.append(row)
        else:
            normalized.append(tuple(row))
    return normalized


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    db_path: str = "/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db",
    exact_match_reward: float = 1.0,
    executable_reward: float = 0.0,
    format_reward: float = 0.0,
) -> float:
    """Reward exact execution match, with optional shaping for parseable SQL."""
    if extra_info is None:
        return 0.0

    table_id = extra_info.get("table_id")
    if not table_id:
        return 0.0

    predicted_queries = find_sql(solution_str)
    if not predicted_queries:
        return 0.0

    conn = _get_conn(db_path)
    gold_result = _normalize_result(extra_info.get("gold_result_json"))
    if gold_result is None:
        gold_result = execute_sql(conn, ground_truth)

    best_reward = 0.0
    for predicted_sql in predicted_queries:
        fixed_sql = fix_table_name(predicted_sql, table_id)
        pred_result = execute_sql(conn, fixed_sql)
        if pred_result is None:
            best_reward = max(best_reward, format_reward)
            continue
        if gold_result is not None and pred_result == gold_result:
            return exact_match_reward
        best_reward = max(best_reward, executable_reward)

    return best_reward
