"""Terminal reward for LLMSQL tool-only agent trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llmsql.utils.evaluation_utils import connect_sqlite, execute_sql, fix_table_name


_CONNECTION_CACHE: dict[str, Any] = {}
_FINAL_SQL_RE = re.compile(r"<final_sql>\s*(.*?)\s*</final_sql>", re.DOTALL | re.IGNORECASE)


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


def _extract_last_final_sql(solution_str: str) -> str | None:
    matches = _FINAL_SQL_RE.findall(solution_str)
    if not matches:
        return None
    candidate = matches[-1].strip()
    return candidate or None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    db_path: str = "/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db",
    exact_match_reward: float = 1.0,
) -> float:
    """Score only the last <final_sql> from an agent trajectory."""
    del data_source
    if extra_info is None:
        return 0.0

    table_id = extra_info.get("table_id")
    if not table_id:
        return 0.0

    final_sql = _extract_last_final_sql(solution_str)
    if not final_sql:
        return 0.0

    conn = _get_conn(db_path)
    gold_result = _normalize_result(extra_info.get("gold_result_json"))
    if gold_result is None:
        gold_result = execute_sql(conn, ground_truth)

    fixed_sql = fix_table_name(final_sql, table_id)
    pred_result = execute_sql(conn, fixed_sql)
    if pred_result is None:
        return 0.0
    if gold_result is not None and pred_result == gold_result:
        return exact_match_reward
    return 0.0
