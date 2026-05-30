"""SQLite-backed environment for LLMSQL tool-only agent loops."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmsql.utils.evaluation_utils import connect_sqlite, fix_table_name

from llmsql_agentic.prompting import (
    INVALID_ACTION_TEXT,
    format_execution_error_observation,
    format_execution_result_observation,
)


_SQL_ACTION_RE = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.DOTALL | re.IGNORECASE)
_FINAL_SQL_ACTION_RE = re.compile(
    r"<final_sql>\s*(.*?)\s*</final_sql>", re.DOTALL | re.IGNORECASE
)


@dataclass
class ParsedAction:
    action_type: str
    sql: str | None = None


@dataclass
class ToolStepResult:
    observation_text: str
    sql: str | None
    action_type: str
    is_final: bool
    is_valid: bool


class LLMSQLToolEnv:
    def __init__(
        self,
        *,
        db_path: str,
        preview_rows: int = 5,
        preview_max_chars: int = 2000,
    ) -> None:
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.preview_rows = preview_rows
        self.preview_max_chars = preview_max_chars
        self.conn = connect_sqlite(self.db_path)

    def parse_action(self, action_text: str) -> ParsedAction:
        sql_matches = _SQL_ACTION_RE.findall(action_text)
        final_matches = _FINAL_SQL_ACTION_RE.findall(action_text)
        total_matches = len(sql_matches) + len(final_matches)

        if total_matches != 1:
            return ParsedAction(action_type="invalid", sql=None)
        if sql_matches:
            return ParsedAction(action_type="sql", sql=sql_matches[0].strip())
        return ParsedAction(action_type="final_sql", sql=final_matches[0].strip())

    def step(
        self,
        *,
        action_text: str,
        table_id: str,
        final_turn: bool = False,
    ) -> ToolStepResult:
        parsed = self.parse_action(action_text)
        if parsed.action_type == "invalid":
            observation_text = INVALID_ACTION_TEXT
            if final_turn:
                observation_text = (
                    f"{observation_text}\n\nThis is your final turn. Output <final_sql> ... </final_sql> now."
                )
            return ToolStepResult(
                observation_text=observation_text,
                sql=None,
                action_type="invalid",
                is_final=False,
                is_valid=False,
            )

        if parsed.action_type == "final_sql":
            return ToolStepResult(
                observation_text="",
                sql=parsed.sql,
                action_type="final_sql",
                is_final=True,
                is_valid=True,
            )

        assert parsed.sql is not None
        fixed_sql = fix_table_name(parsed.sql, table_id)
        columns, rows, error_message = self._execute_sql(fixed_sql)
        if error_message is not None:
            return ToolStepResult(
                observation_text=format_execution_error_observation(
                    error_message,
                    final_turn=final_turn,
                ),
                sql=parsed.sql,
                action_type="sql",
                is_final=False,
                is_valid=True,
            )

        preview_rows = self._truncate_rows(rows)
        return ToolStepResult(
            observation_text=format_execution_result_observation(
                columns=columns,
                row_count=len(rows),
                rows=preview_rows,
                final_turn=final_turn,
            ),
            sql=parsed.sql,
            action_type="sql",
            is_final=False,
            is_valid=True,
        )

    def _execute_sql(
        self,
        sql: str,
    ) -> tuple[list[str], list[list[Any]], str | None]:
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            description = cur.description or []
            columns = [item[0] for item in description]
            rows = [list(row) for row in cur.fetchall()]
            return columns, rows, None
        except Exception as exc:  # noqa: BLE001
            return [], [], f"{type(exc).__name__}: {exc}"

    def _truncate_rows(self, rows: list[list[Any]]) -> list[list[Any]]:
        limited = rows[: self.preview_rows]
        rendered = json.dumps(limited, ensure_ascii=False)
        if len(rendered) <= self.preview_max_chars:
            return limited

        shrunk: list[list[Any]] = []
        for row in limited:
            new_row: list[Any] = []
            for value in row:
                if isinstance(value, str) and len(value) > 120:
                    new_row.append(value[:117] + "...")
                else:
                    new_row.append(value)
            shrunk.append(new_row)
        return shrunk

