"""Prompt and observation helpers for LLMSQL tool-only agent loops."""

from __future__ import annotations

import json
from typing import Any


INVALID_ACTION_TEXT = (
    "<observation>\n"
    "<invalid_action>\n"
    "You must output exactly one action:\n"
    "1. <sql> SELECT ... </sql>\n"
    "2. <final_sql> SELECT ... </final_sql>\n"
    "</invalid_action>\n"
    "</observation>"
)


FINAL_TURN_REMINDER = "This is your final turn. Output <final_sql> ... </final_sql> now."


def format_execution_error_observation(error_message: str, *, final_turn: bool = False) -> str:
    text = (
        "<observation>\n"
        f"<execution_error>{error_message}</execution_error>\n"
        "</observation>"
    )
    if final_turn:
        text += f"\n\n{FINAL_TURN_REMINDER}"
    return text


def format_execution_result_observation(
    *,
    columns: list[str],
    row_count: int,
    rows: list[list[Any]],
    final_turn: bool = False,
) -> str:
    body = (
        "<observation>\n"
        "<execution_result>\n"
        f"columns: {json.dumps(columns, ensure_ascii=False)}\n"
        f"row_count: {row_count}\n"
        "rows:\n"
    )
    if rows:
        body += "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    else:
        body += "[]"
    body += "\n</execution_result>\n</observation>"
    if final_turn:
        body += f"\n\n{FINAL_TURN_REMINDER}"
    return body

