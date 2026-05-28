#!/usr/bin/env python3
"""Run a 2-step LLMSQL agent loop against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp

from llmsql import evaluate
from llmsql.utils.evaluation_utils import connect_sqlite, fix_table_name
from llmsql.utils.regex_extractor import find_sql
from llmsql.utils.utils import choose_prompt_builder, load_jsonl


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class RateGate:
    """Space request starts to avoid overdriving a local or remote server."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._next_allowed = (
                asyncio.get_running_loop().time() + self.min_interval_seconds
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2-step LLMSQL execution+verifier feedback agent loop.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name exposed by the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the API key. Use a dummy key for local vLLM if needed.",
    )
    parser.add_argument(
        "--questions-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/test_questions.jsonl",
        help="Path to LLMSQL questions JSONL.",
    )
    parser.add_argument(
        "--tables-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/tables.jsonl",
        help="Path to LLMSQL tables JSONL.",
    )
    parser.add_argument(
        "--db-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db",
        help="Path to LLMSQL sqlite db for execution and official evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save predictions, traces, and reports.",
    )
    parser.add_argument(
        "--num-fewshots",
        type=int,
        default=5,
        choices=[0, 1, 5],
        help="Official LLMSQL prompt variant.",
    )
    parser.add_argument(
        "--loop-mode",
        default="verify_incorrect",
        choices=["strict_runtime", "verify_incorrect"],
        help="When to trigger the second pass.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests to the server.",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=0.0,
        help="Minimum interval between request starts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max completion tokens per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retries for transient API failures.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds for exponential retry backoff.",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help="Optional limit. Integer-like values mean sample count; floats in (0,1] mean fraction.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing trace and prediction files if present.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip official evaluation reports after generation.",
    )
    return parser.parse_args()


def normalize_limit(raw_limit: str | None) -> int | float | None:
    if raw_limit is None:
        return None
    text = raw_limit.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    value = float(text)
    if value > 1 and value.is_integer():
        return int(value)
    return value


def prepare_questions(
    questions_path: Path,
    limit: int | float | None,
) -> list[dict[str, Any]]:
    questions = load_jsonl(str(questions_path))
    if limit is None:
        return questions
    if isinstance(limit, float):
        if not (0.0 < limit <= 1.0):
            raise ValueError(f"Fractional --limit must be in (0, 1], got {limit}.")
        limit = max(1, int(len(questions) * limit))
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"Invalid --limit value: {limit!r}")
    return questions[:limit]


def load_tables(tables_path: Path) -> dict[str, dict[str, Any]]:
    tables = load_jsonl(str(tables_path))
    return {table["table_id"]: table for table in tables}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_message_content(data: dict[str, Any]) -> str:
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        chunks: list[str] = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "".join(chunks)
    return str(message)


def validate_response_payload(data: dict[str, Any]) -> str | None:
    if not isinstance(data, dict):
        return f"non-dict response payload: {type(data).__name__}"
    if "error" in data:
        return f"error field present: {json.dumps(data, ensure_ascii=False)[:500]}"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return f"missing/invalid choices: {json.dumps(data, ensure_ascii=False)[:500]}"
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return f"invalid first choice: {json.dumps(data, ensure_ascii=False)[:500]}"
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return f"missing/invalid message: {json.dumps(data, ensure_ascii=False)[:500]}"
    if "content" not in message:
        return f"missing message content: {json.dumps(data, ensure_ascii=False)[:500]}"
    return None


async def request_with_retries(
    *,
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    request_label: str,
    rate_gate: RateGate,
    max_retries: int,
    retry_base_delay: float,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        await rate_gate.wait()
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                body = await resp.text()
                if resp.status == 200:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as exc:
                        last_error = f"invalid json body: {exc} raw={body[:500]}"
                    else:
                        payload_error = validate_response_payload(data)
                        if payload_error is None:
                            return data
                        last_error = payload_error
                elif resp.status in {408, 409, 429, 500, 502, 503, 504}:
                    last_error = f"status={resp.status} body={body[:500]}"
                else:
                    raise RuntimeError(
                        f"API request failed for {request_label} "
                        f"with status={resp.status}: {body[:500]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt == max_retries:
            break

        backoff = retry_base_delay * (2 ** (attempt - 1))
        backoff += random.uniform(0, retry_base_delay)
        print(
            f"Retrying {request_label} attempt={attempt}/{max_retries} "
            f"after error: {last_error}"
        )
        await asyncio.sleep(backoff)

    raise RuntimeError(f"API request exhausted retries for {request_label}: {last_error}")


def execute_sql_with_error(
    conn: sqlite3.Connection, sql: str
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return sorted(cur.fetchall()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def safe_preview(value: Any, max_len: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def inspect_completion(
    *,
    completion: str,
    table_id: str,
    conn: sqlite3.Connection,
    gold_result: list[tuple[Any, ...]],
) -> dict[str, Any]:
    extracted_sqls = find_sql(completion)
    if not extracted_sqls:
        return {
            "status": "no_sql_extracted",
            "raw_sql": None,
            "fixed_sql": None,
            "sql_executable": False,
            "exec_match": False,
            "sql_error_message": None,
            "pred_result_preview": None,
            "num_extracted_sqls": 0,
        }

    raw_sql = extracted_sqls[0]
    fixed_sql = fix_table_name(raw_sql, table_id)
    pred_result, pred_error = execute_sql_with_error(conn, fixed_sql)
    if pred_error is not None:
        return {
            "status": "execution_error",
            "raw_sql": raw_sql,
            "fixed_sql": fixed_sql,
            "sql_executable": False,
            "exec_match": False,
            "sql_error_message": pred_error,
            "pred_result_preview": None,
            "num_extracted_sqls": len(extracted_sqls),
        }

    exec_match = pred_result == gold_result
    return {
        "status": "correct" if exec_match else "verification_incorrect",
        "raw_sql": raw_sql,
        "fixed_sql": fixed_sql,
        "sql_executable": True,
        "exec_match": exec_match,
        "sql_error_message": None,
        "pred_result_preview": safe_preview(pred_result),
        "num_extracted_sqls": len(extracted_sqls),
    }


def should_trigger_repair(status: str, loop_mode: str) -> bool:
    if loop_mode == "strict_runtime":
        return status in {"no_sql_extracted", "execution_error"}
    if loop_mode == "verify_incorrect":
        return status != "correct"
    raise ValueError(f"Unsupported loop mode: {loop_mode}")


def build_repair_feedback(first_inspection: dict[str, Any]) -> str:
    status = first_inspection["status"]
    if status == "no_sql_extracted":
        return (
            "Your previous response did not contain a valid SQL SELECT query. "
            "Return exactly one corrected SQL SELECT query."
        )
    if status == "execution_error":
        return (
            "Your previous SQL failed to execute.\n"
            f"SQLite error: {first_inspection['sql_error_message']}\n"
            "Fix the SQL and return exactly one corrected SQL SELECT query."
        )
    if status == "verification_incorrect":
        return (
            "Your previous SQL executed successfully, but the returned result did not answer "
            "the question correctly. Re-check the selected columns, predicates, and any "
            "aggregation. Return exactly one corrected SQL SELECT query."
        )
    return "Return exactly one SQL SELECT query."


def build_repair_prompt(
    *,
    question: str,
    columns: list[Any],
    types: list[Any],
    sample_row: list[Any],
    previous_sql: str | None,
    feedback: str,
) -> str:
    return (
        "You are repairing a SQLite SQL query.\n"
        "Output ONLY one valid SQL SELECT query.\n"
        "Use table name \"Table\".\n"
        "Allowed functions: ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']\n"
        "Allowed condition operators: ['=', '>', '<', '!=']\n"
        "Allowed SQL keywords: ['SELECT', 'WHERE', 'AND']\n"
        "Always use double quotes around all column names and the table name.\n\n"
        f"Question: {question}\n"
        f"Columns: {columns}\n"
        f"Types: {types}\n"
        f"Sample row: {sample_row}\n"
        f"Previous SQL: {previous_sql or '<none>'}\n"
        f"Feedback: {feedback}\n\n"
        "Corrected SQL:"
    )


def read_completed_question_ids(trace_path: Path) -> set[int]:
    return {int(row["question_id"]) for row in read_jsonl(trace_path)}


def compute_counters_from_traces(trace_path: Path) -> dict[str, int]:
    counters: Counter[str] = Counter()
    for row in read_jsonl(trace_path):
        first_inspection = row["first_inspection"]
        final_inspection = row["final_inspection"]
        second_inspection = row.get("second_inspection")
        loop_triggered = bool(row.get("loop_triggered"))

        counters["first_total"] += 1
        counters[f"first_status:{first_inspection['status']}"] += 1
        counters[f"final_status:{final_inspection['status']}"] += 1
        counters["final_correct"] += int(final_inspection["exec_match"])

        if loop_triggered:
            counters["repair_attempts"] += 1
            if second_inspection is not None:
                counters[f"repair_status:{second_inspection['status']}"] += 1
                counters[f"repair_from:{first_inspection['status']}"] += 1
                if second_inspection["exec_match"] and not first_inspection["exec_match"]:
                    counters["repair_successes"] += 1
                    counters[f"repair_success_from:{first_inspection['status']}"] += 1
                if first_inspection["exec_match"] and not second_inspection["exec_match"]:
                    counters["correct_to_wrong"] += 1

    return dict(counters)


def compute_loop_metrics(
    counters: dict[str, int],
    first_report: dict[str, Any] | None = None,
    final_report: dict[str, Any] | None = None,
) -> dict[str, float]:
    first_total = counters.get("first_total", 0)
    repair_attempts = counters.get("repair_attempts", 0)
    repair_successes = counters.get("repair_successes", 0)
    first_correct = counters.get("first_status:correct", 0)
    correct_to_wrong = counters.get("correct_to_wrong", 0)
    verification_attempts = counters.get("repair_from:verification_incorrect", 0)
    execution_attempts = counters.get("repair_from:execution_error", 0)
    no_sql_attempts = counters.get("repair_from:no_sql_extracted", 0)

    metrics: dict[str, float] = {
        "repair_attempt_rate": repair_attempts / first_total if first_total else 0.0,
        "repair_success_rate": repair_successes / repair_attempts if repair_attempts else 0.0,
        "verification_repair_gain": (
            counters.get("repair_success_from:verification_incorrect", 0) / verification_attempts
            if verification_attempts
            else 0.0
        ),
        "execution_repair_gain": (
            counters.get("repair_success_from:execution_error", 0) / execution_attempts
            if execution_attempts
            else 0.0
        ),
        "no_sql_repair_gain": (
            counters.get("repair_success_from:no_sql_extracted", 0) / no_sql_attempts
            if no_sql_attempts
            else 0.0
        ),
        "correct_to_wrong_rate": correct_to_wrong / first_correct if first_correct else 0.0,
    }

    if first_report is not None and final_report is not None:
        metrics["first_to_final_exec_match_gain"] = (
            final_report["accuracy"] - first_report["accuracy"]
        )

    return metrics


async def generate_agent_predictions(
    *,
    questions: list[dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    db_path: Path,
    prompt_builder: Any,
    first_pred_path: Path,
    final_pred_path: Path,
    trace_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    max_concurrency: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    min_request_interval: float,
    loop_mode: str,
    resume: bool,
) -> dict[str, int]:
    completed_question_ids = read_completed_question_ids(trace_path) if resume else set()
    pending_questions = [
        question
        for question in questions
        if int(question["question_id"]) not in completed_question_ids
    ]

    if not resume:
        first_pred_path.write_text("", encoding="utf-8")
        final_pred_path.write_text("", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")

    if not pending_questions:
        print(f"Agent loop outputs already complete: {trace_path}")
        return compute_counters_from_traces(trace_path)

    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    rate_gate = RateGate(min_request_interval)
    counters: Counter[str] = Counter()
    conn = connect_sqlite(str(db_path))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url.rstrip('/')}/chat/completions"

    async def generate_one(
        session: aiohttp.ClientSession,
        question_item: dict[str, Any],
    ) -> None:
        async with semaphore:
            question_id = int(question_item["question_id"])
            table = tables[question_item["table_id"]]
            sample_row = table["rows"][0] if table["rows"] else []
            gold_result, gold_error = execute_sql_with_error(conn, question_item["sql"])
            if gold_error is not None or gold_result is None:
                raise RuntimeError(
                    f"Gold SQL failed for question_id={question_id}: {gold_error}"
                )

            first_prompt = prompt_builder(
                question_item["question"],
                table["header"],
                table["types"],
                sample_row,
            )
            first_payload = {
                "model": model,
                "messages": [{"role": "user", "content": first_prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
            }
            first_data = await request_with_retries(
                session=session,
                url=url,
                payload=first_payload,
                timeout=timeout,
                request_label=f"question_id={question_id}:first_pass",
                rate_gate=rate_gate,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
            first_completion = extract_message_content(first_data)
            first_inspection = inspect_completion(
                completion=first_completion,
                table_id=question_item["table_id"],
                conn=conn,
                gold_result=gold_result,
            )

            loop_triggered = should_trigger_repair(first_inspection["status"], loop_mode)
            second_completion = None
            second_inspection = None
            final_completion = first_completion
            final_inspection = first_inspection

            counters["first_total"] += 1
            counters[f"first_status:{first_inspection['status']}"] += 1

            if loop_triggered:
                counters["repair_attempts"] += 1
                repair_feedback = build_repair_feedback(first_inspection)
                repair_prompt = build_repair_prompt(
                    question=question_item["question"],
                    columns=table["header"],
                    types=table["types"],
                    sample_row=sample_row,
                    previous_sql=first_inspection["raw_sql"],
                    feedback=repair_feedback,
                )
                second_payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": repair_prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_new_tokens,
                }
                second_data = await request_with_retries(
                    session=session,
                    url=url,
                    payload=second_payload,
                    timeout=timeout,
                    request_label=f"question_id={question_id}:repair_pass",
                    rate_gate=rate_gate,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                )
                second_completion = extract_message_content(second_data)
                second_inspection = inspect_completion(
                    completion=second_completion,
                    table_id=question_item["table_id"],
                    conn=conn,
                    gold_result=gold_result,
                )
                final_completion = second_completion
                final_inspection = second_inspection
                counters[f"repair_status:{second_inspection['status']}"] += 1
                if second_inspection["exec_match"] and not first_inspection["exec_match"]:
                    counters["repair_successes"] += 1
                if first_inspection["exec_match"] and not second_inspection["exec_match"]:
                    counters["correct_to_wrong"] += 1

            counters[f"final_status:{final_inspection['status']}"] += 1
            counters["final_correct"] += int(final_inspection["exec_match"])

            first_record = {
                "question_id": question_id,
                "completion": first_completion,
            }
            final_record = {
                "question_id": question_id,
                "completion": final_completion,
            }
            trace_record = {
                "question_id": question_id,
                "table_id": question_item["table_id"],
                "question": question_item["question"],
                "loop_mode": loop_mode,
                "first_completion": first_completion,
                "first_inspection": first_inspection,
                "loop_triggered": loop_triggered,
                "second_completion": second_completion,
                "second_inspection": second_inspection,
                "final_completion_source": "second_pass" if loop_triggered else "first_pass",
                "final_inspection": final_inspection,
            }

            async with write_lock:
                with first_pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(first_record, ensure_ascii=False) + "\n")
                with final_pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    print(
        f"Running agent loop on {len(pending_questions)} / {len(questions)} samples "
        f"with model={model}, loop_mode={loop_mode}, resume={resume}"
    )
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [generate_one(session, question_item) for question_item in pending_questions]
        for future in asyncio.as_completed(tasks):
            await future

    conn.close()
    return dict(counters)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "EMPTY")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = normalize_limit(args.limit)
    suffix = "full" if limit is None else f"limit_{str(limit).replace('.', 'p')}"
    stem = f"{args.num_fewshots}shot_{suffix}_{args.loop_mode}"
    first_pred_path = output_dir / f"preds_first_{stem}.jsonl"
    final_pred_path = output_dir / f"preds_final_{stem}.jsonl"
    trace_path = output_dir / f"agent_traces_{stem}.jsonl"
    first_report_path = output_dir / f"report_first_{stem}.json"
    final_report_path = output_dir / f"report_final_{stem}.json"
    meta_path = output_dir / f"meta_{stem}.json"

    questions = prepare_questions(Path(args.questions_path), limit)
    tables = load_tables(Path(args.tables_path))
    prompt_builder = choose_prompt_builder(args.num_fewshots)

    print("Starting LLMSQL agent loop")
    print(f"model={args.model}")
    print(f"base_url={args.base_url}")
    print(f"questions={len(questions)} output_dir={output_dir}")
    print(f"first_pred_path={first_pred_path}")
    print(f"final_pred_path={final_pred_path}")

    counters = asyncio.run(
        generate_agent_predictions(
            questions=questions,
            tables=tables,
            db_path=Path(args.db_path),
            prompt_builder=prompt_builder,
            first_pred_path=first_pred_path,
            final_pred_path=final_pred_path,
            trace_path=trace_path,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            max_concurrency=args.max_concurrency,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            min_request_interval=args.min_request_interval,
            loop_mode=args.loop_mode,
            resume=args.resume,
        )
    )

    meta: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "questions_path": args.questions_path,
        "tables_path": args.tables_path,
        "db_path": args.db_path,
        "num_fewshots": args.num_fewshots,
        "loop_mode": args.loop_mode,
        "max_concurrency": args.max_concurrency,
        "min_request_interval": args.min_request_interval,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "limit": limit,
        "resume": args.resume,
        "first_pred_path": str(first_pred_path),
        "final_pred_path": str(final_pred_path),
        "trace_path": str(trace_path),
        "counters": counters,
    }

    if not args.no_eval:
        first_report = evaluate(
            str(first_pred_path),
            questions_path=args.questions_path,
            db_path=args.db_path,
            save_report=str(first_report_path),
            show_mismatches=True,
            max_mismatches=5,
        )
        final_report = evaluate(
            str(final_pred_path),
            questions_path=args.questions_path,
            db_path=args.db_path,
            save_report=str(final_report_path),
            show_mismatches=True,
            max_mismatches=5,
        )
        meta.update(
            {
                "first_report_path": str(first_report_path),
                "final_report_path": str(final_report_path),
                "first_accuracy": first_report["accuracy"],
                "first_matches": first_report["matches"],
                "first_sql_errors": first_report["sql_errors"],
                "final_accuracy": final_report["accuracy"],
                "final_matches": final_report["matches"],
                "final_sql_errors": final_report["sql_errors"],
                "loop_metrics": compute_loop_metrics(
                    counters,
                    first_report=first_report,
                    final_report=final_report,
                ),
            }
        )
    else:
        meta["loop_metrics"] = compute_loop_metrics(counters)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
