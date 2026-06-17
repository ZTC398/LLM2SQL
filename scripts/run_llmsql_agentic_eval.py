#!/usr/bin/env python3
"""Run official LLMSQL evaluation for the tool-only agentic model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from llmsql import evaluate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from llmsql_agentic.tool_env import LLMSQLToolEnv  # noqa: E402


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_DATASET_DIR = os.environ.get(
    "LLMSQL_DATASET_DIR",
    "/root/shared-nvme/rlvr/datasets/llmsql-2.0",
)
DEFAULT_AGENT_PARQUET = os.environ.get(
    "LLMSQL_AGENT_DATA_PATH",
    "/root/shared-nvme/rlvr/verl_data/llmsql_agent_0shot/test.parquet",
)
DEFAULT_QUESTIONS_PATH = str(Path(DEFAULT_DATASET_DIR) / "test_questions.jsonl")
DEFAULT_DB_PATH = str(Path(DEFAULT_DATASET_DIR) / "sqlite_tables.db")


class RateGate:
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
            self._next_allowed = asyncio.get_running_loop().time() + self.min_interval_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tool-only LLMSQL agentic evaluation against an OpenAI-compatible server."
    )
    parser.add_argument("--model", required=True, help="Model name exposed by the API server.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the API key. Dummy key is fine for local vLLM.",
    )
    parser.add_argument(
        "--agent-parquet",
        default=DEFAULT_AGENT_PARQUET,
        help="Agent parquet containing prompts and extra_info for evaluation.",
    )
    parser.add_argument(
        "--questions-path",
        default=DEFAULT_QUESTIONS_PATH,
        help="Official LLMSQL questions JSONL used by official evaluator.",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="SQLite DB used both for tool execution and official evaluation.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save predictions and reports.")
    parser.add_argument("--max-concurrency", type=int, default=32, help="Max concurrent API requests.")
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=0.0,
        help="Minimum interval between request starts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per assistant turn.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=6, help="Retry count for transient API failures.")
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
    parser.add_argument("--max-tool-turns", type=int, default=2, help="Max <sql> tool turns before forced final.")
    parser.add_argument("--preview-rows", type=int, default=5, help="Rows shown in execution previews.")
    parser.add_argument(
        "--preview-max-chars",
        type=int,
        default=2000,
        help="Max rendered chars for execution previews.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing trajectory file if present.")
    parser.add_argument("--no-eval", action="store_true", help="Skip official evaluate after generation.")
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


def apply_limit(rows: list[dict[str, Any]], limit: int | float | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if isinstance(limit, float):
        if not (0.0 < limit <= 1.0):
            raise ValueError(f"Fractional --limit must be in (0, 1], got {limit}.")
        limit = max(1, int(len(rows) * limit))
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"Invalid --limit value: {limit!r}")
    return rows[:limit]


def load_agent_rows(agent_parquet: Path, limit: int | float | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(agent_parquet)
    rows = df.to_dict(orient="records")
    return apply_limit(rows, limit)


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


def read_completed_question_ids(trajectory_path: Path) -> set[int]:
    return {int(row["question_id"]) for row in read_jsonl(trajectory_path)}


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def generate_predictions(
    *,
    rows: list[dict[str, Any]],
    db_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    output_dir: Path,
    max_concurrency: int,
    min_request_interval: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    max_tool_turns: int,
    preview_rows: int,
    preview_max_chars: int,
    resume: bool,
) -> dict[str, Any]:
    first_pred_path = output_dir / "preds_first_agent_full.jsonl"
    trajectory_pred_path = output_dir / "preds_trajectory_agent_full.jsonl"
    trace_path = output_dir / "agentic_traces_full.jsonl"

    completed_question_ids = read_completed_question_ids(trace_path) if resume else set()
    pending_rows = [
        row for row in rows if int(row["extra_info"]["question_id"]) not in completed_question_ids
    ]

    if not resume:
        first_pred_path.write_text("", encoding="utf-8")
        trajectory_pred_path.write_text("", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")

    if not pending_rows:
        return {
            "first_pred_path": str(first_pred_path),
            "trajectory_pred_path": str(trajectory_pred_path),
            "trace_path": str(trace_path),
            "generated_count": 0,
            "resumed": True,
        }

    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    rate_gate = RateGate(min_request_interval)
    env = LLMSQLToolEnv(
        db_path=str(db_path),
        preview_rows=preview_rows,
        preview_max_chars=preview_max_chars,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    counters: Counter[str] = Counter()

    async def generate_one(
        session: aiohttp.ClientSession,
        row: dict[str, Any],
    ) -> None:
        async with semaphore:
            extra_info = row["extra_info"]
            question_id = int(extra_info["question_id"])
            table_id = extra_info["table_id"]
            messages = list(row["prompt"])
            assistant_texts: list[str] = []
            trace: list[dict[str, Any]] = []
            first_completion: str | None = None
            tool_turns = 0
            forced_final_turn = False
            assistant_turns = 0

            while True:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_new_tokens,
                }
                data = await request_with_retries(
                    session=session,
                    url=url,
                    payload=payload,
                    timeout=timeout,
                    request_label=f"question_id={question_id}:turn={assistant_turns + 1}",
                    rate_gate=rate_gate,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                )
                completion = extract_message_content(data)
                if first_completion is None:
                    first_completion = completion

                assistant_turns += 1
                assistant_texts.append(completion)

                if forced_final_turn:
                    parsed = env.parse_action(completion)
                    trace.append(
                        {
                            "assistant_text": completion,
                            "action_type": parsed.action_type,
                            "action_sql": parsed.sql,
                            "observation_text": None,
                        }
                    )
                    break

                step_result = env.step(
                    action_text=completion,
                    table_id=table_id,
                    final_turn=False,
                )
                trace.append(
                    {
                        "assistant_text": completion,
                        "action_type": step_result.action_type,
                        "action_sql": step_result.sql,
                        "observation_text": step_result.observation_text or None,
                    }
                )

                if step_result.is_final:
                    break

                force_final_after_obs = False
                if step_result.action_type == "sql":
                    tool_turns += 1
                    if tool_turns >= max_tool_turns:
                        force_final_after_obs = True

                if assistant_turns >= max_tool_turns + 1:
                    break

                observation_text = step_result.observation_text
                if not observation_text:
                    break

                if force_final_after_obs:
                    observation_text = (
                        f"{observation_text}\n\n"
                        "This is your final turn. Output <final_sql> ... </final_sql> now."
                    )
                    forced_final_turn = True

                messages.append({"role": "assistant", "content": completion})
                messages.append({"role": "user", "content": observation_text})

            final_completion = "\n".join(assistant_texts)
            first_record = {
                "question_id": question_id,
                "completion": first_completion or "",
            }
            trajectory_record = {
                "question_id": question_id,
                "completion": final_completion,
            }
            trace_record = {
                "question_id": question_id,
                "table_id": table_id,
                "question": extra_info["question"],
                "tool_turns": tool_turns,
                "assistant_turns": assistant_turns,
                "trace": trace,
            }

            counters["generated"] += 1
            counters["forced_final_turn_count"] += int(forced_final_turn)

            async with write_lock:
                with first_pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(first_record, ensure_ascii=False) + "\n")
                with trajectory_pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trajectory_record, ensure_ascii=False) + "\n")
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    print(
        f"Running agentic eval on {len(pending_rows)} / {len(rows)} samples "
        f"with model={model}, resume={resume}"
    )
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [generate_one(session, row) for row in pending_rows]
        for future in asyncio.as_completed(tasks):
            await future

    return {
        "first_pred_path": str(first_pred_path),
        "trajectory_pred_path": str(trajectory_pred_path),
        "trace_path": str(trace_path),
        "generated_count": counters["generated"],
        "forced_final_turn_count": counters["forced_final_turn_count"],
        "resumed": resume,
    }


def run_export_script(trajectory_pred_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_llmsql_agent_final_sql.py"),
        "--input-path",
        str(trajectory_pred_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(result.stdout)
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "EMPTY")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = normalize_limit(args.limit)
    rows = load_agent_rows(Path(args.agent_parquet), limit)

    generation_meta = asyncio.run(
        generate_predictions(
            rows=rows,
            db_path=Path(args.db_path),
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            output_dir=output_dir,
            max_concurrency=args.max_concurrency,
            min_request_interval=args.min_request_interval,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            max_tool_turns=args.max_tool_turns,
            preview_rows=args.preview_rows,
            preview_max_chars=args.preview_max_chars,
            resume=args.resume,
        )
    )

    trajectory_pred_path = Path(generation_meta["trajectory_pred_path"])
    first_pred_path = Path(generation_meta["first_pred_path"])
    export_summary = run_export_script(trajectory_pred_path)
    final_sql_pred_path = Path(export_summary["output_path"])

    meta: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "agent_parquet": args.agent_parquet,
        "questions_path": args.questions_path,
        "db_path": args.db_path,
        "limit": limit,
        "max_tool_turns": args.max_tool_turns,
        "max_concurrency": args.max_concurrency,
        "min_request_interval": args.min_request_interval,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "resume": args.resume,
        **generation_meta,
        "final_sql_export_summary": export_summary,
    }

    if not args.no_eval:
        first_report_path = output_dir / "report_first_agent_full.json"
        final_report_path = output_dir / "report_final_sql_agent_full.json"
        first_report = evaluate(
            str(first_pred_path),
            questions_path=args.questions_path,
            db_path=args.db_path,
            save_report=str(first_report_path),
            show_mismatches=True,
            max_mismatches=5,
        )
        final_report = evaluate(
            str(final_sql_pred_path),
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
                "gain": final_report["accuracy"] - first_report["accuracy"],
            }
        )

    meta_path = output_dir / "meta_agent_full.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
