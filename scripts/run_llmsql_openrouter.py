#!/usr/bin/env python3
"""Run LLMSQL inference against OpenRouter and optionally evaluate outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from llmsql import evaluate
from llmsql.utils.utils import choose_prompt_builder, load_jsonl


DEFAULT_MODEL = "deepseek/deepseek-v4-flash:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class RateGate:
    """Space request starts for heavily rate-limited free endpoints."""

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
        description="Run LLMSQL benchmark inference with an OpenRouter model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model id.",
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
        help="Path to LLMSQL sqlite db for official evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save predictions and reports.",
    )
    parser.add_argument(
        "--num-fewshots",
        type=int,
        default=5,
        choices=[0, 1, 5],
        help="Prompt variant to use.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent HTTP requests.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max completion tokens.",
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
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help="Optional limit. Integer-like values mean sample count; floats in (0,1] mean fraction.",
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
        default=12,
        help="Maximum retries for 429/5xx/network failures.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=3.0,
        help="Base delay in seconds for exponential retry backoff.",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=2.5,
        help="Minimum interval between request starts.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenRouter API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable that stores the OpenRouter API key.",
    )
    parser.add_argument(
        "--http-referer",
        default="https://local.rlvr",
        help="HTTP-Referer header sent to OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        default="rlvr-llmsql-eval",
        help="X-Title header sent to OpenRouter.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip official evaluator after generation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing prediction file if present.",
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


def read_existing_predictions(pred_path: Path) -> dict[int, dict[str, Any]]:
    existing: dict[int, dict[str, Any]] = {}
    if not pred_path.exists():
        return existing
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing[int(row["question_id"])] = row
    return existing


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


async def request_with_retries(
    *,
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    question_id: int,
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
                    return json.loads(body)
                if resp.status in {429, 500, 502, 503, 504}:
                    last_error = f"status={resp.status} body={body[:500]}"
                else:
                    raise RuntimeError(
                        f"OpenRouter request failed for question_id={question_id} "
                        f"with status={resp.status}: {body[:500]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt == max_retries:
            break

        backoff = retry_base_delay * (2 ** (attempt - 1))
        backoff += random.uniform(0, retry_base_delay)
        print(
            f"Retrying question_id={question_id} attempt={attempt}/{max_retries} "
            f"after error: {last_error}"
        )
        await asyncio.sleep(backoff)

    raise RuntimeError(
        f"OpenRouter request exhausted retries for question_id={question_id}: {last_error}"
    )


async def generate_predictions(
    *,
    questions: list[dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    prompt_builder: Any,
    pred_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    http_referer: str,
    x_title: str,
    max_concurrency: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    min_request_interval: float,
    resume: bool,
) -> None:
    existing = read_existing_predictions(pred_path) if resume else {}
    if resume:
        missing_questions = [
            question
            for question in questions
            if int(question["question_id"]) not in existing
        ]
    else:
        missing_questions = questions
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text("", encoding="utf-8")

    if not missing_questions:
        print(f"Prediction file already complete: {pred_path}")
        return

    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    rate_gate = RateGate(min_request_interval)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer,
        "X-Title": x_title,
    }

    async def one_request(
        session: aiohttp.ClientSession,
        question: dict[str, Any],
    ) -> dict[str, Any]:
        async with semaphore:
            table = tables[question["table_id"]]
            sample_row = table["rows"][0] if table["rows"] else []
            prompt = prompt_builder(
                question["question"],
                table["header"],
                table["types"],
                sample_row,
            )

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
            }

            data = await request_with_retries(
                session=session,
                url=f"{base_url.rstrip('/')}/chat/completions",
                payload=payload,
                timeout=timeout,
                question_id=int(question["question_id"]),
                rate_gate=rate_gate,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )

            content = extract_message_content(data)
            result = {
                "question_id": int(question["question_id"]),
                "completion": content,
            }
            async with write_lock:
                with pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result

    print(
        f"Generating {len(missing_questions)} / {len(questions)} samples with model={model}, "
        f"resume={resume}, concurrency={max_concurrency}, min_interval={min_request_interval}s"
    )
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [one_request(session, question) for question in missing_questions]
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="OpenRouter generating",
        ):
            await future


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please export {args.api_key_env} before running."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = normalize_limit(args.limit)
    suffix = "full" if limit is None else f"limit_{str(limit).replace('.', 'p')}"
    pred_path = output_dir / f"preds_{args.num_fewshots}shot_{suffix}.jsonl"
    report_path = output_dir / f"report_{args.num_fewshots}shot_{suffix}.json"
    meta_path = output_dir / f"meta_{args.num_fewshots}shot_{suffix}.json"

    questions = prepare_questions(Path(args.questions_path), limit)
    tables = load_tables(Path(args.tables_path))
    prompt_builder = choose_prompt_builder(args.num_fewshots)

    print("Starting LLMSQL OpenRouter inference")
    print(f"model={args.model}")
    print(f"questions={len(questions)} output_dir={output_dir}")
    print(f"pred_path={pred_path}")

    asyncio.run(
        generate_predictions(
            questions=questions,
            tables=tables,
            prompt_builder=prompt_builder,
            pred_path=pred_path,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            http_referer=args.http_referer,
            x_title=args.x_title,
            max_concurrency=args.max_concurrency,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            min_request_interval=args.min_request_interval,
            resume=args.resume,
        )
    )

    meta: dict[str, Any] = {
        "model": args.model,
        "questions_path": args.questions_path,
        "tables_path": args.tables_path,
        "db_path": args.db_path,
        "pred_path": str(pred_path),
        "num_fewshots": args.num_fewshots,
        "max_concurrency": args.max_concurrency,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_retries": args.max_retries,
        "retry_base_delay": args.retry_base_delay,
        "min_request_interval": args.min_request_interval,
        "limit": limit,
        "resume": args.resume,
    }

    if not args.no_eval:
        report = evaluate(
            str(pred_path),
            questions_path=args.questions_path,
            db_path=args.db_path,
            save_report=str(report_path),
            show_mismatches=True,
            max_mismatches=5,
        )
        meta.update(
            {
                "report_path": str(report_path),
                "accuracy": report["accuracy"],
                "matches": report["matches"],
                "total": report["total"],
                "sql_errors": report["sql_errors"],
            }
        )

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
