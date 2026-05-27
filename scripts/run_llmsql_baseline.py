#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from llmsql import evaluate, inference_vllm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLMSQL baseline inference with vLLM and evaluate results."
    )
    parser.add_argument(
        "--model-path",
        default="/root/shared-nvme/rlvr/models/Qwen2.5-Coder-3B-Instruct",
        help="Local model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--questions-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/test_questions.jsonl",
        help="Path to LLMSQL questions JSONL. Use test split for benchmark eval.",
    )
    parser.add_argument(
        "--tables-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/tables.jsonl",
        help="Path to LLMSQL tables JSONL.",
    )
    parser.add_argument(
        "--db-path",
        default="/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db",
        help="Path to LLMSQL sqlite db used by official evaluator.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/shared-nvme/rlvr/outputs/baseline_qwen25_coder_3b",
        help="Directory to save predictions and evaluation reports.",
    )
    parser.add_argument(
        "--num-fewshots",
        type=int,
        default=5,
        choices=[0, 1, 5],
        help="Official LLMSQL prompt variant.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="vLLM batch size.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for deterministic baseline.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional limit. Integer-like values mean sample count; floats in (0,1] mean fraction.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used by vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force vLLM eager mode to avoid graph compilation issues.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional vLLM gpu_memory_utilization override.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing preds_*.jsonl in output_dir if present.",
    )
    return parser.parse_args()


def normalize_limit(raw_limit: float | None) -> int | float | None:
    if raw_limit is None:
        return None
    if raw_limit > 1 and raw_limit.is_integer():
        return int(raw_limit)
    return raw_limit


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_completed_rows(main_pred_path: Path, resume_chunk_path: Path) -> list[dict]:
    rows: list[dict] = []
    if main_pred_path.exists():
        rows.extend(read_jsonl(main_pred_path))
    if resume_chunk_path.exists():
        rows.extend(read_jsonl(resume_chunk_path))
    return rows


def consolidate_prediction_files(main_pred_path: Path, resume_chunk_path: Path) -> None:
    if not resume_chunk_path.exists():
        return

    merged_rows: list[dict] = []
    seen_question_ids: set[int] = set()
    for row in collect_completed_rows(main_pred_path, resume_chunk_path):
        question_id = row["question_id"]
        if question_id in seen_question_ids:
            continue
        seen_question_ids.add(question_id)
        merged_rows.append(row)

    with main_pred_path.open("w") as out_f:
        for row in merged_rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resume_chunk_path.unlink()


def prepare_resume_questions(
    questions_path: Path,
    resume_questions_path: Path,
    completed_question_ids: set[int],
    limit: int | float | None,
) -> tuple[int, int]:
    with questions_path.open() as f:
        lines = [line for line in f if line.strip()]

    if isinstance(limit, int):
        lines = lines[:limit]
    elif isinstance(limit, float):
        lines = lines[: int(len(lines) * limit)]

    remaining_lines = []
    for line in lines:
        item = json.loads(line)
        if item["question_id"] not in completed_question_ids:
            remaining_lines.append(line)

    resume_questions_path.write_text("".join(remaining_lines))
    return len(lines), len(remaining_lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = normalize_limit(args.limit)
    suffix = "full" if limit is None else f"limit_{str(limit).replace('.', 'p')}"
    pred_path = output_dir / f"preds_{args.num_fewshots}shot_{suffix}.jsonl"
    resume_chunk_path = output_dir / f"{pred_path.stem}_resume_chunk.jsonl"
    report_path = output_dir / f"report_{args.num_fewshots}shot_{suffix}.json"
    meta_path = output_dir / f"meta_{args.num_fewshots}shot_{suffix}.json"

    print("Starting LLMSQL baseline inference")
    print(f"model_path={args.model_path}")
    print(f"questions_path={args.questions_path}")
    print(f"tables_path={args.tables_path}")
    print(f"db_path={args.db_path}")
    print(f"output_dir={output_dir}")
    print(f"num_fewshots={args.num_fewshots} limit={limit}")

    llm_kwargs = {}
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    inference_questions_path = args.questions_path
    inference_pred_path = pred_path
    inference_limit = limit

    if args.resume and pred_path.exists() and resume_chunk_path.exists():
        consolidate_prediction_files(pred_path, resume_chunk_path)
        print(f"Resume mode: merged existing resume chunk into {pred_path}")

    if args.resume and pred_path.exists():
        completed_rows = collect_completed_rows(pred_path, resume_chunk_path)
        completed_question_ids = {row["question_id"] for row in completed_rows}
        resume_questions_path = output_dir / f"questions_{args.num_fewshots}shot_{suffix}_remaining.jsonl"
        chunk_pred_path = resume_chunk_path
        if chunk_pred_path.exists():
            chunk_pred_path.unlink()

        total_count, remaining_count = prepare_resume_questions(
            questions_path=Path(args.questions_path),
            resume_questions_path=resume_questions_path,
            completed_question_ids=completed_question_ids,
            limit=limit,
        )
        print(
            f"Resume mode: found {len(completed_question_ids)} completed predictions, "
            f"{remaining_count}/{total_count} questions remaining"
        )

        if remaining_count > 0:
            inference_questions_path = str(resume_questions_path)
            inference_pred_path = chunk_pred_path
            inference_limit = None
        else:
            print("Resume mode: nothing remaining, skip inference")
            inference_questions_path = None

    if inference_questions_path is not None:
        inference_vllm(
            model_name=args.model_path,
            output_file=str(inference_pred_path),
            questions_path=inference_questions_path,
            tables_path=args.tables_path,
            workdir_path=str(output_dir / "llmsql_workdir"),
            num_fewshots=args.num_fewshots,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            tensor_parallel_size=args.tensor_parallel_size,
            limit=inference_limit,
            seed=args.seed,
            llm_kwargs=llm_kwargs,
        )

    if args.resume and pred_path.exists() and inference_pred_path != pred_path and Path(inference_pred_path).exists():
        with pred_path.open("a") as out_f, Path(inference_pred_path).open() as in_f:
            for line in in_f:
                if line.strip():
                    out_f.write(line)
        print(f"Resume mode: appended new predictions from {inference_pred_path} into {pred_path}")

    report = evaluate(
        str(pred_path),
        questions_path=args.questions_path,
        db_path=args.db_path,
        save_report=str(report_path),
        show_mismatches=True,
        max_mismatches=5,
    )

    meta = {
        "model_path": args.model_path,
        "questions_path": args.questions_path,
        "tables_path": args.tables_path,
        "db_path": args.db_path,
        "pred_path": str(pred_path),
        "report_path": str(report_path),
        "num_fewshots": args.num_fewshots,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "limit": limit,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "resume": args.resume,
        "accuracy": report["accuracy"],
        "matches": report["matches"],
        "total": report["total"],
        "sql_errors": report["sql_errors"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Baseline evaluation finished")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
