#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from llmsql import evaluate, inference_transformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLMSQL baseline inference with Transformers and evaluate results."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model path or Hugging Face model id.",
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
        help="Path to LLMSQL sqlite db used by official evaluator.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
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
        default=1,
        help="Transformers batch size.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to Transformers model loading.",
    )
    return parser.parse_args()


def normalize_limit(raw_limit: float | None) -> int | float | None:
    if raw_limit is None:
        return None
    if raw_limit > 1 and raw_limit.is_integer():
        return int(raw_limit)
    return raw_limit


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = normalize_limit(args.limit)
    suffix = "full" if limit is None else f"limit_{str(limit).replace('.', 'p')}"
    pred_path = output_dir / f"preds_{args.num_fewshots}shot_{suffix}.jsonl"
    report_path = output_dir / f"report_{args.num_fewshots}shot_{suffix}.json"
    meta_path = output_dir / f"meta_{args.num_fewshots}shot_{suffix}.json"

    print("Starting LLMSQL Transformers inference")
    print(f"model_path={args.model_path}")
    print(f"output_dir={output_dir}")
    print(f"batch_size={args.batch_size}")
    print(f"num_fewshots={args.num_fewshots} limit={limit}")

    inference_transformers(
        model_or_model_name_or_path=args.model_path,
        output_file=str(pred_path),
        questions_path=args.questions_path,
        tables_path=args.tables_path,
        workdir_path=str(output_dir / "llmsql_workdir"),
        num_fewshots=args.num_fewshots,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        limit=limit,
        seed=args.seed,
        dtype=torch.float16,
        device_map="auto",
        model_kwargs={
            "attn_implementation": args.attn_implementation,
        },
    )

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
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "accuracy": report["accuracy"],
        "matches": report["matches"],
        "total": report["total"],
        "sql_errors": report["sql_errors"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Transformers evaluation finished")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
