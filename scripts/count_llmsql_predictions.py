#!/usr/bin/env python3
"""Count LLMSQL prediction rows across main and resume chunk files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_question_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    question_ids: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            question_ids.append(json.loads(line)["question_id"])
    return question_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Count LLMSQL prediction progress.")
    parser.add_argument("--output-dir", required=True, help="Eval output directory.")
    parser.add_argument("--num-fewshots", type=int, default=5, help="Prompt variant.")
    parser.add_argument("--suffix", default="full", help="Prediction suffix, e.g. full or limit_5.")
    parser.add_argument("--total", type=int, default=None, help="Optional total question count.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    stem = f"preds_{args.num_fewshots}shot_{args.suffix}"
    main_path = output_dir / f"{stem}.jsonl"
    resume_chunk_path = output_dir / f"{stem}_resume_chunk.jsonl"

    main_qids = read_question_ids(main_path)
    resume_qids = read_question_ids(resume_chunk_path)
    unique_qids = set(main_qids) | set(resume_qids)

    main_count = len(main_qids)
    resume_count = len(resume_qids)
    total_count = len(unique_qids)

    print(f"main={main_count}")
    print(f"resume_chunk={resume_count}")
    print(f"total={total_count}")
    if args.total:
        pct = 100.0 * total_count / args.total
        print(f"progress={pct:.2f}%")


if __name__ == "__main__":
    main()
