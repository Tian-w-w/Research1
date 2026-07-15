#!/usr/bin/env python3
"""Run deterministic fixed reasoning-token baselines for BARS."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bars.config import read_yaml, required_path
from bars.metrics import extract_final_answer, is_correct
from bars.model import generate, load_qwen_vl, set_offline_mode


def read_records(path: Path, max_samples: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:max_samples] if max_samples else records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if any(budget <= 0 for budget in args.budgets):
        raise ValueError("All --budgets must be positive.")
    set_offline_mode()
    records = read_records(args.manifest, args.max_samples)
    if not records:
        raise ValueError("Manifest is empty.")
    cfg = read_yaml(args.config)
    device = torch.device("cuda:0")
    model, processor = load_qwen_vl(required_path(cfg, "qwen3_vl_4b_path"), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for record in tqdm(records, desc="Fixed reasoning budgets"):
            for budget in args.budgets:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                # Use the raw question, rather than the legacy manifest prompt.
                # The latter requests a concise answer and suppresses the
                # explicit reasoning trace required for this budget baseline.
                raw_prediction, generated_tokens = generate(
                    model, processor, record["image"], record["question"], device, budget
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                answer = extract_final_answer(raw_prediction)
                row = {
                    "id": record["id"], "dataset": record["dataset"], "split": record.get("split"),
                    "budget": budget, "max_new_tokens": budget, "generated_tokens": generated_tokens,
                    "prediction": raw_prediction, "final_answer": answer, "answer": str(record["answer"]),
                    "correct": is_correct(record["dataset"], raw_prediction, str(record["answer"])),
                    "latency_seconds": elapsed,
                    "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
    print(f"Wrote {len(records) * len(args.budgets)} rows to {args.output}")


if __name__ == "__main__":
    main()
