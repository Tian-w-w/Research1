#!/usr/bin/env python3
"""Summarise the quality-cost curve from ``bars_fixed_budget.py`` JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Input JSONL is empty.")
    by_budget: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[int(row["budget"])].append(row)
    lines = [
        "# BARS fixed reasoning-budget baseline", "",
        f"- Input: `{args.input}`",
        "- Metric: ChartQA relaxed accuracy (or ScienceQA exact option accuracy).", "",
        "| Budget cap | Samples | Accuracy (%) | Complete final answer (%) | Budget exhausted (%) | Mean generated tokens | P50 latency (s) | P95 latency (s) | Mean peak memory (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in sorted(by_budget):
        group = by_budget[budget]
        accuracy = 100 * sum(bool(row["correct"]) for row in group) / len(group)
        completed = 100 * sum(bool(row.get("has_final_answer", False)) for row in group) / len(group)
        exhausted = 100 * sum(bool(row.get("budget_exhausted", False)) for row in group) / len(group)
        tokens = [float(row["generated_tokens"]) for row in group]
        latency = [float(row["latency_seconds"]) for row in group]
        memory = [float(row["peak_memory_mb"]) for row in group]
        lines.append(
            f"| {budget} | {len(group)} | {accuracy:.2f} | {completed:.2f} | {exhausted:.2f} | {statistics.mean(tokens):.1f} | "
            f"{percentile(latency, .50):.3f} | {percentile(latency, .95):.3f} | "
            f"{statistics.mean(memory):.1f} |"
        )
    lines.extend([
        "", "Interpret this table as a quality-cost curve, not as evidence for dynamic routing. "
        "Select later BARS chunk sizes from observed marginal gains, and tune controllers on train/val only.", "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
