#!/usr/bin/env python3
"""Summarise fixed-budget JSONL results into a Markdown report."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def pct(values: list[bool]) -> float:
    return 100.0 * sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Input JSONL is empty.")

    rows_by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows_by_id: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        budget = int(row["budget"])
        rows_by_budget[budget].append(row)
        rows_by_id[row["id"]][budget] = row
    budgets = sorted(rows_by_budget)
    full_budget = max(budgets)

    lines = [
        "# Fixed visual-token budget baseline",
        "",
        f"- Input: `{args.input}`",
        f"- Samples: {len(rows_by_id)}",
        f"- Full-token reference: {full_budget}",
        "",
        "## Aggregate metrics",
        "",
        "| Budget | Samples | Accuracy (%) | Prefill median (ms) | Prefill mean (ms) | Generate mean (ms) | Peak memory mean (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        group = rows_by_budget[budget]
        correct = [bool(row["quick_exact_correct"]) for row in group]
        prefill = [1000 * float(row["prefill_seconds"]) for row in group]
        generation = [1000 * float(row["generate_seconds"]) for row in group]
        memory = [float(row["peak_memory_mb"]) for row in group]
        lines.append(
            f"| {budget} | {len(group)} | {pct(correct):.2f} | "
            f"{statistics.median(prefill):.2f} | {mean(prefill):.2f} | "
            f"{mean(generation):.2f} | {mean(memory):.1f} |"
        )

    full_rows = rows_by_budget[full_budget]
    full_correct_by_id = {row["id"]: bool(row["quick_exact_correct"]) for row in full_rows}
    lines.extend([
        "",
        "## Comparison with full-token reference",
        "",
        "| Budget | Same prediction as full (%) | Low correct / full wrong | Low wrong / full correct |",
        "|---:|---:|---:|---:|",
    ])
    for budget in budgets:
        common = [
            (row, rows_by_id[row["id"]][full_budget])
            for row in rows_by_budget[budget]
            if full_budget in rows_by_id[row["id"]]
        ]
        same_prediction = [low["prediction"].strip() == full["prediction"].strip() for low, full in common]
        low_only = sum(bool(low["quick_exact_correct"]) and not bool(full["quick_exact_correct"]) for low, full in common)
        full_only = sum(not bool(low["quick_exact_correct"]) and bool(full["quick_exact_correct"]) for low, full in common)
        lines.append(f"| {budget} | {pct(same_prediction):.2f} | {low_only} | {full_only} |")

    # A sample is safe if its ground-truth correctness is not lower than the
    # full-token reference. This is the per-sample supervision target for the
    # future budget router; it deliberately permits a compressed run to improve
    # a full-token mistake.
    safe_budgets: list[int] = []
    incomplete = 0
    for sample_id, per_budget in rows_by_id.items():
        if full_budget not in per_budget:
            incomplete += 1
            continue
        full_correct = bool(per_budget[full_budget]["quick_exact_correct"])
        safe = [
            budget for budget in budgets
            if budget in per_budget and bool(per_budget[budget]["quick_exact_correct"]) >= full_correct
        ]
        safe_budgets.append(min(safe))
    allocation = {budget: safe_budgets.count(budget) for budget in budgets}
    lines.extend([
        "",
        "## Ground-truth-defined minimum safe budget",
        "",
        "A budget is considered safe when its correctness is no worse than the "
        "576-token reference on the same example. This is a preliminary oracle "
        "label for testing whether fixed budgets are mismatched.",
        "",
        "| Minimum safe budget | Sample count | Share (%) |",
        "|---:|---:|---:|",
    ])
    for budget in budgets:
        lines.append(f"| {budget} | {allocation[budget]} | {100 * allocation[budget] / len(safe_budgets):.2f} |")
    if incomplete:
        lines.append(f"\nWarning: {incomplete} samples lacked the full-budget result and were excluded.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
