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

    # Teacher labels must only be made on examples where the full-token teacher
    # is correct. If a full-token model is already wrong, treating every equally
    # wrong compressed answer as "safe" would collapse the label distribution
    # toward the smallest budget and create invalid router supervision.
    min_correct_budgets: list[int] = []
    full_incorrect = 0
    incomplete = 0
    for sample_id, per_budget in rows_by_id.items():
        if full_budget not in per_budget:
            incomplete += 1
            continue
        full_correct = bool(per_budget[full_budget]["quick_exact_correct"])
        if not full_correct:
            full_incorrect += 1
            continue
        correct_budgets = [
            budget for budget in budgets
            if budget in per_budget and bool(per_budget[budget]["quick_exact_correct"])
        ]
        # full_budget is correct, so this list is guaranteed to be non-empty.
        min_correct_budgets.append(min(correct_budgets))
    allocation = {budget: min_correct_budgets.count(budget) for budget in budgets}
    lines.extend([
        "",
        "## Oracle minimum correct budget (teacher-correct subset)",
        "",
        f"Only the {len(min_correct_budgets)} examples answered correctly at {full_budget} "
        "tokens are eligible for budget supervision. For each such example, the "
        "oracle label is the smallest tested budget that is also correct against "
        "ground truth. The remaining full-token errors are excluded rather than "
        "being mislabeled as safe low-budget examples.",
        "",
        "| Minimum correct budget | Sample count | Share among eligible (%) |",
        "|---:|---:|---:|",
    ])
    for budget in budgets:
        share = 100 * allocation[budget] / len(min_correct_budgets) if min_correct_budgets else 0.0
        lines.append(f"| {budget} | {allocation[budget]} | {share:.2f} |")
    lines.append(f"\nExcluded full-token-incorrect examples: {full_incorrect}.")
    if incomplete:
        lines.append(f"\nWarning: {incomplete} samples lacked the full-budget result and were excluded.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
