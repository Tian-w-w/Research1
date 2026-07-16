#!/usr/bin/env python3
"""Run the no-tool rule-based BARS controller on a ChartQA/ScienceQA manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bars.config import read_yaml, required_path
from bars.metrics import extract_marked_final_answer, has_final_answer, is_correct
from bars.model import generate, load_qwen_vl, set_offline_mode
from bars.router import Action, RoutingState, choose_action, choose_solve_only_action


def read_records(path: Path, max_samples: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:max_samples] if max_samples else records


def action_instruction(action: Action, continuation: bool = False) -> str:
    if action is Action.SOLVE:
        return (
            "Continue the unfinished reasoning from exactly where it stopped. Re-read the chart if "
            "needed, do not repeat earlier steps, and finish with `Final answer: <answer>`."
        )
    if action is Action.VERIFY:
        if continuation:
            return (
                "Continue the independent verification from exactly where it stopped. Do not repeat "
                "earlier evidence; complete the check and finish with `Final answer: <answer>`."
            )
        return (
            "Act as an independent verifier. Re-read the chart and recompute the answer; do not "
            "assume the draft is correct. State the key evidence and calculation, then finish with "
            "`Final answer: <answer>`."
        )
    if action is Action.REPLAN:
        return (
            "The prior answer disagreed with an independent check. Discard that path and solve the "
            "question again using a different reading or calculation approach. Finish with "
            "`Final answer: <answer>`."
        )
    raise ValueError(f"No generation instruction for {action}.")


def run_action(
    action: Action,
    model: Any,
    processor: Any,
    record: dict[str, Any],
    draft: str | None,
    device: torch.device,
    chunk: int,
    initial: bool = False,
) -> tuple[dict[str, Any], str]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if initial:
        trace, generated_tokens = generate(
            model, processor, record["image"], record["question"], device, chunk,
        )
    elif action is Action.VERIFY and draft is None:
        # A verifier must not receive the candidate's assistant trace; otherwise
        # deterministic decoding merely echoes it instead of independently
        # re-reading the image.
        trace, generated_tokens = generate(
            model, processor, record["image"], record["question"], device, chunk,
            initial_instruction=action_instruction(action),
        )
    else:
        trace, generated_tokens = generate(
            model, processor, record["image"], record["question"], device, chunk,
            prior_trace=draft,
            follow_up=action_instruction(action, continuation=action is Action.VERIFY),
        )
    torch.cuda.synchronize(device)
    return {
        "action": action.value,
        "max_new_tokens": chunk,
        "generated_tokens": generated_tokens,
        "latency_seconds": time.perf_counter() - started,
        "trace": trace,
        "has_final_answer": has_final_answer(trace),
        "final_answer": extract_marked_final_answer(trace),
    }, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-budget", type=int, default=128)
    parser.add_argument("--action-chunk", type=int, default=128)
    parser.add_argument("--total-budget", type=int, default=512)
    parser.add_argument("--policy", choices=("bars_rule", "solve_only"), default="bars_rule",
                        help="bars_rule uses Solve/Verify/Replan/Stop; solve_only is its chunked fair baseline.")
    parser.add_argument("--max-actions", type=int, default=8,
                        help="Safety guard for malformed model outputs; includes the initial solve.")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if min(args.initial_budget, args.action_chunk, args.total_budget) <= 0:
        raise ValueError("All budgets must be positive.")
    if args.initial_budget > args.total_budget:
        raise ValueError("--initial-budget must not exceed --total-budget.")

    set_offline_mode()
    records = read_records(args.manifest, args.max_samples)
    if not records:
        raise ValueError("Manifest is empty.")
    cfg = read_yaml(args.config)
    device = torch.device("cuda:0")
    model, processor = load_qwen_vl(required_path(cfg, "qwen3_vl_4b_path"), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as writer:
        for record in tqdm(records, desc="Rule-based BARS"):
            initial, draft = run_action(
                Action.SOLVE, model, processor, record, None, device, args.initial_budget, initial=True,
            )
            # The initial generation is a solve, but retain its own label for
            # transparent action traces.
            initial["action"] = "initial_solve"
            steps = [initial]
            candidate = initial
            verification_answer: str | None = None
            verification_trace: str | None = None
            replan_count = 0

            while len(steps) < args.max_actions:
                used_tokens = sum(int(step["generated_tokens"]) for step in steps)
                state = RoutingState(
                    used_tokens=used_tokens,
                    total_budget=args.total_budget,
                    candidate_complete=bool(candidate["has_final_answer"]),
                    candidate_answer=str(candidate["final_answer"]),
                    verification_answer=verification_answer,
                    replan_count=replan_count,
                )
                action = (
                    choose_action(state) if args.policy == "bars_rule"
                    else choose_solve_only_action(state)
                )
                if action is Action.STOP:
                    break
                chunk = min(args.action_chunk, state.remaining_tokens)
                # The first verification starts independently from the image and
                # question. If it reaches its chunk limit, later verification
                # actions continue *that verifier's* trace, never the candidate
                # trace, so it remains an independent evidence path.
                action_draft = (
                    verification_trace if action is Action.VERIFY else draft
                )
                step, new_trace = run_action(
                    action, model, processor, record, action_draft, device, chunk
                )
                steps.append(step)
                if action is Action.VERIFY:
                    verification_trace = new_trace
                    # An unfinished verifier has no opinion yet. Continue the
                    # verification trace instead of treating an empty string as
                    # a disagreement and prematurely replanning.
                    verification_answer = (
                        str(step["final_answer"]) if step["has_final_answer"] else None
                    )
                    if verification_answer is not None:
                        draft = (
                            f"Original candidate:\n{candidate['trace']}\n\n"
                            f"Independent verification:\n{step['trace']}"
                        )
                else:
                    candidate = step
                    draft = new_trace
                    if action is Action.REPLAN:
                        replan_count += 1
                        verification_answer = None
                        verification_trace = None

            total_tokens = sum(int(step["generated_tokens"]) for step in steps)
            final_answer = str(candidate["final_answer"])
            result = {
                "id": record["id"],
                "policy": args.policy,
                "dataset": record["dataset"],
                "split": record.get("split"),
                "answer": str(record["answer"]),
                "prediction": final_answer,
                "correct": is_correct(record["dataset"], final_answer, str(record["answer"])),
                "has_final_answer": bool(candidate["has_final_answer"]),
                "total_generated_tokens": total_tokens,
                "total_latency_seconds": sum(float(step["latency_seconds"]) for step in steps),
                "stop_reason": (
                    "budget_exhausted" if total_tokens >= args.total_budget
                    else "max_actions" if len(steps) >= args.max_actions
                    else "rule_stop"
                ),
                "steps": steps,
            }
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()
    print(f"Wrote {len(records)} BARS rule rollouts to {args.output}")


if __name__ == "__main__":
    main()
