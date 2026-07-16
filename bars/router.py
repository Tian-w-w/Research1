"""Deterministic no-tool BARS controller used before learning action values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    SOLVE = "solve"
    VERIFY = "verify"
    REPLAN = "replan"
    STOP = "stop"


@dataclass
class RoutingState:
    used_tokens: int
    total_budget: int
    candidate_complete: bool
    candidate_answer: str
    verification_answer: str | None = None
    replan_count: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.total_budget - self.used_tokens)


def canonical_answer(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def choose_action(state: RoutingState) -> Action:
    """Apply the paper's first transparent solve/verify/replan/stop policy."""
    if state.remaining_tokens <= 0:
        return Action.STOP
    if not state.candidate_complete:
        return Action.SOLVE
    if state.verification_answer is None:
        return Action.VERIFY
    if canonical_answer(state.candidate_answer) == canonical_answer(state.verification_answer):
        return Action.STOP
    if state.replan_count == 0:
        return Action.REPLAN
    return Action.STOP
