from bars.router import Action, RoutingState, choose_action, choose_solve_only_action


def state(**kwargs):
    defaults = dict(used_tokens=128, total_budget=512, candidate_complete=True, candidate_answer="42")
    defaults.update(kwargs)
    return RoutingState(**defaults)


def test_incomplete_candidate_is_solved() -> None:
    assert choose_action(state(candidate_complete=False, candidate_answer="")) is Action.SOLVE


def test_completed_candidate_is_verified_then_stopped_if_agreed() -> None:
    assert choose_action(state()) is Action.VERIFY
    assert choose_action(state(verification_answer="42")) is Action.STOP


def test_conflicting_verification_replans_once() -> None:
    assert choose_action(state(verification_answer="24")) is Action.REPLAN
    assert choose_action(state(verification_answer="24", replan_count=1)) is Action.STOP


def test_budget_exhaustion_stops() -> None:
    assert choose_action(state(used_tokens=512)) is Action.STOP


def test_solve_only_baseline_never_verifies_or_replans() -> None:
    assert choose_solve_only_action(state(candidate_complete=False)) is Action.SOLVE
    assert choose_solve_only_action(state(candidate_complete=True)) is Action.STOP
