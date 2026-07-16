import pytest

from bars.model import render_prompt


class CapturingProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "rendered"


def test_continuation_preserves_prior_trace_as_assistant_turn() -> None:
    processor = CapturingProcessor()
    assert render_prompt(processor, "What is the value?", "partial trace", "Continue.") == "rendered"
    assert processor.messages[2]["role"] == "assistant"
    assert processor.messages[2]["content"][0]["text"] == "partial trace"
    assert processor.messages[3]["role"] == "user"


def test_independent_instruction_has_no_candidate_history() -> None:
    processor = CapturingProcessor()
    render_prompt(processor, "What is the value?", initial_instruction="Verify independently.")
    assert len(processor.messages) == 2
    assert "Verify independently." in processor.messages[1]["content"][1]["text"]


def test_initial_instruction_and_history_are_incompatible() -> None:
    with pytest.raises(ValueError):
        render_prompt(CapturingProcessor(), "Q", "draft", "continue", "verify")
