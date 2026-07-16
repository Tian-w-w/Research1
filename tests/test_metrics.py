from bars.metrics import (
    chartqa_relaxed_correct,
    extract_final_answer,
    extract_marked_final_answer,
    has_final_answer,
    scienceqa_correct,
)


def test_chartqa_relaxed_accuracy_numeric_and_text() -> None:
    assert chartqa_relaxed_correct("Final answer: 105", "100")
    assert not chartqa_relaxed_correct("Final answer: 106", "100")
    assert chartqa_relaxed_correct("New York", "new york")
    assert chartqa_relaxed_correct("0%", "0")


def test_chartqa_relaxed_accuracy_tolerates_nearby_years() -> None:
    # This follows ChartQA's official numeric relaxed-accuracy definition.
    assert chartqa_relaxed_correct("2005", "2018")


def test_final_answer_and_scienceqa() -> None:
    assert extract_final_answer("work\nFinal answer: 42") == "42"
    assert extract_final_answer("Final answer: <answer>green</answer>") == "green"
    assert extract_final_answer("unfinished reasoning") == "unfinished reasoning"
    assert extract_marked_final_answer("unfinished reasoning") == ""
    assert has_final_answer("work\nFinal answer: 42")
    assert not has_final_answer("unfinished reasoning")
    assert not has_final_answer("Final answer: <answer>7")
    assert has_final_answer("Final answer: <answer>72</answer>")
    assert scienceqa_correct("Reasoning.\nAnswer: C", "C")
