"""Dataset metrics used by the initial BARS baselines."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def extract_final_answer(text: str) -> str:
    """Return a compact answer, including a direct answer from a non-BARS baseline."""
    text = text.strip()
    tagged = re.findall(r"(?:final answer|answer)\s*[:：]\s*([^\n]+)", text, re.I)
    answer = tagged[-1].strip() if tagged else text.splitlines()[-1].strip() if text else ""
    wrapped = re.fullmatch(r"<answer>\s*(.*?)\s*</answer>", answer, re.I | re.S)
    return wrapped.group(1).strip() if wrapped else answer


def has_final_answer(text: str) -> bool:
    """Whether the reasoning trace reached the required BARS final-answer marker."""
    return bool(re.search(r"\bfinal\s+answer\s*[:：]", text, re.I))


def extract_marked_final_answer(text: str) -> str:
    """Return a BARS final answer only if the trace completed its final marker."""
    return extract_final_answer(text) if has_final_answer(text) else ""


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).strip(" .")


def _number(value: str) -> Decimal | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1].strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    return number / Decimal("100") if percent else number


def chartqa_relaxed_correct(prediction: str, answer: str, tolerance: float = 0.05) -> bool:
    """ChartQA relaxed accuracy: exact string match or relative numeric error <=5%."""
    prediction, answer = extract_final_answer(prediction), str(answer)
    if _normalise_text(prediction) == _normalise_text(answer):
        return True
    predicted_number, gold_number = _number(prediction), _number(answer)
    if predicted_number is None or gold_number is None:
        return False
    if gold_number == 0:
        return predicted_number == 0
    return abs(predicted_number - gold_number) / abs(gold_number) <= Decimal(str(tolerance))


def scienceqa_correct(prediction: str, answer: str) -> bool:
    match = re.search(r"\b([A-Z])\b", extract_final_answer(prediction).upper())
    return bool(match and match.group(1) == str(answer).strip().upper())


def is_correct(dataset: str, prediction: str, answer: str) -> bool:
    if dataset == "chartqa":
        return chartqa_relaxed_correct(prediction, answer)
    if dataset == "scienceqa":
        return scienceqa_correct(prediction, answer)
    raise ValueError(f"Unsupported dataset: {dataset}")
