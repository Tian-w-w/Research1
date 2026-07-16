"""Offline Qwen3-VL loading and deterministic one-image generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SYSTEM_PROMPT = """You are solving a ChartQA question from an image.
Read the chart carefully before answering. First write the relevant chart values
and a step-by-step calculation or comparison. Do not answer with only a short
phrase or number. After the reasoning, put the result on its own final line in
exactly this form: Final answer: <answer>"""


def set_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_qwen_vl(model_path: Path, device: torch.device) -> tuple[Any, Any]:
    """Load a local Transformers-compatible Qwen3-VL checkpoint only."""
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; BARS inference must run on the GPU server.")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    return model, processor


def render_prompt(
    processor: Any,
    question: str,
    prior_trace: str | None = None,
    follow_up: str | None = None,
    initial_instruction: str | None = None,
) -> str:
    if prior_trace is not None and initial_instruction is not None:
        raise ValueError("An initial instruction cannot be combined with a prior trace.")
    user_text = question if initial_instruction is None else f"{question}\n\n{initial_instruction}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": user_text}],
    }]
    if prior_trace is not None:
        if not follow_up:
            raise ValueError("A follow-up instruction is required when continuing a trace.")
        messages.extend([
            {"role": "assistant", "content": [{"type": "text", "text": prior_trace}]},
            {"role": "user", "content": [{"type": "text", "text": follow_up}]},
        ])
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("The configured processor lacks apply_chat_template; check checkpoint format.")
    # Qwen3-VL checkpoints expose this switch to enable their reasoning channel.
    # Older compatible processors may not accept it, hence the narrow fallback.
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(
    model: Any,
    processor: Any,
    image_path: str,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    prior_trace: str | None = None,
    follow_up: str | None = None,
    initial_instruction: str | None = None,
) -> tuple[str, int]:
    image = Image.open(image_path).convert("RGB")
    prompt = render_prompt(processor, question, prior_trace, follow_up, initial_instruction)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                   use_cache=True)
    prompt_tokens = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_tokens:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()
    return text, int(new_tokens.shape[1])
