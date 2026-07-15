"""Offline Qwen3-VL loading and deterministic one-image generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SYSTEM_PROMPT = (
    "You are solving a chart question. Read the chart carefully, reason briefly, "
    "and end with 'Final answer: <answer>'."
)


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


def render_prompt(processor: Any, question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("The configured processor lacks apply_chat_template; check checkpoint format.")
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model: Any, processor: Any, image_path: str, question: str,
             device: torch.device, max_new_tokens: int) -> tuple[str, int]:
    image = Image.open(image_path).convert("RGB")
    prompt = render_prompt(processor, question)
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
