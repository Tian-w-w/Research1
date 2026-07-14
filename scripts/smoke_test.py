#!/usr/bin/env python3
"""Offline LLaVA-1.5 smoke test and visual-token inspection.

This script uses the complete Hugging Face LLaVA checkpoint.  For such a
checkpoint the vision tower is already inside the model; the separately cached
CLIP directory is checked for completeness but is not loaded a second time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = read_config(args.config)
    model_path = Path(cfg["model_path"])
    clip_path = Path(cfg["clip_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Missing LLaVA model: {model_path}")
    if not clip_path.exists():
        raise FileNotFoundError(f"Missing local CLIP cache: {clip_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This experiment requires a GPU server.")

    with args.manifest.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not 0 <= args.index < len(records):
        raise IndexError(f"--index must be in [0, {len(records) - 1}]")
    record = records[args.index]
    image_path = Path(record["image"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model from {model_path} using {dtype} on {torch.cuda.get_device_name(0)}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{record['prompt']}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = move_to_device(inputs, device)

    with torch.inference_mode():
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        )
    print(f"Image features shape: {tuple(image_features.shape)}")
    if image_features.ndim != 3 or image_features.shape[1] != 576:
        raise RuntimeError(
            "Expected [batch, 576, hidden] visual features for LLaVA-1.5-7B. "
            f"Got {tuple(image_features.shape)}. Stop here and check checkpoint/processor versions."
        )
    print("Verified: 576 visual tokens.")

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    print("=" * 72)
    print(f"sample id : {record['id']}")
    print(f"dataset   : {record['dataset']}")
    print(f"image     : {image_path}")
    print(f"question  : {record['question']}")
    print(f"gold      : {record['answer']}")
    print(f"prediction: {answer}")
    print(f"generate time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
