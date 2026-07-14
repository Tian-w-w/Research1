#!/usr/bin/env python3
"""Measure fixed visual-token budgets on one-image LLaVA-1.5 inputs.

This is deliberately a *pre-router* baseline.  It keeps uniformly spaced visual
tokens, physically removes the other image placeholders before LLM prefill, and
patches the image-feature method to return the matching subset.  The vision
encoder still processes the full image; the measured saving is therefore in the
LLM prefill/KV-cache stage, which is exactly the stage targeted by this project.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import types
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def scienceqa_prediction_letter(text: str) -> str | None:
    match = re.search(r"\b([A-Z])\b", text.upper())
    return match.group(1) if match else None


def is_correct(record: dict[str, Any], prediction: str) -> bool | None:
    if record["dataset"] == "scienceqa":
        return scienceqa_prediction_letter(prediction) == str(record["answer"]).upper()
    # ChartQA's official metric has number tolerance and answer aliases.  Keep
    # this exact-match value only as a quick diagnostic; official scoring comes
    # after the routing method is ready.
    if record["dataset"] == "chartqa":
        return normalise(prediction) == normalise(str(record["answer"]))
    return None


def uniform_indices(total_tokens: int, budget: int, device: torch.device) -> torch.Tensor:
    if not 1 <= budget <= total_tokens:
        raise ValueError(f"Budget must be in [1, {total_tokens}], got {budget}.")
    if budget == total_tokens:
        return torch.arange(total_tokens, device=device)
    # Rounded linspace is sorted and unique because budget <= total_tokens.
    return torch.linspace(0, total_tokens - 1, budget, device=device).round().long()


def prune_feature_result(value: Any, indices: torch.Tensor) -> Any:
    """Apply the same ordered selection to all HF LLaVA return conventions."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("This baseline supports exactly one image per sample.")
        return [value[0].index_select(0, indices.to(value[0].device))]
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            return value.index_select(0, indices.to(value.device))
        if value.ndim == 3:
            return value.index_select(1, indices.to(value.device))
        raise ValueError(f"Unsupported image tensor shape: {tuple(value.shape)}")
    # Some Transformers releases return a ModelOutput with the features stored
    # in pooler_output.
    if hasattr(value, "pooler_output"):
        value.pooler_output = prune_feature_result(value.pooler_output, indices)
        return value
    raise TypeError(f"Unsupported get_image_features return type: {type(value)}")


class ImageFeaturePruner:
    """Temporarily replace the internal image-feature method for one generation."""

    def __init__(self, model: LlavaForConditionalGeneration, indices: torch.Tensor):
        self.module = model.model
        self.indices = indices
        self.original = self.module.get_image_features

    def __enter__(self) -> None:
        original = self.original
        indices = self.indices

        def patched(_module: Any, *args: Any, **kwargs: Any) -> Any:
            return prune_feature_result(original(*args, **kwargs), indices)

        self.module.get_image_features = types.MethodType(patched, self.module)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.module.get_image_features = self.original


def prune_image_placeholders(
    inputs: dict[str, Any], image_token_id: int, indices: torch.Tensor
) -> dict[str, Any]:
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("This baseline uses batch size 1 for latency measurement.")
    placeholder_positions = torch.where(input_ids[0] == image_token_id)[0]
    total_tokens = placeholder_positions.numel()
    if total_tokens != 576:
        raise RuntimeError(f"Expected 576 image placeholders, got {total_tokens}.")

    selected_positions = placeholder_positions.index_select(0, indices.to(input_ids.device))
    sequence_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    sequence_mask[placeholder_positions] = False
    sequence_mask[selected_positions] = True

    pruned = dict(inputs)
    pruned["input_ids"] = input_ids[:, sequence_mask]
    if "attention_mask" in inputs:
        pruned["attention_mask"] = inputs["attention_mask"][:, sequence_mask]
    return pruned


def load_records(manifest: Path, max_samples: int) -> list[dict[str, Any]]:
    with manifest.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:max_samples] if max_samples else records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[72, 144, 288, 432, 576])
    parser.add_argument("--max-samples", type=int, default=50,
                        help="0 means every record. Start with 50 for a smoke benchmark.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = read_config(args.config)
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    records = load_records(args.manifest, args.max_samples)
    if not records:
        raise ValueError("Manifest is empty.")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise AttributeError("Cannot find image_token_index/image_token_id in model config.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for record in tqdm(records, desc="Fixed-budget benchmark"):
            image = Image.open(record["image"]).convert("RGB")
            prompt = f"USER: <image>\n{record['prompt']}\nASSISTANT:"
            base_inputs = move_to_device(
                processor(text=prompt, images=image, return_tensors="pt"), device
            )
            for budget in args.budgets:
                indices = uniform_indices(576, budget, device)
                pruned_inputs = prune_image_placeholders(base_inputs, image_token_id, indices)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                with ImageFeaturePruner(model, indices):
                    # Time the prefill separately. It is the latency component
                    # that token pruning is expected to improve.
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        _ = model(**pruned_inputs, use_cache=True)
                    torch.cuda.synchronize()
                    prefill_seconds = time.perf_counter() - start

                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **pruned_inputs,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                        )
                    torch.cuda.synchronize()
                    total_seconds = time.perf_counter() - start

                prompt_length = pruned_inputs["input_ids"].shape[1]
                new_tokens = generated[:, prompt_length:]
                prediction = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                result = {
                    "id": record["id"],
                    "dataset": record["dataset"],
                    "budget": budget,
                    "visual_tokens": budget,
                    "prompt_tokens_after_pruning": prompt_length,
                    "prediction": prediction,
                    "answer": record["answer"],
                    "quick_exact_correct": is_correct(record, prediction),
                    "prefill_seconds": prefill_seconds,
                    "generate_seconds": total_seconds,
                    "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                }
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                writer.flush()

    print(f"Wrote fixed-budget results to {args.output}")


if __name__ == "__main__":
    main()
