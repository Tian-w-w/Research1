#!/usr/bin/env python3
"""Fail-fast preflight for an offline BARS server environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bars.config import read_yaml, required_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Install a CUDA-enabled PyTorch wheel offline.")
    print(f"CUDA runtime: {torch.version.cuda}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {properties.name}; {properties.total_memory / 1024**3:.1f} GiB")
    for key in ("qwen3_vl_4b_path", "chartqa_root", "scienceqa_root", "output_root"):
        path = required_path(cfg, key)
        print(f"{key}: {path} ({'OK' if path.exists() else 'MISSING'})")
        if not path.exists():
            raise FileNotFoundError(path)
    config_path = required_path(cfg, "qwen3_vl_4b_path") / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "unknown")
    print(f"Model type: {model_type}")
    print("BARS preflight passed.")


if __name__ == "__main__":
    main()
