"""Configuration helpers shared by BARS command-line scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def required_path(config: dict[str, Any], name: str) -> Path:
    value = config.get(name)
    if not value:
        raise KeyError(f"Missing required config key: {name}")
    return Path(value).expanduser()
