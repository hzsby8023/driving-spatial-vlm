"""Strict JSON configuration helpers for the M1 action-token pipeline."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def _merge_strict(base: dict[str, Any], override: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in result:
            raise ValueError(f"Unknown configuration key: {dotted}")
        if isinstance(result[key], dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"Configuration section must be an object: {dotted}")
            result[key] = _merge_strict(result[key], value, dotted)
        else:
            result[key] = value
    return result


def load_json_config(path: str | Path, defaults: Mapping[str, Any]) -> dict[str, Any]:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be an object: {config_path}")
    return _merge_strict(dict(defaults), raw)


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply non-None dotted-key CLI overrides with the same strict validation."""
    result = deepcopy(config)
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        node = result
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown configuration key: {dotted}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown configuration key: {dotted}")
        node[parts[-1]] = value
    return result
