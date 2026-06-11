from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_profile_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return package_dir() / path


def load_yaml_profile(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    profile_path = resolve_profile_path(path)
    if not profile_path.exists():
        return {} if default is None else dict(default)
    with profile_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {} if default is None else dict(default)
    return data


def write_yaml_profile(path: str | Path, data: dict[str, Any]) -> Path:
    profile_path = resolve_profile_path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return profile_path


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
