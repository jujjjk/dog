from __future__ import annotations

from typing import Any

from .profile_utils import load_yaml_profile


def load_randomization_profile(path: str = "config/randomization_profile.yaml") -> dict[str, Any]:
    return load_yaml_profile(path)
