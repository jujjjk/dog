from __future__ import annotations

from typing import Any

from .profile_utils import load_yaml_profile


def load_motor_profile(path: str = "config/motor_profile.yaml") -> dict[str, Any]:
    return load_yaml_profile(path)


def load_fitted_motor_profile(path: str = "config/motor_profile_fitted.yaml") -> dict[str, Any]:
    return load_yaml_profile(path)
