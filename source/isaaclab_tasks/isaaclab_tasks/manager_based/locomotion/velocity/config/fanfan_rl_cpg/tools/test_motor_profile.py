#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from cpg_tool_common import package_dir, read_yaml


profile = read_yaml(package_dir() / "config" / "motor_profile.yaml")
missing = []
for path in ("motor.peak_torque_nm", "motor.continuous_torque_nm", "safety.torque_limit_for_training_nm"):
    cur = profile
    for key in path.split("."):
        cur = cur.get(key) if isinstance(cur, dict) else None
    if cur is None:
        missing.append(path)
print("motor_profile:", package_dir() / "config" / "motor_profile.yaml")
print("missing:", missing)
if profile.get("motor", {}).get("continuous_torque_nm") is None:
    peak = profile.get("motor", {}).get("peak_torque_nm", 17.0)
    print(f"WARNING: continuous torque missing; conservative range would be {0.4 * peak:.2f}-{0.6 * peak:.2f} Nm")
