#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cpg_tool_common import package_dir, workspace_root, write_text, write_yaml


def default_manual_text() -> Path:
    return workspace_root() / "机器狗" / "_rs01_manual_text.txt"


def build_profile() -> dict:
    return {
        "motor": {
            "name": "RS01",
            "peak_torque_nm": 17.0,
            "continuous_torque_nm": 6.0,
            "safe_training_torque_nm": 5.0,
            "max_velocity_rad_s": 33.0,
            "rated_velocity_rad_s": 10.47,
            "rated_current_a": None,
            "rated_phase_current_peak_a": 7.0,
            "over_current_threshold_a": None,
            "max_phase_current_peak_a": 23.0,
            "temperature_limit_c": 103.0,
            "winding_protection_temperature_c": 145.0,
            "voltage_range_v": [24.0, 50.0],
            "rated_voltage_v": 36.0,
            "reduction_ratio": 7.75,
        },
        "control": {
            "mode": "position_pd",
            "kp_range": None,
            "kd_range": None,
            "recommended_kp": 40.0,
            "recommended_kd": 5.0,
        },
        "safety": {
            "torque_limit_for_training_nm": 5.0,
            "torque_limit_for_deploy_nm": 6.0,
            "current_limit_for_deploy_a": None,
            "temperature_limit_c": 103.0,
        },
        "notes": {
            "source": "RS01 manual text extraction plus current project deployment gains.",
            "warnings": [
                "rated_current_a is not filled because the manual text provides phase peak current, not a directly interchangeable DC bus current.",
                "safe_training_torque_nm is conservative and below the 6 Nm continuous rating.",
                "Kp/Kd recommendations come from the current project configuration, not the RS01 manual.",
            ],
        },
    }


def render_text(profile: dict, source: Path) -> str:
    lines = ["RS01 motor specs report", "", f"source_text: {source}", ""]
    for section, data in profile.items():
        lines.append(f"[{section}]")
        if isinstance(data, dict):
            for key, value in data.items():
                lines.append(f"  {key}: {value}")
        lines.append("")
    for warning in profile["notes"]["warnings"]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-text", default=str(default_manual_text()))
    parser.add_argument("--out-dir", default=str(package_dir()))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    source = Path(args.manual_text)
    profile = build_profile()
    if not source.exists():
        profile["notes"]["warnings"].append(f"manual text file missing: {source}")
    write_yaml(out_dir / "config" / "motor_profile.yaml", profile)
    write_text(out_dir / "logs" / "motor_specs_report.txt", render_text(profile, source))
    print(f"wrote {out_dir / 'config' / 'motor_profile.yaml'}")


if __name__ == "__main__":
    main()
