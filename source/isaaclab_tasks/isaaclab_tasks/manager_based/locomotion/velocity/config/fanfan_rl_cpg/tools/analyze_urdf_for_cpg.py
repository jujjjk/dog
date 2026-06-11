#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cpg_tool_common import (
    JOINT_ORDER,
    default_robot_cfg_path,
    default_urdf_path,
    leg_report,
    load_default_joint_pos,
    package_dir,
    parse_urdf,
    write_text,
    write_yaml,
)


def build_report(urdf_path: Path, robot_cfg: Path) -> dict:
    default_pos = load_default_joint_pos(robot_cfg)
    urdf = parse_urdf(urdf_path)
    legs, warnings = leg_report(urdf, default_pos)
    joints = urdf["joints"]
    nominal_leg = [v["nominal_leg_length_m"] for v in legs.values()]
    leg_len = sum(nominal_leg) / max(len(nominal_leg), 1)
    cpg = {
        "step_height_m": max(0.025, min(0.040, 0.10 * leg_len)),
        "step_length_min_m": 0.015,
        "step_length_max_m": max(0.030, min(0.060, 0.18 * leg_len)),
        "frequency_min_hz": 0.8,
        "frequency_max_hz": 1.8,
        "duty_factor": 0.60,
        "residual_limit_rad": {"hip": 0.04, "thigh": 0.06, "calf": 0.06},
        "joint_safety_margin_rad": 0.08,
    }
    joint_report = {}
    for name in JOINT_ORDER:
        item = joints.get(name)
        if item is None:
            warnings.append(f"missing expected joint: {name}")
            continue
        lower = item["limit"]["lower"]
        upper = item["limit"]["upper"]
        default = default_pos.get(name)
        margin = None
        if lower is not None and upper is not None and default is not None:
            margin = min(default - lower, upper - default)
            if margin < 0.08:
                warnings.append(f"{name}: default pose is close to joint limit, margin={margin:.4f} rad")
        joint_report[name] = {
            "parent": item["parent"],
            "child": item["child"],
            "axis": item["axis"],
            "origin_xyz": item["origin_xyz"],
            "origin_rpy": item["origin_rpy"],
            "limit": item["limit"],
            "default_joint_pos": default,
            "default_limit_margin_rad": margin,
        }
    warnings.append("URDF cannot determine gait phase/frequency; phase is gait design, frequency comes from speed, step length, motor data, and logs.")
    return {
        "source": {"urdf": str(urdf_path), "robot_cfg": str(robot_cfg)},
        "leg_order": ["FR", "FL", "RR", "RL"],
        "joint_order": list(JOINT_ORDER),
        "legs": legs,
        "joints": joint_report,
        "cpg_recommendation": cpg,
        "warnings": warnings,
    }


def render_text(report: dict) -> str:
    lines = ["Fanfan CPG URDF report", ""]
    lines.append(f"URDF: {report['source']['urdf']}")
    lines.append(f"leg_order: {report['leg_order']}")
    lines.append(f"joint_order: {report['joint_order']}")
    lines.append("")
    for leg, item in report["legs"].items():
        lines.append(f"[{leg}] joints={item['joints']} foot={item['foot_link']}")
        lines.append(
            f"  thigh={item['thigh_length_m']:.5f} m calf={item['calf_length_m']:.5f} m leg={item['nominal_leg_length_m']:.5f} m"
        )
        lines.append(f"  nominal_foot_position={item['nominal_foot_position_m']}")
    lines.append("")
    for name, item in report["joints"].items():
        lines.append(
            f"{name}: axis={item['axis']} limit={item['limit']} default={item['default_joint_pos']} margin={item['default_limit_margin_rad']}"
        )
    lines.append("")
    lines.append("CPG recommendation:")
    for key, value in report["cpg_recommendation"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=str(default_urdf_path()))
    parser.add_argument("--robot-cfg", default=str(default_robot_cfg_path()))
    parser.add_argument("--out-dir", default=str(package_dir()))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report = build_report(Path(args.urdf), Path(args.robot_cfg))
    write_yaml(out_dir / "logs" / "cpg_urdf_report.yaml", report)
    write_text(out_dir / "logs" / "cpg_urdf_report.txt", render_text(report))
    print(f"wrote {out_dir / 'logs' / 'cpg_urdf_report.yaml'}")


if __name__ == "__main__":
    main()
