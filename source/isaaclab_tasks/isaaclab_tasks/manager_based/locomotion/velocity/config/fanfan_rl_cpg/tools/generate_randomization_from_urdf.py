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
    read_yaml,
    write_text,
    write_yaml,
)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def generate(urdf_path: Path, robot_cfg: Path, motor_profile_path: Path, fitted_path: Path | None) -> dict:
    urdf = parse_urdf(urdf_path)
    default_pos = load_default_joint_pos(robot_cfg)
    legs, warnings = leg_report(urdf, default_pos)
    links = urdf["links"]
    joints = urdf["joints"]
    motor_profile = read_yaml(motor_profile_path)
    fitted = read_yaml(fitted_path) if fitted_path and fitted_path.exists() else {}

    trunk = links.get("Trunk", {})
    trunk_geom = trunk.get("collision_geometry", {})
    trunk_size = trunk_geom.get("size", [0.26, 0.16, 0.10])
    base_mass = trunk.get("mass") or 2.0
    if trunk.get("mass") is None:
        warnings.append("Trunk mass missing; using conservative base mass randomization.")
    nominal_leg = sum(v["nominal_leg_length_m"] for v in legs.values()) / max(len(legs), 1)
    body_height = 0.293
    step_height = clamp(0.10 * nominal_leg, 0.025, 0.040)
    step_length_max = clamp(0.18 * nominal_leg, 0.030, 0.060)

    joint_zero = {}
    reset_noise_by_joint = {}
    tight_joints = []
    for name in JOINT_ORDER:
        item = joints.get(name)
        if item is None:
            warnings.append(f"missing joint for randomization: {name}")
            continue
        lower = item["limit"]["lower"]
        upper = item["limit"]["upper"]
        default = default_pos.get(name)
        if lower is None or upper is None or default is None:
            warnings.append(f"{name}: missing limit/default; zero/reset randomization disabled for this joint.")
            continue
        joint_range = upper - lower
        margin = max(0.0, min(default - lower, upper - default))
        zero = min(0.03, 0.03 * joint_range, 0.5 * margin)
        reset = min(0.05, 0.05 * joint_range, 0.5 * margin)
        if margin < 0.08:
            tight_joints.append(name)
        joint_zero[name] = [-round(zero, 5), round(zero, 5)]
        reset_noise_by_joint[name] = [-round(reset, 5), round(reset, 5)]

    safe_torque = (
        fitted.get("control_limits", {}).get("torque_limit_train_nm")
        or motor_profile.get("motor", {}).get("safe_training_torque_nm")
        or motor_profile.get("safety", {}).get("torque_limit_for_training_nm")
        or 5.0
    )
    urdf_effort = min(
        item["limit"]["effort"]
        for name, item in joints.items()
        if name in JOINT_ORDER and item["limit"]["effort"] is not None
    )
    effective_torque = min(float(urdf_effort), float(safe_torque))
    if urdf_effort < safe_torque:
        warnings.append(f"URDF effort limit {urdf_effort} Nm is below safe motor torque {safe_torque} Nm.")
    if tight_joints:
        warnings.append(f"default pose close to limits, randomization tightened: {tight_joints}")

    max_com_x = min(0.015, max(0.005, 0.03 * float(trunk_size[0])))
    max_com_y = min(0.012, max(0.005, 0.03 * float(trunk_size[1])))
    max_com_z = min(0.010, max(0.004, 0.02 * body_height))
    height_noise = min(0.01, 0.25 * step_height, 0.02 * nominal_leg)
    push_lo = round(0.03 * base_mass * 9.81, 3)
    push_hi = round(0.08 * base_mass * 9.81, 3)

    profile = {
        "source": {
            "urdf": str(urdf_path),
            "robot_cfg": str(robot_cfg),
            "motor_profile": str(motor_profile_path),
            "motor_profile_fitted": str(fitted_path) if fitted_path else None,
        },
        "model_summary": {
            "base_mass_kg": base_mass,
            "trunk_size_m": trunk_size,
            "nominal_leg_length_m": nominal_leg,
            "nominal_body_height_m": body_height,
            "effective_torque_limit_nm": effective_torque,
        },
        "mass_randomization": {
            "base_mass_scale": [0.92, 1.08],
            "thigh_mass_scale": [0.95, 1.05],
            "calf_mass_scale": [0.95, 1.05],
            "foot_mass_scale": [0.95, 1.05],
        },
        "inertia_randomization": {
            "enable": True,
            "base_inertia_scale": [0.92, 1.08],
            "leg_inertia_scale": [0.95, 1.05],
        },
        "com_randomization": {
            "base_com_x_m": [-round(max_com_x, 5), round(max_com_x, 5)],
            "base_com_y_m": [-round(max_com_y, 5), round(max_com_y, 5)],
            "base_com_z_m": [-round(max_com_z, 5), round(max_com_z, 5)],
        },
        "joint_zero_offset_randomization": joint_zero,
        "reset_joint_position_noise": reset_noise_by_joint,
        "joint_dynamics_randomization": {
            "damping_scale": [0.8, 1.2],
            "friction_scale": [0.8, 1.2],
            "additive_friction_nm": {"hip": [0.0, 0.03], "thigh": [0.0, 0.05], "calf": [0.0, 0.05]},
        },
        "motor_strength_randomization": {
            "global_scale": fitted.get("training_randomization", {}).get("motor_strength_range", [0.65, 1.0]),
            "per_leg_scale": [0.75, 1.0],
            "per_joint_scale": {"hip": [0.75, 1.0], "thigh": [0.65, 1.0], "calf": [0.65, 1.0]},
        },
        "torque_limit_randomization": {
            "scale": [0.7, 1.0],
            "absolute_train_limit_nm": effective_torque,
        },
        "pd_gain_randomization": {
            "kp_scale": fitted.get("training_randomization", {}).get("kp_scale_range", [0.85, 1.15]),
            "kd_scale": fitted.get("training_randomization", {}).get("kd_scale_range", [0.85, 1.15]),
        },
        "action_delay_randomization": {
            "frames": fitted.get("training_randomization", {}).get("action_delay_frames", [0, 3]),
            "estimated_ms": fitted.get("actuator_dynamics", {}).get("delay_random_range_ms", [0, 60]),
        },
        "terrain_randomization": {"height_noise_m": [0.0, round(height_noise, 5)]},
        "ground_slope_randomization": {"roll_deg": [-3, 3], "pitch_deg": [-3, 3]},
        "contact_randomization": {"ground_friction": [0.6, 1.2], "restitution": [0.0, 0.1]},
        "reset_base_randomization": {"height_m": [-0.015, 0.015], "roll_deg": [-3, 3], "pitch_deg": [-3, 3], "yaw_deg": [-5, 5]},
        "push_randomization": {
            "enable": True,
            "force_n": [push_lo, push_hi],
            "force_scale_body_weight": [0.03, 0.08],
            "interval_s": [3.0, 6.0],
        },
        "cpg_limits": {
            "freq_min_hz": fitted.get("cpg_limits", {}).get("freq_min_hz", 0.8),
            "freq_max_hz": fitted.get("cpg_limits", {}).get("freq_max_hz", 1.8),
            "step_height_m": fitted.get("cpg_limits", {}).get("step_height_m", round(step_height, 5)),
            "step_length_min_m": 0.015,
            "step_length_max_m": fitted.get("cpg_limits", {}).get("step_length_max_m", round(step_length_max, 5)),
        },
        "cpg_parameter_randomization": {
            "step_height_scale": [0.85, 1.15],
            "step_length_scale": [0.85, 1.15],
            "frequency_scale": [0.90, 1.10],
            "duty_factor": [0.56, 0.66],
            "initial_phase_random": True,
        },
        "curriculum": {
            "stage_1_cpg_only": {"randomization_level": "none_or_minimal"},
            "stage_2_cpg_residual_basic": {"enable": ["command_randomization", "initial_phase_randomization", "small_reset_noise"]},
            "stage_3_motor_random": {"enable": ["motor_strength_randomization", "action_delay_randomization", "pd_gain_randomization", "joint_zero_offset_randomization"]},
            "stage_4_model_random": {"enable": ["mass_randomization", "inertia_randomization", "com_randomization", "joint_friction_damping_randomization"]},
            "stage_5_environment_random": {"enable": ["ground_friction_randomization", "small_terrain_noise", "small_push", "small_slope"]},
            "stage_6_eval_unseen": {"enable": ["unseen_motor_strength", "unseen_delay", "unseen_friction", "unseen_zero_offset"]},
        },
        "default_enabled_first_version": [
            "command_randomization",
            "cpg_initial_phase_randomization",
            "small_reset_pose_noise",
            "action_delay_randomization",
            "motor_strength_randomization",
            "kp_kd_randomization",
            "joint_zero_offset_randomization",
        ],
        "warnings": warnings,
    }
    return profile


def render_text(profile: dict) -> str:
    lines = ["Randomization from URDF report", ""]
    lines.append("Sources:")
    for key, value in profile["source"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("URDF-derived ranges: mass, inertia, CoM, joint zero/reset noise, terrain scale, push scale.")
    lines.append("Motor-profile-derived ranges: torque limit, motor strength, PD gains, action delay when fitted data exists.")
    lines.append("Conservative defaults: friction, restitution, slope, CPG duty-factor scale.")
    lines.append("")
    for warning in profile["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append("")
    lines.append("First version should enable only the items in default_enabled_first_version.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=str(default_urdf_path()))
    parser.add_argument("--robot-cfg", default=str(default_robot_cfg_path()))
    parser.add_argument("--motor-profile", default=str(package_dir() / "config" / "motor_profile.yaml"))
    parser.add_argument("--motor-profile-fitted", default=str(package_dir() / "config" / "motor_profile_fitted.yaml"))
    parser.add_argument("--out-dir", default=str(package_dir()))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    profile = generate(Path(args.urdf), Path(args.robot_cfg), Path(args.motor_profile), Path(args.motor_profile_fitted))
    write_yaml(out_dir / "config" / "randomization_profile.yaml", profile)
    write_yaml(out_dir / "logs" / "randomization_from_urdf_report.yaml", profile)
    write_text(out_dir / "logs" / "randomization_from_urdf_report.txt", render_text(profile))
    print(f"wrote {out_dir / 'config' / 'randomization_profile.yaml'}")


if __name__ == "__main__":
    main()
