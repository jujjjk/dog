from __future__ import annotations

from collections.abc import Sequence


WAVE_CURRICULUM_STAGES = (
    {
        "stage": 1,
        "start_iter": 0,
        "end_iter": 5_000,
        "lin_vel_x": (0.10, 0.15),
        "standing": 0.18,
        "swing_contact": -0.10,
        "stance_loss": -0.05,
        "mass_delta": (0.0, 0.0),
        "joint_friction": (0.08, 0.08),
        "armature": (0.010, 0.010),
        "actuator_gain": (1.0, 1.0),
        "motor_strength": (1.0, 1.0),
        "delay_steps": (0, 0),
        "reset_tilt": 0.010,
        "noise": {"base_ang_vel": 0.0, "projected_gravity": 0.0, "joint_pos": 0.0, "joint_vel": 0.0},
        "noise_level": 0.0,
        "push_enabled": 0.0,
    },
    {
        "stage": 2,
        "start_iter": 5_000,
        "end_iter": 30_000,
        "lin_vel_x": (0.10, 0.18),
        "standing": 0.10,
        "swing_contact": -0.10,
        "stance_loss": -0.05,
        "mass_delta": (-0.10, 0.10),
        "joint_friction": (0.06, 0.10),
        "armature": (0.008, 0.012),
        "actuator_gain": (0.97, 1.03),
        "motor_strength": (0.97, 1.03),
        "delay_steps": (0, 0),
        "reset_tilt": 0.025,
        "noise": {"base_ang_vel": 0.0, "projected_gravity": 0.0, "joint_pos": 0.0, "joint_vel": 0.0},
        "noise_level": 0.0,
        "push_enabled": 0.0,
    },
    {
        "stage": 3,
        "start_iter": 30_000,
        "end_iter": 60_000,
        "lin_vel_x": (0.12, 0.20),
        "standing": 0.05,
        "swing_contact": -0.30,
        "stance_loss": -0.15,
        "mass_delta": (-0.20, 0.20),
        "joint_friction": (0.04, 0.12),
        "armature": (0.006, 0.016),
        "actuator_gain": (0.90, 1.10),
        "motor_strength": (0.95, 1.05),
        "delay_steps": (0, 2),
        "reset_tilt": 0.050,
        "noise": {"base_ang_vel": 0.05, "projected_gravity": 0.02, "joint_pos": 0.005, "joint_vel": 0.15},
        "noise_level": 1.0,
        "push_enabled": 0.0,
    },
    {
        "stage": 4,
        "start_iter": 60_000,
        "end_iter": None,
        "lin_vel_x": (0.10, 0.22),
        "standing": 0.05,
        "swing_contact": -0.60,
        "stance_loss": -0.30,
        "mass_delta": (-0.30, 0.30),
        "joint_friction": (0.03, 0.15),
        "armature": (0.005, 0.020),
        "actuator_gain": (0.85, 1.15),
        "motor_strength": (0.90, 1.05),
        "delay_steps": (0, 3),
        "reset_tilt": 0.080,
        "noise": {"base_ang_vel": 0.07, "projected_gravity": 0.03, "joint_pos": 0.008, "joint_vel": 0.20},
        "noise_level": 1.5,
        "push_enabled": 1.0,
    },
)


def get_wave_stage(iteration: int, stages: Sequence[dict] = WAVE_CURRICULUM_STAGES) -> dict:
    for stage in stages:
        if iteration >= stage["start_iter"] and (stage["end_iter"] is None or iteration < stage["end_iter"]):
            return stage
    return stages[-1]


def get_wave_stage_number(stage_number: int, stages: Sequence[dict] = WAVE_CURRICULUM_STAGES) -> dict:
    for stage in stages:
        if int(stage["stage"]) == int(stage_number):
            return stage
    return stages[-1]


def reference_scales(command_x: float) -> tuple[float, float, float]:
    """Return stride, frequency, and swing-height scales for curriculum logging."""
    command_x = max(0.0, float(command_x))
    nominal = min(command_x / 0.15, 1.0)
    overspeed_x = min(max((command_x - 0.15) / 0.03, 0.0), 1.0)
    overspeed = overspeed_x * overspeed_x * (3.0 - 2.0 * overspeed_x)
    stride_scale = nominal + 0.20 * overspeed
    frequency_scale = 0.35 + 0.65 * nominal**0.5 + 0.10 * overspeed
    swing_scale = (0.030 + 0.042 * nominal) / 0.072
    return stride_scale, frequency_scale, swing_scale
