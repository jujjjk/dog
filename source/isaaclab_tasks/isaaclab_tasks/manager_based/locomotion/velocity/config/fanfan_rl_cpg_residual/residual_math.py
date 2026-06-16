from __future__ import annotations

import torch


def smootherstep_scalar(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def rear_lift_phase_profile(
    elapsed: float,
    *,
    settle_sec: float,
    preload_sec: float,
    cycle_sec: float,
) -> tuple[int, float, float]:
    """Return phase id, support-preload gate, and lift gate."""
    elapsed = max(0.0, float(elapsed))
    settle_sec = max(0.0, float(settle_sec))
    preload_sec = max(0.0, float(preload_sec))
    cycle_sec = max(0.5, float(cycle_sec))
    if elapsed < settle_sec:
        return 0, 0.0, 0.0
    if elapsed < settle_sec + preload_sec:
        progress = (elapsed - settle_sec) / max(preload_sec, 1.0e-6)
        return 1, smootherstep_scalar(progress), 0.0
    cycle_phase = ((elapsed - settle_sec - preload_sec) % cycle_sec) / cycle_sec
    if cycle_phase < 0.5:
        lift = smootherstep_scalar(cycle_phase * 2.0)
    else:
        lift = 1.0 - smootherstep_scalar((cycle_phase - 0.5) * 2.0)
    return 2, 1.0, lift


def filter_residual(
    policy_output: torch.Tensor,
    previous_residual: torch.Tensor,
    residual_scale: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Convert policy output to a bounded, low-pass-filtered joint residual."""
    bounded = torch.tanh(policy_output) * residual_scale
    return (1.0 - float(alpha)) * previous_residual + float(alpha) * bounded


def clamp_joint_targets(
    targets: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp joint targets and return the per-joint clipping mask."""
    clamped = torch.clamp(targets, min=lower, max=upper)
    return clamped, torch.abs(clamped - targets) > 1.0e-6


def joint_mapping_index(
    step: int,
    *,
    control_dt: float,
    initial_hold_sec: float,
    active_hold_sec: float,
    rest_sec: float,
    num_joints: int = 12,
) -> int:
    """Return the active mapping-test joint, or -1 during default-pose holds."""
    initial_steps = max(0, round(float(initial_hold_sec) / float(control_dt)))
    if step < initial_steps:
        return -1
    active_steps = max(1, round(float(active_hold_sec) / float(control_dt)))
    rest_steps = max(1, round(float(rest_sec) / float(control_dt)))
    cycle_steps = active_steps + rest_steps
    offset = int(step) - initial_steps
    if offset % cycle_steps >= active_steps:
        return -1
    return (offset // cycle_steps) % int(num_joints)


def validate_reference_control_stage(stage: int, enable_vmc: bool, vmc_mode: str) -> None:
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"control_stage must be 0, 1, 2, or 3; got {stage}.")
    if vmc_mode not in ("off", "light", "full"):
        raise ValueError(f"vmc_mode must be off, light, or full; got {vmc_mode!r}.")
    if stage < 2 and (enable_vmc or vmc_mode != "off"):
        raise ValueError("Stage 0/1 must run with enable_vmc=False and vmc_mode='off'.")
    if stage == 2 and (not enable_vmc or vmc_mode != "light"):
        raise ValueError("Stage 2 requires enable_vmc=True and vmc_mode='light'.")
    if stage == 3 and (not enable_vmc or vmc_mode != "full"):
        raise ValueError("Stage 3 requires enable_vmc=True and vmc_mode='full'.")


def filter_vmc_delta(
    raw_delta: torch.Tensor,
    previous_delta: torch.Tensor,
    *,
    joint_limit_rad: float,
    rate_limit_rad_s: float,
    lowpass_alpha: float,
    dt: float,
) -> torch.Tensor:
    raw_delta = torch.clamp(raw_delta, min=-joint_limit_rad, max=joint_limit_rad)
    filtered = previous_delta + lowpass_alpha * (raw_delta - previous_delta)
    max_step = rate_limit_rad_s * dt
    return previous_delta + torch.clamp(filtered - previous_delta, min=-max_step, max=max_step)
