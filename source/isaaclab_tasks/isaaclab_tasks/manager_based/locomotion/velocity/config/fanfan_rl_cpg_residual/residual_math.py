from __future__ import annotations

import torch


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
