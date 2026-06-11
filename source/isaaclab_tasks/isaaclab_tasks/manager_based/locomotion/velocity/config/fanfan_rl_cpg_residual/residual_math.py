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
