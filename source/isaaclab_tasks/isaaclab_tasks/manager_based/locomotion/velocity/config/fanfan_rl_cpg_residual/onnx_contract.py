from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .joint_semantics import FanfanJointSemanticAdapter, POLICY_JOINT_NAMES
from .reference_gait import FanfanReferenceGaitCfg


OBSERVATION_ORDER = [
    "base_ang_vel[3]", "projected_gravity[3]", "command[3]",
    "joint_pos_rel[12]", "joint_vel[12]", "last_action[12]",
    "q_ref[12]", "q_ref_minus_joint_pos[12]",
    "per_leg_phase_sin_cos[8]", "active_leg_one_hot[4]",
]
RESIDUAL_SCALE = [0.05, 0.08, 0.10] * 4


class ResidualOnnxWrapper(torch.nn.Module):
    """Wrap an actor so ONNX emits scaled joint residuals, never absolute targets."""

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.register_buffer("residual_scale", torch.tensor(RESIDUAL_SCALE))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observations)) * self.residual_scale


def export_residual_policy_as_onnx(
    policy,
    output_dir: str | Path,
    normalizer=None,
    filename: str = "policy.onnx",
) -> Path:
    """Export a feed-forward RSL-RL policy as a scaled 12-D residual."""
    if bool(getattr(policy, "is_recurrent", False)):
        raise NotImplementedError("The Fanfan residual ONNX contract currently supports the configured MLP actor only.")
    if hasattr(policy, "actor"):
        actor = copy.deepcopy(policy.actor)
    elif hasattr(policy, "student"):
        actor = copy.deepcopy(policy.student)
    else:
        raise ValueError("RSL-RL policy has no actor/student module.")

    normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
    model = ResidualOnnxWrapper(torch.nn.Sequential(normalizer, actor)).cpu().eval()
    first_linear = next(module for module in actor.modules() if isinstance(module, torch.nn.Linear))
    observations = torch.zeros(1, first_linear.in_features)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    torch.onnx.export(
        model,
        observations,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["observations"],
        output_names=["residual_rad"],
        dynamic_axes={},
    )
    write_deployment_contract(output_dir)
    return output_path


def write_deployment_contract(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FanfanReferenceGaitCfg()
    semantic_adapter = FanfanJointSemanticAdapter()
    contract = {
        "warning": "ONNX output is a residual. Never send it directly to motors.",
        "observation_dim": 81,
        "observation_order": OBSERVATION_ORDER,
        "onnx_output": "scaled_joint_residual_rad[12]",
        "onnx_output_joint_order": list(POLICY_JOINT_NAMES),
        "residual_scale_rad": RESIDUAL_SCALE,
        "residual_lowpass_alpha": 0.30,
        "reference_gait": vars(cfg),
        "joint_semantics": semantic_adapter.contract_metadata(),
        "safety": {
            "target_rate_limit_rad_s": [1.9, 2.1],
            "training_torque_budget_nm": [7.0, 10.0],
            "training_short_peak_nm": [10.0, 14.0],
            "motor_delay_steps": [0, 2],
        },
        "deployment_chain": [
            "FanfanReferenceGait -> q_ref",
            "ONNX -> residual",
            "residual low-pass",
            "q_ref + residual",
            "joint/rate/torque safety filter",
            "deployment-only policy target -> real motor mapper",
            "motor target",
        ],
    }
    path = output_dir / "fanfan_residual_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path
