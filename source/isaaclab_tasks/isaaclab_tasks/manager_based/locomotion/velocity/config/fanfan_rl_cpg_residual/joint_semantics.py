from __future__ import annotations

from dataclasses import dataclass

import torch


POLICY_LEG_ORDER = ("FR", "FL", "RR", "RL")
REAL_LEG_ORDER = ("FR", "FL", "RL", "RR")
JOINT_TYPES = ("hip", "thigh", "calf")

POLICY_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in POLICY_LEG_ORDER for joint in JOINT_TYPES
)
SIM_JOINT_NAMES = POLICY_JOINT_NAMES
REAL_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in REAL_LEG_ORDER for joint in JOINT_TYPES
)

# policy index -> real motor-array index
POLICY_TO_REAL_INDEX = (0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8)

# Hardware signs are expressed in policy order. They are deployment metadata only.
REAL_JOINT_SIGN_POLICY_ORDER = (
    -1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0,
    1.0, -1.0, -1.0,
)
REAL_ZERO_OFFSET_POLICY_ORDER = (0.0,) * 12

# The current URDF uses the same angle convention as the policy.
SIM_JOINT_SIGN_POLICY_ORDER = (1.0,) * 12
SIM_JOINT_OFFSET_POLICY_ORDER = (0.0,) * 12
SIM_HIP_SIDE_SIGNS = (-1.0, 1.0, -1.0, 1.0)


@dataclass
class FanfanJointSemanticCfg:
    sim_joint_sign_policy_order: tuple[float, ...] = SIM_JOINT_SIGN_POLICY_ORDER
    sim_joint_offset_policy_order: tuple[float, ...] = SIM_JOINT_OFFSET_POLICY_ORDER
    real_joint_sign_policy_order: tuple[float, ...] = REAL_JOINT_SIGN_POLICY_ORDER
    real_zero_offset_policy_order: tuple[float, ...] = REAL_ZERO_OFFSET_POLICY_ORDER
    policy_to_real_index: tuple[int, ...] = POLICY_TO_REAL_INDEX


class FanfanJointSemanticAdapter:
    """Convert Fanfan joint tensors without leaking hardware semantics into simulation."""

    def __init__(
        self,
        cfg: FanfanJointSemanticCfg | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg or FanfanJointSemanticCfg()
        self.device = torch.device(device)
        self.dtype = dtype

        self.sim_sign = self._tensor(self.cfg.sim_joint_sign_policy_order, "sim_joint_sign")
        self.sim_offset = self._tensor(self.cfg.sim_joint_offset_policy_order, "sim_joint_offset")
        self.real_sign = self._tensor(self.cfg.real_joint_sign_policy_order, "real_joint_sign")
        self.real_zero = self._tensor(self.cfg.real_zero_offset_policy_order, "real_zero_offset")
        if torch.any(self.sim_sign == 0.0) or torch.any(self.real_sign == 0.0):
            raise ValueError("Joint semantic signs must be non-zero.")

        indices = tuple(int(index) for index in self.cfg.policy_to_real_index)
        if sorted(indices) != list(range(12)):
            raise ValueError(f"policy_to_real_index must be a permutation of 0..11, got {indices}.")
        self.policy_to_real_index = torch.tensor(indices, device=self.device, dtype=torch.long)

    def _tensor(self, values: tuple[float, ...], name: str) -> torch.Tensor:
        if len(values) != 12:
            raise ValueError(f"{name} must contain 12 values, got {len(values)}.")
        return torch.tensor(values, device=self.device, dtype=self.dtype).unsqueeze(0)

    @staticmethod
    def assert_sim_joint_names(joint_names: tuple[str, ...] | list[str]) -> None:
        actual = tuple(joint_names)
        if actual != SIM_JOINT_NAMES:
            raise ValueError(
                "Residual action requires simulator joints in exact FR->FL->RR->RL order. "
                f"Expected {SIM_JOINT_NAMES}, got {actual}."
            )

    @staticmethod
    def _check_last_dim(values: torch.Tensor, name: str) -> None:
        if values.shape[-1] != 12:
            raise ValueError(f"{name} must have 12 joints in its last dimension, got {tuple(values.shape)}.")

    def policy_to_sim(self, q_policy: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(q_policy, "q_policy")
        return q_policy * self.sim_sign + self.sim_offset

    def sim_to_policy(self, q_sim: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(q_sim, "q_sim")
        return (q_sim - self.sim_offset) / self.sim_sign

    def sim_limits_to_policy(
        self, lower_sim: torch.Tensor, upper_sim: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower_candidate = self.sim_to_policy(lower_sim)
        upper_candidate = self.sim_to_policy(upper_sim)
        return torch.minimum(lower_candidate, upper_candidate), torch.maximum(lower_candidate, upper_candidate)

    def policy_to_real(self, q_policy: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(q_policy, "q_policy")
        real_values_policy_order = q_policy * self.real_sign + self.real_zero
        q_real = torch.empty_like(real_values_policy_order)
        q_real[..., self.policy_to_real_index] = real_values_policy_order
        return q_real

    def real_to_policy(self, q_real: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(q_real, "q_real")
        real_values_policy_order = q_real[..., self.policy_to_real_index]
        return (real_values_policy_order - self.real_zero) / self.real_sign

    def contract_metadata(self) -> dict[str, object]:
        return {
            "policy_joint_order": list(POLICY_JOINT_NAMES),
            "simulator_joint_order": list(SIM_JOINT_NAMES),
            "real_motor_joint_order": list(REAL_JOINT_NAMES),
            "policy_to_sim_sign": list(self.cfg.sim_joint_sign_policy_order),
            "policy_to_sim_offset_rad": list(self.cfg.sim_joint_offset_policy_order),
            "policy_to_real_index": list(self.cfg.policy_to_real_index),
            "policy_to_real_sign": list(self.cfg.real_joint_sign_policy_order),
            "policy_to_real_zero_offset_rad": list(self.cfg.real_zero_offset_policy_order),
            "hardware_mapping_boundary": "deployment_only",
        }
