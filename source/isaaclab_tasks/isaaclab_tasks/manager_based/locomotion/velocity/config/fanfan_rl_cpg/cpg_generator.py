from __future__ import annotations

import math
from typing import Any

import torch

from .cpg_cfg import CPGCfg


class QuadrupedCPG:
    """Small, vectorized CPG for Fanfan joint-space residual training."""

    def __init__(
        self,
        cfg: CPGCfg,
        device: str | torch.device,
        num_envs: int,
        dt: float,
        default_joint_pos: torch.Tensor,
        joint_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
        urdf_params: dict[str, Any] | None = None,
        motor_profile: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.default_joint_pos = default_joint_pos.to(self.device).clone()
        if self.default_joint_pos.ndim == 1:
            self.default_joint_pos = self.default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.joint_limits = joint_limits
        self.urdf_params = urdf_params or {}
        self.motor_profile = motor_profile or {}

        self.base_phase = torch.zeros(self.num_envs, device=self.device)
        self.last_frequency = torch.zeros(self.num_envs, device=self.device)
        self.last_step_length = torch.zeros(self.num_envs, device=self.device)
        self.last_step_height = torch.full((self.num_envs,), float(cfg.step_height), device=self.device)
        self.last_leg_phase = torch.zeros(self.num_envs, len(cfg.leg_order), device=self.device)
        self.last_q_cpg = self.default_joint_pos.clone()
        self.last_q_cpg_before_clip = self.default_joint_pos.clone()
        self.last_hip_stride_delta = torch.zeros(
            self.num_envs, len(cfg.leg_order), device=self.device, dtype=self.default_joint_pos.dtype
        )
        self.last_hip_balance_delta = torch.zeros_like(self.last_hip_stride_delta)

        self._residual_limits = self._make_joint_type_tensor(
            hip=float(cfg.residual_limit_hip),
            thigh=float(cfg.residual_limit_thigh),
            calf=float(cfg.residual_limit_calf),
            fallback=float(cfg.residual_limit),
        )
        self._joint_signs = torch.tensor(cfg.joint_signs, device=self.device, dtype=self.default_joint_pos.dtype).view(
            1, -1
        )
        self._joint_offsets = torch.tensor(
            cfg.joint_offsets, device=self.device, dtype=self.default_joint_pos.dtype
        ).view(1, -1)
        self._hip_balance_signs = torch.tensor(
            cfg.joint_sine.hip_balance_signs, device=self.device, dtype=self.default_joint_pos.dtype
        ).view(1, -1)
        if self._hip_balance_signs.shape[1] != len(cfg.leg_order):
            raise ValueError(
                "hip_balance_signs must have one value per leg in cfg.leg_order "
                f"({cfg.leg_order}), got {cfg.joint_sine.hip_balance_signs}."
            )

    @property
    def residual_limits(self) -> torch.Tensor:
        return self._residual_limits

    def _make_joint_type_tensor(self, hip: float, thigh: float, calf: float, fallback: float) -> torch.Tensor:
        values = []
        for name in self.cfg.joint_order:
            if "hip" in name:
                values.append(hip)
            elif "thigh" in name:
                values.append(thigh)
            elif "calf" in name or "knee" in name:
                values.append(calf)
            else:
                values.append(fallback)
        return torch.tensor(values, device=self.device, dtype=self.default_joint_pos.dtype).view(1, -1)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if bool(self.cfg.initial_phase_random):
            self.base_phase[env_ids_tensor] = torch.rand(env_ids_tensor.numel(), device=self.device) * (2.0 * math.pi)
        else:
            self.base_phase[env_ids_tensor] = 0.0
        self.last_q_cpg[env_ids_tensor] = self.default_joint_pos[env_ids_tensor]
        self.last_q_cpg_before_clip[env_ids_tensor] = self.default_joint_pos[env_ids_tensor]
        self.last_hip_stride_delta[env_ids_tensor] = 0.0
        self.last_hip_balance_delta[env_ids_tensor] = 0.0

    def compute_frequency(self, commands: torch.Tensor) -> torch.Tensor:
        cmd_x = torch.abs(commands[:, 0])
        standing = cmd_x < float(self.cfg.standing_cmd_threshold)
        if self.cfg.freq_mode == "velocity_step_length":
            step_length = torch.clamp(
                float(self.cfg.k_step) * cmd_x,
                min=float(self.cfg.step_length_min),
                max=float(self.cfg.step_length_max),
            )
            freq = cmd_x / torch.clamp(step_length, min=1.0e-5)
        else:
            step_length = torch.clamp(
                float(self.cfg.k_step) * cmd_x,
                min=float(self.cfg.step_length_min),
                max=float(self.cfg.step_length_max),
            )
            freq = float(self.cfg.freq_min) + float(self.cfg.k_freq) * cmd_x
        freq = torch.clamp(freq, min=float(self.cfg.freq_min), max=float(self.cfg.freq_max))
        freq = torch.where(standing, torch.zeros_like(freq), freq)
        self.last_frequency = freq
        self.last_step_length = torch.where(standing, torch.zeros_like(step_length), step_length)
        return freq

    def compute_phase(self, commands: torch.Tensor) -> torch.Tensor:
        frequency = self.compute_frequency(commands)
        self.base_phase = torch.remainder(self.base_phase + 2.0 * math.pi * frequency * self.dt, 2.0 * math.pi)
        offsets = self.cfg.phase_offsets.get(self.cfg.gait, self.cfg.phase_offsets["trot"])
        phase_cols = []
        for leg in self.cfg.leg_order:
            phase_cols.append(torch.remainder(self.base_phase + 2.0 * math.pi * float(offsets[leg]), 2.0 * math.pi))
        self.last_leg_phase = torch.stack(phase_cols, dim=1)
        return self.last_leg_phase

    def compute_joint_sine(self, leg_phase: torch.Tensor) -> torch.Tensor:
        q = self.default_joint_pos.clone()
        hip_amp = float(self.cfg.joint_sine.hip_amp)
        thigh_amp = float(self.cfg.joint_sine.thigh_amp)
        duty = float(self.cfg.duty_factor)
        swing_fraction = max(1.0 - duty, 0.05)
        lift_calf_amp = float(self.cfg.joint_sine.swing_lift_calf_amp)
        stance_calf_amp = float(self.cfg.joint_sine.stance_calf_amp)
        stride_sign = float(self.cfg.joint_sine.stride_sign)
        enable_hip_balance = bool(self.cfg.joint_sine.enable_hip_balance)
        hip_stance_widen_amp = float(self.cfg.joint_sine.hip_stance_widen_amp)
        hip_swing_relax_amp = float(self.cfg.joint_sine.hip_swing_relax_amp)
        hip_balance_max_abs = float(self.cfg.joint_sine.hip_balance_max_abs)
        hip_balance_use_stance_mask = bool(self.cfg.joint_sine.hip_balance_use_stance_mask)
        use_sin_balance_shape = str(self.cfg.joint_sine.hip_balance_smooth_shape).lower() == "sin"
        moving = (self.last_frequency > 0.0).to(q.dtype).unsqueeze(1)
        hip_stride_deltas = torch.zeros_like(self.last_hip_stride_delta)
        hip_balance_deltas = torch.zeros_like(self.last_hip_balance_delta)

        for leg_i, leg in enumerate(self.cfg.leg_order):
            base = leg_i * 3
            phase = leg_phase[:, leg_i]
            phase01 = torch.remainder(phase / (2.0 * math.pi), 1.0)
            swing = phase01 < swing_fraction
            s_swing = torch.clamp(phase01 / swing_fraction, 0.0, 1.0)
            s_stance = torch.clamp((phase01 - swing_fraction) / max(1.0 - swing_fraction, 1.0e-5), 0.0, 1.0)
            swing_shape = torch.where(swing, torch.sin(math.pi * s_swing), torch.zeros_like(phase))
            stance_shape = torch.where(swing, torch.zeros_like(phase), torch.sin(math.pi * s_stance))
            stride = torch.where(swing, -1.0 + 2.0 * s_swing, 1.0 - 2.0 * s_stance)

            hip_stride_delta = hip_amp * stride
            hip_balance_delta = torch.zeros_like(hip_stride_delta)
            if enable_hip_balance:
                if use_sin_balance_shape:
                    stance_weight = stance_shape
                    swing_weight = swing_shape
                else:
                    stance_weight = (~swing).to(q.dtype)
                    swing_weight = swing.to(q.dtype)
                if not hip_balance_use_stance_mask:
                    stance_weight = torch.ones_like(stance_weight)
                hip_side_sign = self._hip_balance_signs[:, leg_i]
                stance_widen_delta = hip_side_sign * hip_stance_widen_amp * stance_weight
                swing_relax_delta = -hip_side_sign * hip_swing_relax_amp * swing_weight
                hip_balance_delta = torch.clamp(
                    stance_widen_delta + swing_relax_delta,
                    min=-hip_balance_max_abs,
                    max=hip_balance_max_abs,
                )

            hip_stride_deltas[:, leg_i] = hip_stride_delta
            hip_balance_deltas[:, leg_i] = hip_balance_delta
            q[:, base + 0] += moving[:, 0] * (hip_stride_delta + hip_balance_delta)
            q[:, base + 1] += moving[:, 0] * stride_sign * thigh_amp * stride
            q[:, base + 2] += moving[:, 0] * (-lift_calf_amp * swing_shape + stance_calf_amp * stance_shape)
        q = q * self._joint_signs + self._joint_offsets
        self.last_hip_stride_delta = hip_stride_deltas * moving
        self.last_hip_balance_delta = hip_balance_deltas * moving
        self.last_q_cpg_before_clip = q
        return self._clip_to_limits(q)

    def compute_foot_trajectory(self, leg_phase: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        phase01 = torch.remainder(leg_phase / (2.0 * math.pi), 1.0)
        duty = float(self.cfg.duty_factor)
        step = self.last_step_length.unsqueeze(1)
        z = torch.zeros_like(phase01)
        x = torch.zeros_like(phase01)
        swing = phase01 >= duty
        s_swing = torch.clamp((phase01 - duty) / max(1.0 - duty, 1.0e-5), 0.0, 1.0)
        s_stance = torch.clamp(phase01 / max(duty, 1.0e-5), 0.0, 1.0)
        x = torch.where(swing, -0.5 * step + step * s_swing, 0.5 * step - step * s_stance)
        z = torch.where(swing, float(self.cfg.step_height) * torch.sin(math.pi * s_swing), z)
        y = torch.zeros_like(x)
        return torch.stack((x, y, z), dim=-1)

    def inverse_kinematics(self, foot_pos: torch.Tensor) -> torch.Tensor:
        """Map URDF-frame foot offsets to simulation joint targets.

        The Fanfan URDF uses +Y axes for all thigh/calf joints.  In that
        convention a positive thigh angle moves the foot backward and slightly
        upward around the default stance, while a positive calf angle moves the
        foot downward.  This IK therefore solves directly in URDF joint space
        instead of reusing the real-motor sign map used by deployment.
        """
        q = self.default_joint_pos.clone()
        l1 = float(self.cfg.foot_ik_thigh_length)
        l2 = float(self.cfg.foot_ik_calf_length)
        max_reach = max(l1 + l2 - float(self.cfg.foot_ik_reach_margin), 1.0e-4)
        min_reach = max(abs(l1 - l2) + float(self.cfg.foot_ik_reach_margin), 1.0e-4)

        for leg_i, _leg in enumerate(self.cfg.leg_order):
            base = leg_i * 3
            q1_default = self.default_joint_pos[:, base + 1]
            q2_default = self.default_joint_pos[:, base + 2]
            default_x = -l1 * torch.sin(q1_default) - l2 * torch.sin(q1_default + q2_default)
            default_z = -l1 * torch.cos(q1_default) - l2 * torch.cos(q1_default + q2_default)

            target_x = default_x + foot_pos[:, leg_i, 0]
            target_z = default_z + foot_pos[:, leg_i, 2]
            reach = torch.sqrt(torch.clamp(target_x * target_x + target_z * target_z, min=1.0e-8))
            scale = torch.clamp(reach, min=min_reach, max=max_reach) / torch.clamp(reach, min=1.0e-8)
            target_x = target_x * scale
            target_z = target_z * scale

            x_forward = -target_x
            z_down = -target_z
            cos_knee = (x_forward * x_forward + z_down * z_down - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
            cos_knee = torch.clamp(cos_knee, -1.0, 1.0)
            q2 = -torch.acos(cos_knee)
            q1 = torch.atan2(x_forward, z_down) - torch.atan2(
                l2 * torch.sin(q2),
                l1 + l2 * torch.cos(q2),
            )

            q[:, base + 1] = q1
            q[:, base + 2] = q2
        return self._clip_to_limits(q)

    def update(self, commands: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        leg_phase = self.compute_phase(commands)
        if self.cfg.mode == "foot_ik":
            foot_pos = self.compute_foot_trajectory(leg_phase, commands)
            q_cpg = self.inverse_kinematics(foot_pos)
        else:
            q_cpg = self.compute_joint_sine(leg_phase)
        self.last_q_cpg = q_cpg
        return q_cpg

    def _clip_to_limits(self, q: torch.Tensor) -> torch.Tensor:
        if self.joint_limits is None or not bool(self.cfg.filter.use_joint_limit_clip):
            return q
        lower, upper = self.joint_limits
        return torch.clamp(q, lower.to(q.device), upper.to(q.device))
