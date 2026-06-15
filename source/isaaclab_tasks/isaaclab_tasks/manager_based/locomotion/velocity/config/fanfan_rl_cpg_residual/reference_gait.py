from __future__ import annotations

from dataclasses import dataclass
import math

import torch

LEG_ORDER = ("FR", "FL", "RR", "RL")
SWING_ORDER = ("RR", "FR", "RL", "FL")
SMALL_HIGH_FREQUENCY_SWING_START = {
    "RR": 0.00,
    "FR": 0.25,
    "RL": 0.50,
    "FL": 0.75,
}
LEGACY_REFERENCE_HIP_OUTWARD_SIGNS = (1.0, 1.0, -1.0, 1.0)
URDF_HIP_OUTWARD_SIGNS = (-1.0, 1.0, -1.0, 1.0)
LEG_START = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
FRONT_LEGS = ("FR", "FL")
REAR_LEGS = ("RR", "RL")
DIAGONAL_PARTNER = {"FR": "RL", "FL": "RR", "RR": "FL", "RL": "FR"}
SAME_SIDE_REAR = {"FR": "RR", "FL": "RL"}


@dataclass
class FanfanReferenceGaitCfg:
    step_hz: float = 0.62
    stride_length: float = 0.038
    swing_height: float = 0.072
    duty_factor: float = 0.78
    warmup_sec: float = 5.0
    command_full_speed: float = 0.15
    command_gate_start: float = 0.005
    command_gate_end: float = 0.030
    command_overspeed_end: float = 0.18
    max_stride_scale: float = 1.20
    max_frequency_scale: float = 1.10
    small_high_frequency_mode: bool = False
    reference_rate_limit_rad_s: float = 0.0
    apply_default_pose_offsets: bool = True
    hip_outward_signs: tuple[float, float, float, float] = LEGACY_REFERENCE_HIP_OUTWARD_SIGNS

    thigh_length: float = 0.1560608
    calf_length: float = 0.1489418
    workspace_margin_m: float = 1.0e-5
    front_stride_gain: float = 0.92
    rear_stride_gain: float = 0.82
    front_swing_height_gain: float = 1.26
    rear_swing_height_gain: float = 1.12
    front_calf_lift_extra: float = 0.210
    rear_calf_lift_extra: float = 0.165
    front_thigh_delta_scale: float = 0.18
    rear_thigh_delta_scale: float = 0.16

    hip_default_scale: float = 0.38
    hip_default_inward_offset: float = 0.010
    front_calf_min_rad: float = -1.12
    rear_hip_default_outward_offset: float = 0.006
    rear_thigh_default_back_offset: float = 0.045

    preload_fraction: float = 0.12
    front_load_active_hold_until: float = 0.82
    front_load_post_touchdown_hold: float = 0.04
    advance_start: float = 0.16
    advance_end: float = 0.84
    front_x_bias: float = 0.004
    front_z_extend: float = -0.002
    front_swing_forward_unfold: float = 0.018
    support_stand_tall_m: float = 0.006
    diag_support_preload_z_m: float = 0.012
    diag_support_calf_push_amp: float = 0.030
    diag_support_thigh_back_amp: float = 0.014
    diag_support_hip_amp: float = 0.018
    same_rear_unload_z_m: float = 0.006
    same_rear_calf_relief_amp: float = 0.018
    same_rear_unload_hip_amp: float = 0.014
    other_support_scale: float = 0.28
    front_swing_body_x_shift_m: float = 0.018
    front_swing_body_y_shift_m: float = 0.022
    rear_swing_body_x_shift_m: float = 0.010
    rear_swing_body_y_shift_m: float = 0.012


@dataclass
class FanfanSmallHighFreqReferenceGaitCfg(FanfanReferenceGaitCfg):
    step_hz: float = 0.95
    stride_length: float = 0.024
    swing_height: float = 0.050
    duty_factor: float = 0.78
    warmup_sec: float = 2.0
    small_high_frequency_mode: bool = True
    reference_rate_limit_rad_s: float = 0.0
    apply_default_pose_offsets: bool = False
    hip_outward_signs: tuple[float, float, float, float] = URDF_HIP_OUTWARD_SIGNS

    front_stride_gain: float = 1.00
    rear_stride_gain: float = 0.80
    front_swing_height_gain: float = 1.05
    rear_swing_height_gain: float = 0.64
    rear_lift_rise_fraction: float = 0.42
    rear_lift_fall_start: float = 0.58
    front_calf_lift_extra: float = 0.147
    rear_calf_lift_extra: float = 0.116
    front_thigh_delta_scale: float = 0.18
    rear_thigh_delta_scale: float = 0.16

    preload_fraction: float = 0.10
    post_touchdown_hold: float = 0.04
    front_x_bias: float = 0.003
    front_z_extend: float = -0.0014
    front_swing_forward_unfold: float = 0.0126
    support_stand_tall_m: float = 0.0042
    diag_support_preload_z_m: float = 0.0084
    diag_support_calf_push_amp: float = 0.021
    diag_support_thigh_back_amp: float = 0.0098
    diag_support_hip_amp: float = 0.0126
    same_rear_unload_z_m: float = 0.0
    same_rear_calf_relief_amp: float = 0.0126
    same_rear_unload_hip_amp: float = 0.0098
    front_swing_body_x_shift_m: float = 0.0126
    front_swing_body_y_shift_m: float = 0.0154
    rear_swing_body_x_shift_m: float = 0.007
    rear_swing_body_y_shift_m: float = 0.0084
    workspace_margin_m: float = 0.005

    def validate_parameters(self) -> None:
        ranges = {
            "step_hz": (0.75, 1.15),
            "stride_length": (0.024, 0.030),
            "swing_height": (0.045, 0.070),
            "duty_factor": (0.74, 0.80),
        }
        for name, (lower, upper) in ranges.items():
            value = float(getattr(self, name))
            if not lower <= value <= upper:
                raise ValueError(f"small_high_freq {name}={value} is outside [{lower}, {upper}].")


class FanfanReferenceGait:
    """Vectorized one-leg-at-a-time wave gait shared by simulation and deployment."""

    def __init__(
        self,
        cfg: FanfanReferenceGaitCfg,
        num_envs: int,
        device: str | torch.device,
        dt: float,
        default_joint_pos: torch.Tensor,
        joint_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.cfg = cfg
        validate_parameters = getattr(self.cfg, "validate_parameters", None)
        if validate_parameters is not None:
            validate_parameters()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.default_joint_pos = default_joint_pos.to(self.device).clone()
        if self.default_joint_pos.ndim == 1:
            self.default_joint_pos = self.default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        if self.cfg.apply_default_pose_offsets:
            self.default_joint_pos = self._apply_default_pose_offsets(self.default_joint_pos)
        self.joint_limits = joint_limits

        self.base_phase = torch.zeros(self.num_envs, device=self.device)
        self.walk_time = torch.zeros(self.num_envs, device=self.device)
        self.last_q_ref = self.default_joint_pos.clone()
        self.last_walk_gate = torch.zeros(self.num_envs, device=self.device)
        self.last_frequency = torch.zeros(self.num_envs, device=self.device)
        self.last_stride = torch.zeros(self.num_envs, device=self.device)
        self.last_swing_height = torch.zeros(self.num_envs, device=self.device)
        self.last_overspeed_scale = torch.zeros(self.num_envs, device=self.device)
        self.last_leg_phase = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_swing_mask = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.last_active_swing_one_hot = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_preload_gate = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_post_touchdown_gate = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_support_gate = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_warmup = torch.zeros(self.num_envs, device=self.device)
        self.last_body_shift = torch.zeros(self.num_envs, 2, device=self.device)
        self.default_foot_x, self.default_foot_z = self._forward_sagittal(
            self.default_joint_pos[:, 1::3],
            self.default_joint_pos[:, 2::3],
        )
        self.last_predicted_foot_z = self.default_foot_z.clone()
        self.last_predicted_foot_lift = torch.zeros_like(self.default_foot_z)

    @staticmethod
    def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _smootherstep01(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, 0.0, 1.0)
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)

    def _apply_default_pose_offsets(self, q: torch.Tensor) -> torch.Tensor:
        q = q.clone()
        normal_sign = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=q.device, dtype=q.dtype)
        hip_ids = torch.tensor((0, 3, 6, 9), device=q.device)
        q[:, hip_ids] *= float(self.cfg.hip_default_scale)
        q[:, hip_ids] -= normal_sign * float(self.cfg.hip_default_inward_offset)
        q[:, 6] -= float(self.cfg.rear_hip_default_outward_offset)
        q[:, 9] += float(self.cfg.rear_hip_default_outward_offset)
        q[:, 7] += float(self.cfg.rear_thigh_default_back_offset)
        q[:, 10] += float(self.cfg.rear_thigh_default_back_offset)
        q[:, 2] = torch.clamp(q[:, 2], min=float(self.cfg.front_calf_min_rad))
        q[:, 5] = torch.clamp(q[:, 5], min=float(self.cfg.front_calf_min_rad))
        return q

    def _forward_sagittal(self, thigh: torch.Tensor, calf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = -float(self.cfg.thigh_length) * torch.sin(thigh)
        x -= float(self.cfg.calf_length) * torch.sin(thigh + calf)
        z = -float(self.cfg.thigh_length) * torch.cos(thigh)
        z -= float(self.cfg.calf_length) * torch.cos(thigh + calf)
        return x, z

    def _inverse_sagittal(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l1 = float(self.cfg.thigh_length)
        l2 = float(self.cfg.calf_length)
        reach = torch.sqrt(torch.clamp(x * x + z * z, min=1.0e-8))
        workspace_margin = max(float(self.cfg.workspace_margin_m), 1.0e-5)
        min_reach = abs(l1 - l2) + workspace_margin
        max_reach = l1 + l2 - workspace_margin
        scale = torch.clamp(reach, min=min_reach, max=max_reach) / reach
        x = x * scale
        z = z * scale
        cos_calf = torch.clamp((x * x + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        calf = -torch.acos(cos_calf)
        thigh = torch.atan2(-x, -z) - torch.atan2(l2 * torch.sin(calf), l1 + l2 * torch.cos(calf))
        return thigh, calf

    def _solve_calf_for_z(self, thigh: torch.Tensor, z: torch.Tensor, calf_default: torch.Tensor) -> torch.Tensor:
        l1 = float(self.cfg.thigh_length)
        l2 = float(self.cfg.calf_length)
        value = torch.clamp((-z - l1 * torch.cos(thigh)) / l2, -1.0, 1.0)
        angle = torch.acos(value)
        candidate_a = angle - thigh
        candidate_b = -angle - thigh
        return torch.where(
            torch.abs(candidate_a - calf_default) < torch.abs(candidate_b - calf_default),
            candidate_a,
            candidate_b,
        )

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.base_phase[env_ids] = 0.0
        self.walk_time[env_ids] = 0.0
        self.last_q_ref[env_ids] = self.default_joint_pos[env_ids]
        self.last_walk_gate[env_ids] = 0.0
        self.last_frequency[env_ids] = 0.0
        self.last_stride[env_ids] = 0.0
        self.last_swing_height[env_ids] = 0.0
        self.last_overspeed_scale[env_ids] = 0.0
        self.last_leg_phase[env_ids] = 0.0
        self.last_swing_mask[env_ids] = False
        self.last_active_swing_one_hot[env_ids] = 0.0
        self.last_preload_gate[env_ids] = 0.0
        self.last_post_touchdown_gate[env_ids] = 0.0
        self.last_support_gate[env_ids] = 0.0
        self.last_warmup[env_ids] = 0.0
        self.last_body_shift[env_ids] = 0.0
        self.last_predicted_foot_z[env_ids] = self.default_foot_z[env_ids]
        self.last_predicted_foot_lift[env_ids] = 0.0

    def _command_parameters(self, commands: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cmd_x = torch.clamp(commands[:, 0], min=0.0)
        gate_input = (cmd_x - float(self.cfg.command_gate_start)) / max(
            float(self.cfg.command_gate_end - self.cfg.command_gate_start), 1.0e-6
        )
        walk_gate = self._smoothstep01(gate_input)
        u = torch.clamp(cmd_x / float(self.cfg.command_full_speed), 0.0, 1.0)
        overspeed_range = max(
            float(self.cfg.command_overspeed_end - self.cfg.command_full_speed), 1.0e-6
        )
        overspeed_input = (cmd_x - float(self.cfg.command_full_speed)) / overspeed_range
        overspeed = self._smoothstep01(overspeed_input)
        stride = walk_gate * float(self.cfg.stride_length) * (
            u + (float(self.cfg.max_stride_scale) - 1.0) * overspeed
        )
        frequency = walk_gate * float(self.cfg.step_hz) * (
            0.35
            + 0.65 * torch.sqrt(u)
            + (float(self.cfg.max_frequency_scale) - 1.0) * overspeed
        )
        swing_height = walk_gate * (0.030 + (float(self.cfg.swing_height) - 0.030) * u)
        self.last_overspeed_scale = overspeed
        return walk_gate, u, stride, frequency, swing_height

    def update(self, commands: torch.Tensor) -> torch.Tensor:
        if self.cfg.small_high_frequency_mode:
            return self._update_small_high_frequency(commands)

        walk_gate, _u, stride, frequency, swing_height = self._command_parameters(commands)
        moving = walk_gate > 1.0e-4
        self.walk_time = torch.where(moving, self.walk_time + self.dt, torch.zeros_like(self.walk_time))
        warmup = torch.clamp(self.walk_time / max(float(self.cfg.warmup_sec), 1.0e-6), 0.0, 1.0)
        self.base_phase = torch.remainder(self.base_phase + frequency * self.dt, 1.0)

        phase_offsets = torch.tensor(
            [SWING_ORDER.index(leg) / 4.0 for leg in LEG_ORDER],
            device=self.device,
            dtype=self.default_joint_pos.dtype,
        )
        leg_phase = torch.remainder(self.base_phase.unsqueeze(1) - phase_offsets.unsqueeze(0), 1.0)
        swing_fraction = min(0.235, max(0.12, 1.0 - min(max(float(self.cfg.duty_factor), 0.70), 0.88)))
        swing_mask = leg_phase < swing_fraction
        active_one_hot = swing_mask.to(self.default_joint_pos.dtype)

        q = self.default_joint_pos.clone()
        front_mask = torch.tensor((True, True, False, False), device=self.device)
        stride_gain = torch.tensor(
            (self.cfg.front_stride_gain, self.cfg.front_stride_gain, self.cfg.rear_stride_gain, self.cfg.rear_stride_gain),
            device=self.device,
            dtype=q.dtype,
        )
        height_gain = torch.tensor(
            (
                self.cfg.front_swing_height_gain,
                self.cfg.front_swing_height_gain,
                self.cfg.rear_swing_height_gain,
                self.cfg.rear_swing_height_gain,
            ),
            device=self.device,
            dtype=q.dtype,
        )
        thigh_scale = torch.tensor(
            (
                self.cfg.front_thigh_delta_scale,
                self.cfg.front_thigh_delta_scale,
                self.cfg.rear_thigh_delta_scale,
                self.cfg.rear_thigh_delta_scale,
            ),
            device=self.device,
            dtype=q.dtype,
        )
        calf_extra = torch.tensor(
            (
                self.cfg.front_calf_lift_extra,
                self.cfg.front_calf_lift_extra,
                self.cfg.rear_calf_lift_extra,
                self.cfg.rear_calf_lift_extra,
            ),
            device=self.device,
            dtype=q.dtype,
        )

        s_swing = torch.clamp(leg_phase / swing_fraction, 0.0, 1.0)
        s_stance = torch.clamp((leg_phase - swing_fraction) / (1.0 - swing_fraction), 0.0, 1.0)
        advance = self._smootherstep01(
            (s_swing - float(self.cfg.advance_start))
            / max(float(self.cfg.advance_end - self.cfg.advance_start), 1.0e-6)
        )
        lift_up = self._smootherstep01(s_swing / max(float(self.cfg.advance_start), 1.0e-6))
        lift_down = self._smootherstep01(
            (1.0 - s_swing) / max(1.0 - float(self.cfg.advance_end), 1.0e-6)
        )
        swing_shape = lift_up * lift_down * swing_mask
        stance_shape = torch.sin(math.pi * s_stance) ** 2 * (~swing_mask)
        stance_progress = self._smootherstep01(s_stance)

        leg_stride = stride.unsqueeze(1) * stride_gain.unsqueeze(0)
        leg_height = swing_height.unsqueeze(1) * height_gain.unsqueeze(0)
        x_center = self.default_foot_x + front_mask.to(q.dtype).unsqueeze(0) * float(self.cfg.front_x_bias)
        z_center = self.default_foot_z + front_mask.to(q.dtype).unsqueeze(0) * float(self.cfg.front_z_extend)
        x_swing = x_center - 0.5 * leg_stride + leg_stride * advance
        x_swing += front_mask.to(q.dtype).unsqueeze(0) * float(self.cfg.front_swing_forward_unfold) * swing_shape
        z_swing = z_center + leg_height * swing_shape
        x_stance = x_center + 0.5 * leg_stride - leg_stride * stance_progress
        z_stance = z_center - float(self.cfg.support_stand_tall_m) * (0.35 + 0.65 * stance_shape)
        x_des = torch.where(swing_mask, x_swing, x_stance)
        z_des = torch.where(swing_mask, z_swing, z_stance)

        hip_outward_signs = torch.tensor(self.cfg.hip_outward_signs, device=q.device, dtype=q.dtype)
        # Keep the real-machine gait semantics here. Any URDF sign difference
        # belongs in FanfanJointSemanticAdapter, not in the gait equations.
        hip_delta = -0.004 * swing_shape * hip_outward_signs
        calf_push = -calf_extra.unsqueeze(0) * swing_shape
        calf_push += 0.006 * stance_shape * torch.any(swing_mask, dim=1, keepdim=True)
        thigh_bias = torch.zeros_like(x_des)

        preload_width = min(0.24, max(0.04, float(self.cfg.preload_fraction)))
        pre_progress = torch.clamp((leg_phase - (1.0 - preload_width)) / preload_width, 0.0, 1.0)
        pre_gate = self._smootherstep01(pre_progress)
        active_progress = s_swing
        hold_until = float(self.cfg.front_load_active_hold_until)
        active_gate = torch.where(
            active_progress <= hold_until,
            torch.ones_like(active_progress),
            self._smootherstep01((1.0 - active_progress) / max(1.0 - hold_until, 1.0e-6)),
        )
        front_load_gate = torch.maximum(pre_gate * front_mask, active_gate * swing_mask * front_mask)
        post_hold = min(0.12, max(0.0, float(self.cfg.front_load_post_touchdown_hold)))
        if post_hold > 1.0e-6:
            post_progress = torch.clamp((leg_phase - swing_fraction) / post_hold, 0.0, 1.0)
            post_gate = self._smootherstep01(1.0 - post_progress)
            in_post = (leg_phase >= swing_fraction) & (leg_phase < swing_fraction + post_hold)
            front_load_gate = torch.maximum(
                front_load_gate, post_gate * in_post * front_mask
            )

        for front_leg in FRONT_LEGS:
            front_idx = LEG_ORDER.index(front_leg)
            gate = front_load_gate[:, front_idx]
            diag_idx = LEG_ORDER.index(DIAGONAL_PARTNER[front_leg])
            same_idx = LEG_ORDER.index(SAME_SIDE_REAR[front_leg])
            diag_stance = (~swing_mask[:, diag_idx]).to(q.dtype)
            same_stance = (~swing_mask[:, same_idx]).to(q.dtype)
            diag_gate = gate * diag_stance
            same_gate = gate * same_stance
            z_des[:, diag_idx] -= float(self.cfg.diag_support_preload_z_m) * diag_gate
            x_des[:, diag_idx] += 0.018 * diag_gate
            calf_push[:, diag_idx] += float(self.cfg.diag_support_calf_push_amp) * diag_gate
            thigh_bias[:, diag_idx] += float(self.cfg.diag_support_thigh_back_amp) * diag_gate
            hip_delta[:, diag_idx] += (
                hip_outward_signs[diag_idx]
                * float(self.cfg.diag_support_hip_amp)
                * diag_gate
            )
            z_des[:, same_idx] += float(self.cfg.same_rear_unload_z_m) * same_gate
            calf_push[:, same_idx] -= float(self.cfg.same_rear_calf_relief_amp) * same_gate
            hip_delta[:, same_idx] -= (
                hip_outward_signs[same_idx]
                * float(self.cfg.same_rear_unload_hip_amp)
                * same_gate
            )

        # A rear-leg swing shifts support toward its diagonal front leg.
        for rear_leg in REAR_LEGS:
            rear_idx = LEG_ORDER.index(rear_leg)
            gate = swing_mask[:, rear_idx].to(q.dtype)
            diag_idx = LEG_ORDER.index(DIAGONAL_PARTNER[rear_leg])
            other_idx = 1 - diag_idx
            diag_gate = gate * (~swing_mask[:, diag_idx]).to(q.dtype)
            other_gate = gate * (~swing_mask[:, other_idx]).to(q.dtype)
            rear_shape = 0.65 + 0.35 * stance_shape
            z_des[:, diag_idx] -= 0.006 * rear_shape[:, diag_idx] * diag_gate
            calf_push[:, diag_idx] += 0.018 * stance_shape[:, diag_idx] * diag_gate
            z_des[:, other_idx] -= 0.003 * rear_shape[:, other_idx] * other_gate
            calf_push[:, other_idx] += 0.006 * stance_shape[:, other_idx] * other_gate

        body_shift = torch.zeros(self.num_envs, 2, device=q.device, dtype=q.dtype)
        fl_gate = front_load_gate[:, LEG_ORDER.index("FL")]
        fr_gate = front_load_gate[:, LEG_ORDER.index("FR")]
        body_shift[:, 0] += float(self.cfg.front_swing_body_x_shift_m) * torch.maximum(fl_gate, fr_gate)
        body_shift[:, 1] += float(self.cfg.front_swing_body_y_shift_m) * (fr_gate - fl_gate)
        rr_gate = swing_mask[:, LEG_ORDER.index("RR")].to(q.dtype)
        rl_gate = swing_mask[:, LEG_ORDER.index("RL")].to(q.dtype)
        no_front_load = torch.maximum(fl_gate, fr_gate) < 1.0e-6
        body_shift[:, 0] += (
            float(self.cfg.rear_swing_body_x_shift_m) * (rr_gate + rl_gate) * no_front_load
        )
        body_shift[:, 1] += (
            float(self.cfg.rear_swing_body_y_shift_m) * (rr_gate - rl_gate) * no_front_load
        )

        thigh_ik, _calf_ik = self._inverse_sagittal(x_des, z_des)
        default_thigh = self.default_joint_pos[:, 1::3]
        default_calf = self.default_joint_pos[:, 2::3]
        thigh_target = default_thigh + thigh_scale.unsqueeze(0) * (thigh_ik - default_thigh) + thigh_bias
        calf_target = self._solve_calf_for_z(thigh_target, z_des, default_calf) + calf_push
        calf_target[:, :2] = torch.clamp(calf_target[:, :2], min=float(self.cfg.front_calf_min_rad))

        q[:, 0::3] = self.default_joint_pos[:, 0::3] + warmup.unsqueeze(1) * hip_delta
        q[:, 1::3] = default_thigh + warmup.unsqueeze(1) * (thigh_target - default_thigh)
        q[:, 2::3] = default_calf + warmup.unsqueeze(1) * (calf_target - default_calf)
        q = torch.where(moving.unsqueeze(1), q, self.default_joint_pos)
        if self.joint_limits is not None:
            lower, upper = self.joint_limits
            q = torch.clamp(q, lower.to(q.device), upper.to(q.device))

        self.last_q_ref = q
        self.last_walk_gate = walk_gate
        self.last_frequency = frequency
        self.last_stride = stride
        self.last_swing_height = swing_height
        self.last_leg_phase = leg_phase
        self.last_swing_mask = swing_mask
        self.last_active_swing_one_hot = active_one_hot
        self.last_warmup = warmup
        self.last_body_shift = body_shift
        self.last_predicted_foot_z = self._forward_sagittal(q[:, 1::3], q[:, 2::3])[1]
        self.last_predicted_foot_lift = self.last_predicted_foot_z - self.default_foot_z
        return q

    def _update_small_high_frequency(self, commands: torch.Tensor) -> torch.Tensor:
        walk_gate, _u, stride, frequency, swing_height = self._command_parameters(commands)
        moving = walk_gate > 1.0e-4
        self.walk_time = torch.where(moving, self.walk_time + self.dt, torch.zeros_like(self.walk_time))
        warmup = torch.clamp(self.walk_time / max(float(self.cfg.warmup_sec), 1.0e-6), 0.0, 1.0)
        self.base_phase = torch.remainder(self.base_phase + frequency * self.dt, 1.0)

        swing_start = torch.tensor(
            [SMALL_HIGH_FREQUENCY_SWING_START[leg] for leg in LEG_ORDER],
            device=self.device,
            dtype=self.default_joint_pos.dtype,
        )
        leg_phase = torch.remainder(self.base_phase.unsqueeze(1) - swing_start.unsqueeze(0), 1.0)
        swing_fraction = min(0.249, max(0.12, 1.0 - float(self.cfg.duty_factor)))
        swing_mask = leg_phase < swing_fraction
        active_one_hot = swing_mask.to(self.default_joint_pos.dtype)
        swing_progress = torch.clamp(leg_phase / swing_fraction, 0.0, 1.0)
        stance_progress = torch.clamp(
            (leg_phase - swing_fraction) / (1.0 - swing_fraction), 0.0, 1.0
        )
        swing_advance = self._smootherstep01(swing_progress)
        stance_return = self._smootherstep01(stance_progress)
        swing_shape = torch.sin(math.pi * swing_progress) ** 2 * swing_mask
        rear_rise_end = min(0.45, max(0.20, float(self.cfg.rear_lift_rise_fraction)))
        rear_fall_start = min(
            0.80, max(rear_rise_end + 0.10, float(self.cfg.rear_lift_fall_start))
        )
        rear_lift_up = self._smootherstep01(swing_progress / rear_rise_end)
        rear_lift_down = 1.0 - self._smootherstep01(
            (swing_progress - rear_fall_start) / (1.0 - rear_fall_start)
        )
        rear_swing_shape = (
            torch.minimum(rear_lift_up, rear_lift_down) * swing_mask
        )
        swing_shape = swing_shape.clone()
        swing_shape[:, 2:4] = rear_swing_shape[:, 2:4]

        preload_fraction = min(0.24, max(1.0e-6, float(self.cfg.preload_fraction)))
        preload_progress = torch.clamp(
            (leg_phase - (1.0 - preload_fraction)) / preload_fraction, 0.0, 1.0
        )
        pre_swing_gate = self._smootherstep01(preload_progress)
        post_touchdown_hold = min(
            0.12, max(1.0e-6, float(self.cfg.post_touchdown_hold))
        )
        post_progress = torch.clamp(
            (leg_phase - swing_fraction) / post_touchdown_hold, 0.0, 1.0
        )
        landed_leg_gate = (
            1.0 - self._smootherstep01(post_progress)
        ) * ((leg_phase >= swing_fraction) & (leg_phase < swing_fraction + post_touchdown_hold))
        stance_gate = (~swing_mask).to(self.default_joint_pos.dtype)
        support_gate = stance_gate
        preload_gate = torch.zeros_like(stance_gate)
        swing_support_gate = torch.zeros_like(stance_gate)
        for leg_index in range(4):
            other_legs = [index for index in range(4) if index != leg_index]
            preload_gate[:, other_legs] = torch.maximum(
                preload_gate[:, other_legs],
                pre_swing_gate[:, leg_index].unsqueeze(1) * stance_gate[:, other_legs],
            )
            swing_support_gate[:, other_legs] = torch.maximum(
                swing_support_gate[:, other_legs],
                active_one_hot[:, leg_index].unsqueeze(1) * stance_gate[:, other_legs],
            )
        post_touchdown_gate = landed_leg_gate * stance_gate
        support_load_gate = torch.maximum(
            preload_gate,
            torch.maximum(swing_support_gate, post_touchdown_gate),
        )

        stride_gain = torch.tensor(
            (
                self.cfg.front_stride_gain,
                self.cfg.front_stride_gain,
                self.cfg.rear_stride_gain,
                self.cfg.rear_stride_gain,
            ),
            device=self.device,
            dtype=self.default_joint_pos.dtype,
        )
        height_gain = torch.tensor(
            (
                self.cfg.front_swing_height_gain,
                self.cfg.front_swing_height_gain,
                self.cfg.rear_swing_height_gain,
                self.cfg.rear_swing_height_gain,
            ),
            device=self.device,
            dtype=self.default_joint_pos.dtype,
        )
        leg_stride = stride.unsqueeze(1) * stride_gain.unsqueeze(0)
        leg_height = swing_height.unsqueeze(1) * height_gain.unsqueeze(0)
        x_swing = self.default_foot_x - 0.5 * leg_stride + leg_stride * swing_advance
        x_stance = self.default_foot_x + 0.5 * leg_stride - leg_stride * stance_return
        x_des = torch.where(swing_mask, x_swing, x_stance)
        z_des = self.default_foot_z + leg_height * swing_shape
        support_joint_preload = support_load_gate.clone()
        z_des -= 0.0012 * support_load_gate

        # The candidate/swing leg is excluded from support loading. Its
        # diagonal stance partner gets a small extra preload.
        for active_leg in LEG_ORDER:
            active_idx = LEG_ORDER.index(active_leg)
            event_gate = torch.maximum(
                active_one_hot[:, active_idx],
                pre_swing_gate[:, active_idx],
            )
            diagonal_idx = LEG_ORDER.index(DIAGONAL_PARTNER[active_leg])
            diagonal_gate = event_gate * stance_gate[:, diagonal_idx]
            if active_leg in FRONT_LEGS:
                z_des[:, diagonal_idx] -= 0.0018 * diagonal_gate
                support_joint_preload[:, diagonal_idx] += diagonal_gate
            else:
                z_des[:, diagonal_idx] -= 0.0012 * diagonal_gate
                support_joint_preload[:, diagonal_idx] += 0.75 * diagonal_gate

        thigh_target, calf_target = self._inverse_sagittal(x_des, z_des)
        # Near full leg extension the Cartesian z request can saturate at the
        # IK workspace boundary. A small calf extension target preserves the
        # intended preload as position-error torque instead of unloading the
        # support leg.
        calf_target += 0.006 * torch.clamp(support_joint_preload, max=2.0)
        q_target = self.default_joint_pos.clone()
        hip_signs = torch.tensor(self.cfg.hip_outward_signs, device=self.device, dtype=q_target.dtype)
        q_target[:, 0::3] += -0.003 * swing_shape * hip_signs.unsqueeze(0)
        q_target[:, 1::3] = thigh_target
        q_target[:, 2::3] = calf_target

        q = self.default_joint_pos + warmup.unsqueeze(1) * (q_target - self.default_joint_pos)
        q = torch.where(moving.unsqueeze(1), q, self.default_joint_pos)
        if float(self.cfg.reference_rate_limit_rad_s) > 0.0:
            max_step = float(self.cfg.reference_rate_limit_rad_s) * self.dt
            q = self.last_q_ref + torch.clamp(q - self.last_q_ref, min=-max_step, max=max_step)
        if self.joint_limits is not None:
            lower, upper = self.joint_limits
            q = torch.clamp(q, lower.to(q.device), upper.to(q.device))

        body_shift = torch.zeros(self.num_envs, 2, device=self.device, dtype=q.dtype)
        body_shift[:, 0] = 0.006 * (
            swing_shape[:, LEG_ORDER.index("FR")] + swing_shape[:, LEG_ORDER.index("FL")]
        )
        body_shift[:, 1] = 0.006 * (
            swing_shape[:, LEG_ORDER.index("FR")]
            - swing_shape[:, LEG_ORDER.index("FL")]
            + swing_shape[:, LEG_ORDER.index("RR")]
            - swing_shape[:, LEG_ORDER.index("RL")]
        )

        self.last_q_ref = q
        self.last_walk_gate = walk_gate
        self.last_frequency = frequency
        self.last_stride = stride
        self.last_swing_height = swing_height
        self.last_leg_phase = leg_phase
        self.last_swing_mask = swing_mask
        self.last_active_swing_one_hot = active_one_hot
        self.last_preload_gate = preload_gate
        self.last_post_touchdown_gate = post_touchdown_gate
        self.last_support_gate = support_gate
        self.last_warmup = warmup
        self.last_body_shift = body_shift
        self.last_predicted_foot_z = self._forward_sagittal(q[:, 1::3], q[:, 2::3])[1]
        self.last_predicted_foot_lift = self.last_predicted_foot_z - self.default_foot_z
        return q

    def get_q_ref(self) -> torch.Tensor:
        return self.last_q_ref

    def get_phase_features(self) -> torch.Tensor:
        angle = 2.0 * math.pi * self.last_leg_phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(self.num_envs, 8)

    def get_debug_info(self) -> dict[str, torch.Tensor]:
        return {
            "walk_gate": self.last_walk_gate,
            "frequency": self.last_frequency,
            "stride": self.last_stride,
            "swing_height": self.last_swing_height,
            "duty_factor": torch.full(
                (self.num_envs,), float(self.cfg.duty_factor), device=self.device
            ),
            "overspeed_scale": self.last_overspeed_scale,
            "warmup": self.last_warmup,
            "leg_phase": self.last_leg_phase,
            "swing_mask": self.last_swing_mask,
            "active_swing_one_hot": self.last_active_swing_one_hot,
            "preload_gate": self.last_preload_gate,
            "post_touchdown_gate": self.last_post_touchdown_gate,
            "support_gate": self.last_support_gate,
            "body_shift": self.last_body_shift,
            "policy_q_ref": self.last_q_ref,
            "predicted_foot_z": self.last_predicted_foot_z,
            "predicted_foot_lift": self.last_predicted_foot_lift,
        }
