from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Protocol

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.deploy_actions import (
    DeployFilteredJointPositionAction,
    DeployFilteredJointPositionActionCfg,
)

from .joint_semantics import FanfanJointSemanticAdapter, FanfanJointSemanticCfg
from .curriculum_profiles import get_wave_stage_number
from .reference_gait import FanfanReferenceGait, FanfanReferenceGaitCfg
from .residual_math import (
    clamp_joint_targets,
    filter_residual,
    filter_vmc_delta,
    joint_mapping_index,
    validate_reference_control_stage,
)


class FullVmcProvider(Protocol):
    def compute(
        self,
        action_term: "WaveResidualJointPositionAction",
        q_cpg_policy: torch.Tensor,
    ) -> torch.Tensor: ...


class WaveResidualJointPositionAction(DeployFilteredJointPositionAction):
    """Wave-gait reference plus bounded residual, followed by deployment-like filtering."""

    cfg: WaveResidualJointPositionActionCfg

    def __init__(self, cfg: "WaveResidualJointPositionActionCfg", env):
        super().__init__(cfg, env)
        FanfanJointSemanticAdapter.assert_sim_joint_names(self._joint_names)
        self.semantic_adapter = FanfanJointSemanticAdapter(
            cfg.semantic_cfg,
            device=self.device,
            dtype=self.processed_actions.dtype,
        )
        max_delay = max(0, int(cfg.sim_motor_delay_steps_range[1]))
        self._delay_buffer = torch.zeros(
            self.num_envs, max_delay + 1, self.action_dim, device=self.device
        )
        default_q_sim = self._asset.data.default_joint_pos[:, self._joint_ids]
        limits_sim = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._hard_joint_lower = limits_sim[:, :, 0].clone()
        self._hard_joint_upper = limits_sim[:, :, 1].clone()
        default_q_policy = self.semantic_adapter.sim_to_policy(default_q_sim)
        limits_policy = self.semantic_adapter.sim_limits_to_policy(
            limits_sim[:, :, 0], limits_sim[:, :, 1]
        )
        self.reference = FanfanReferenceGait(
            cfg=cfg.reference_cfg,
            num_envs=self.num_envs,
            device=self.device,
            dt=float(self._env.step_dt),
            default_joint_pos=default_q_policy,
            joint_limits=None,
        )
        self._policy_joint_limits = limits_policy
        self._foot_body_ids, resolved_foot_names = self._asset.find_bodies(
            ["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True
        )
        if tuple(resolved_foot_names) != ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            raise ValueError(f"Unexpected Fanfan foot body order: {resolved_foot_names}")
        self._trunk_body_ids, resolved_trunk_names = self._asset.find_bodies(
            ["Trunk"], preserve_order=True
        )
        if tuple(resolved_trunk_names) != ("Trunk",):
            raise ValueError(f"Unexpected Fanfan trunk body: {resolved_trunk_names}")
        self._rear_lift_step = 0
        self.last_rear_lift_phase = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_support_preload_delta_z = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_target_leg_unload_delta_z = torch.zeros(
            self.num_envs, device=self.device
        )
        self._rear_lift_state_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._rear_lift_force_drop_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_force_drop_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.last_failure_reason = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_force_below_threshold = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.last_force_below_timer = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_first_force_drop_time = torch.full(
            (self.num_envs,), -1.0, device=self.device
        )
        self.last_lift_entry_time = torch.full(
            (self.num_envs,), -1.0, device=self.device
        )
        self.last_missed_force_drop_window = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.last_state_transition_reason = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_active_swing_pair = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_expected_support_pair = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_phase_switch_guard_strength = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_phase_to_switch = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_guard_kp_scale = torch.zeros_like(self.processed_actions)
        self.last_light_vmc_weight = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_light_vmc_foot_z_offset = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_light_vmc_foot_x_offset = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_light_vmc_foot_y_offset = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_light_vmc_height_corr_z = torch.zeros(self.num_envs, device=self.device)
        self.last_light_vmc_roll_corr_z = torch.zeros(self.num_envs, device=self.device)
        self.last_light_vmc_pitch_corr_z = torch.zeros(self.num_envs, device=self.device)
        self.last_light_vmc_foot_x_corr = torch.zeros(self.num_envs, device=self.device)
        self.last_light_vmc_foot_y_corr = torch.zeros(self.num_envs, device=self.device)
        self._light_vmc_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._light_vmc_target_yaw_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.last_light_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self.last_light_yaw_corr_hip_raw = torch.zeros(self.num_envs, device=self.device)
        self.last_light_yaw_corr_hip = torch.zeros(self.num_envs, device=self.device)
        self.last_light_yaw_hip_offset = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_light_yaw_hip_rate_limited = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_preswing_unload_gate = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_preswing_vmc_fade = torch.ones(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_preswing_unload_z_offset = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_touchdown_vmc_ramp_weight = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_touchdown_kp_scale = torch.ones(
            self.num_envs, 4, device=self.device
        )
        self.last_phase_switch_vmc_weight_scale_applied = torch.ones(
            self.num_envs, device=self.device
        )
        self.last_phase_switch_yaw_weight_scale_applied = torch.ones(
            self.num_envs, device=self.device
        )
        self.last_phase_switch_kp_scale_applied = torch.ones(
            self.num_envs, device=self.device
        )
        self.last_rear_late_swing_guard_active = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.bool
        )
        self.last_rear_late_swing_clearance_offset = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_late_swing_height = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_late_swing_height_error = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_late_swing_descent_scale_applied = torch.ones(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_early_contact_guard_active = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.bool
        )
        self.last_rear_early_contact_relief_offset = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_early_contact_kp_scale = torch.ones(
            self.num_envs, 4, device=self.device
        )
        self.last_rear_touchdown_kp_ramp_weight = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.last_debug_kp = torch.full_like(
            self.processed_actions, max(float(cfg.sim_kp), 1.0e-6)
        )
        self.last_debug_kd = torch.full_like(
            self.processed_actions, max(float(cfg.sim_kd), 0.0)
        )
        self.last_body_shift_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_diagnostic_leg = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.last_diagnostic_delta_z = torch.zeros(self.num_envs, device=self.device)
        self.last_diagnostic_force_before = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_diagnostic_force_after = torch.zeros(
            self.num_envs, device=self.device
        )
        contact_sensor = self._env.scene.sensors["contact_forces"]
        self._contact_foot_ids, contact_foot_names = contact_sensor.find_bodies(
            ["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True
        )
        if tuple(contact_foot_names) != ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            raise ValueError(f"Unexpected contact-sensor foot order: {contact_foot_names}")
        self._validate_control_stage()
        self._residual_scale = self._make_residual_scale()
        self._filtered_residual = torch.zeros_like(self.processed_actions)
        self._filtered_vmc_delta = torch.zeros_like(self.processed_actions)
        self.last_q_ref_policy = self.reference.default_joint_pos.clone()
        self.last_q_ref = self.semantic_adapter.policy_to_sim(self.last_q_ref_policy)
        self.last_q_cpg_policy = self.last_q_ref_policy.clone()
        self.last_q_cpg = self.last_q_ref.clone()
        self.last_q_vmc_delta = torch.zeros_like(self.processed_actions)
        self._previous_raw_target = self.last_q_ref.clone()
        self.last_raw_target_rate = torch.zeros_like(self.processed_actions)
        self.last_delta_q_rl = torch.zeros_like(self.processed_actions)
        self.last_q_raw_policy = self.last_q_ref_policy.clone()
        self.last_q_raw_reference = self.last_q_ref.clone()
        self.last_q_after_joint_limit = self.last_q_ref.clone()
        self.last_q_after_rate_limit = self.last_q_ref.clone()
        self.last_q_after_accel_limit = self.last_q_ref.clone()
        self.last_q_after_torque_clip = self.last_q_ref.clone()
        self.last_q_before_delay = self.last_q_ref.clone()
        self.last_q_after_delay = self.last_q_ref.clone()
        self.last_tau_est = torch.zeros_like(self.processed_actions)
        self.last_tau_est_raw_ref = torch.zeros_like(self.processed_actions)
        self.last_tau_est_after_rate = torch.zeros_like(self.processed_actions)
        self.last_tau_est_after_accel = torch.zeros_like(self.processed_actions)
        self.last_tau_est_cmd_final = torch.zeros_like(self.processed_actions)
        self.last_q_error_raw_ref = torch.zeros_like(self.processed_actions)
        self.last_rate_demand = torch.zeros_like(self.processed_actions)
        self.last_accel_demand = torch.zeros_like(self.processed_actions)
        self._previous_rate_demand = torch.zeros_like(self.processed_actions)
        self.last_rate_limit_delta = torch.zeros_like(self.processed_actions)
        self.last_accel_limit_delta = torch.zeros_like(self.processed_actions)
        self.last_torque_clip_delta = torch.zeros_like(self.processed_actions)
        self.last_joint_limit_margin = torch.zeros_like(self.processed_actions)
        self.last_joint_limit_warning = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_kp_actual = self.last_debug_kp.clone()
        self.last_kd_actual = self.last_debug_kd.clone()
        self.last_torque_budget_per_joint = torch.zeros_like(self.processed_actions)
        self.last_err_limit_per_joint = torch.zeros_like(self.processed_actions)
        self.debug_kp_override: torch.Tensor | None = None
        self.debug_kd_override: torch.Tensor | None = None
        self.last_joint_limit_clip_mask = torch.zeros_like(self.processed_actions, dtype=torch.bool)
        self.last_rate_clip_mask = torch.zeros_like(self.processed_actions, dtype=torch.bool)
        self.last_accel_clip_mask = torch.zeros_like(self.processed_actions, dtype=torch.bool)
        self.last_torque_clip_mask = torch.zeros_like(self.processed_actions, dtype=torch.bool)
        self.last_joint_limit_clipping_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_rate_clipping_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_accel_clipping_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_filter_error = torch.zeros(self.num_envs, device=self.device)
        self.last_filter_clipping_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_torque_clipping_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_over_8nm_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_over_12nm_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_over_17nm_ratio = torch.zeros(self.num_envs, device=self.device)
        self._joint_mapping_step = 0
        self._joint_mapping_index = -1
        self._last_joint_limit_warning_step = -10**9
        self._csv_playback_time = torch.zeros(self.num_envs, device=self.device)
        self._csv_playback = None
        if cfg.action_mode == "csv_playback":
            try:
                from .csv_playback import LoopingJointCsvPlayback, load_joint_csv
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "CsvPlayback-v0 requires fanfan_rl_cpg_residual/csv_playback.py. "
                    "The file is not needed by Reference or SmallHighFreq Stage 0/1/2 tasks."
                ) from exc
            csv_path = os.environ.get("FANFAN_CSV_PLAYBACK_PATH", cfg.csv_playback_path)
            times, values, value_space = load_joint_csv(csv_path)
            if value_space == "real":
                values = self.semantic_adapter.real_to_policy(values.to(self.device)).cpu()
            self._csv_playback = LoopingJointCsvPlayback(times, values, device=self.device)
            print(
                f"[FANFAN CSV PLAYBACK] loaded {values.shape[0]} frames from {csv_path}, "
                f"duration={self._csv_playback.duration:.3f}s, source={value_space}"
            )

    def _validate_control_stage(self) -> None:
        stage = int(self.cfg.control_stage)
        mode = str(self.cfg.vmc_mode)
        validate_reference_control_stage(stage, bool(self.cfg.enable_vmc), mode)
        if stage == 3 and self.cfg.full_vmc_provider is None:
            raise NotImplementedError(
                "Full VMC has no calibrated provider. Use Stage 0/1/2 until a full VMC provider is injected."
            )

    def _make_residual_scale(self) -> torch.Tensor:
        values = []
        for name in self._joint_names:
            if "_hip_joint" in name:
                values.append(float(self.cfg.residual_scale_hip))
            elif "_thigh_joint" in name:
                values.append(float(self.cfg.residual_scale_thigh))
            elif "_calf_joint" in name:
                values.append(float(self.cfg.residual_scale_calf))
            else:
                values.append(float(self.cfg.residual_scale_default))
        return torch.tensor(values, device=self.device, dtype=self.processed_actions.dtype).unsqueeze(0)

    def _commands(self) -> torch.Tensor:
        return self._env.command_manager.get_command(self.cfg.command_name)

    def _foot_normal_forces(self) -> torch.Tensor:
        contact_sensor = self._env.scene.sensors["contact_forces"]
        return torch.norm(
            contact_sensor.data.net_forces_w[:, self._contact_foot_ids, :], dim=-1
        )

    def _foot_target_to_policy(
        self,
        q_policy: torch.Tensor,
        *,
        foot_x_delta: torch.Tensor | None = None,
        foot_z_delta: torch.Tensor | None = None,
        body_shift_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply body-frame foot deltas to a policy-order stand target."""
        thigh = q_policy[:, 1::3]
        calf = q_policy[:, 2::3]
        x_default, z_default = self.reference._forward_sagittal(thigh, calf)
        x_target = x_default if foot_x_delta is None else x_default + foot_x_delta
        z_target = z_default if foot_z_delta is None else z_default + foot_z_delta
        thigh_target, calf_target = self.reference._inverse_sagittal(x_target, z_target)
        q_policy[:, 1::3] = thigh_target
        q_policy[:, 2::3] = calf_target
        if body_shift_y is not None:
            leg_length = torch.clamp(torch.abs(z_default), min=0.15)
            q_policy[:, 0::3] += -body_shift_y.unsqueeze(1) / leg_length
        return q_policy

    @staticmethod
    def _roll_pitch_from_quat(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sin_roll, cos_roll)
        sin_pitch = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = torch.asin(sin_pitch)
        return roll, pitch

    def _compute_vmc_delta(self, q_cpg_policy: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_vmc or self.cfg.vmc_mode == "off":
            self._filtered_vmc_delta.zero_()
            return self._filtered_vmc_delta
        if self.cfg.vmc_mode == "full":
            provider = self.cfg.full_vmc_provider
            if provider is None:
                raise NotImplementedError("Full VMC provider is not configured.")
            delta = provider.compute(self, q_cpg_policy)
            if delta.shape != q_cpg_policy.shape:
                raise ValueError(
                    f"Full VMC returned {tuple(delta.shape)}, expected {tuple(q_cpg_policy.shape)}."
                )
            return delta
        if self.cfg.vmc_mode != "light":
            raise ValueError(f"Unsupported VMC mode: {self.cfg.vmc_mode!r}.")

        roll, pitch = self._roll_pitch_from_quat(self._asset.data.root_quat_w)
        ang_vel = self._asset.data.root_ang_vel_b
        root_height = self._asset.data.root_pos_w[:, 2]
        roll_cmd = float(self.cfg.vmc_roll_kp_m_per_rad) * roll
        roll_cmd += float(self.cfg.vmc_roll_kd_m_per_rad_s) * ang_vel[:, 0]
        pitch_cmd = float(self.cfg.vmc_pitch_kp_m_per_rad) * pitch
        pitch_cmd += float(self.cfg.vmc_pitch_kd_m_per_rad_s) * ang_vel[:, 1]
        height_cmd = -float(self.cfg.vmc_height_kp) * (
            float(self.cfg.vmc_body_height_target_m) - root_height
        )

        side = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=self.device)
        fore_aft = torch.tensor((1.0, 1.0, -1.0, -1.0), device=self.device)
        dz = height_cmd.unsqueeze(1)
        dz = dz - side.unsqueeze(0) * roll_cmd.unsqueeze(1)
        dz = dz - fore_aft.unsqueeze(0) * pitch_cmd.unsqueeze(1)
        dz = torch.clamp(
            dz,
            min=-float(self.cfg.vmc_foot_z_limit_m),
            max=float(self.cfg.vmc_foot_z_limit_m),
        )

        swing_fraction = min(
            0.235,
            max(0.12, 1.0 - min(max(float(self.reference.cfg.duty_factor), 0.70), 0.88)),
        )
        leg_phase = self.reference.last_leg_phase
        blend_width = max(float(self.cfg.vmc_stance_blend_fraction), 1.0e-4)
        stance_in = self.reference._smootherstep01((leg_phase - swing_fraction) / blend_width)
        stance_out = self.reference._smootherstep01((1.0 - leg_phase) / blend_width)
        stance_blend = stance_in * stance_out * (~self.reference.last_swing_mask)
        dz *= stance_blend

        thigh = q_cpg_policy[:, 1::3]
        calf = q_cpg_policy[:, 2::3]
        foot_x, foot_z = self.reference._forward_sagittal(thigh, calf)
        thigh_target, calf_target = self.reference._inverse_sagittal(foot_x, foot_z + dz)
        raw_delta = torch.zeros_like(q_cpg_policy)
        raw_delta[:, 1::3] = thigh_target - thigh
        raw_delta[:, 2::3] = calf_target - calf
        filtered = filter_vmc_delta(
            raw_delta,
            self._filtered_vmc_delta,
            joint_limit_rad=float(self.cfg.vmc_joint_delta_limit_rad),
            rate_limit_rad_s=float(self.cfg.vmc_joint_rate_limit_rad_s),
            lowpass_alpha=float(self.cfg.vmc_lowpass_alpha),
            dt=float(self._env.step_dt),
        )
        joint_stance_blend = torch.repeat_interleave(stance_blend, repeats=3, dim=1)
        filtered *= joint_stance_blend
        self._filtered_vmc_delta.copy_(filtered)
        return self._filtered_vmc_delta

    def _actual_pd_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.debug_kp_override is None:
            kp_eff = max(float(self.cfg.sim_kp), 1.0e-6) * self._kp_scale * self._motor_strength
        else:
            kp_eff = self.debug_kp_override
        if self.debug_kd_override is None:
            kd_eff = max(float(self.cfg.sim_kd), 0.0) * self._kd_scale * self._motor_strength
        else:
            kd_eff = self.debug_kd_override
        if kp_eff.shape[1] == 1:
            kp_eff = kp_eff.expand(-1, self.action_dim)
        if kd_eff.shape[1] == 1:
            kd_eff = kd_eff.expand(-1, self.action_dim)
        return kp_eff, kd_eff

    def _per_joint_torque_budget(self) -> torch.Tensor:
        budget = getattr(
            self,
            "_torque_budget",
            torch.full(
                (self.num_envs, 1),
                float(self.cfg.sim_torque_budget_range[0]),
                device=self.device,
                dtype=self.processed_actions.dtype,
            ),
        )
        if budget.shape[1] == 1:
            budget = budget.expand(-1, self.action_dim)
        return budget

    def _update_safety_debug(
        self,
        *,
        kp_eff: torch.Tensor | None = None,
        kd_eff: torch.Tensor | None = None,
        torque_budget: torch.Tensor | None = None,
        err_limit: torch.Tensor | None = None,
    ) -> None:
        if kp_eff is None or kd_eff is None:
            kp_eff, kd_eff = self._actual_pd_gains()
        if torque_budget is None:
            torque_budget = self._per_joint_torque_budget()
        if err_limit is None:
            err_limit = torque_budget / torch.clamp(kp_eff, min=1.0e-6)
        self.last_kp_actual[:] = kp_eff
        self.last_kd_actual[:] = kd_eff
        self.last_torque_budget_per_joint[:] = torque_budget
        self.last_err_limit_per_joint[:] = err_limit

    def _update_torque_threshold_debug(self) -> None:
        abs_tau = torch.abs(self.last_tau_est_cmd_final)
        self.last_over_8nm_ratio[:] = torch.mean((abs_tau > 8.0).to(abs_tau.dtype), dim=1)
        self.last_over_12nm_ratio[:] = torch.mean((abs_tau > 12.0).to(abs_tau.dtype), dim=1)
        self.last_over_17nm_ratio[:] = torch.mean(
            (abs_tau > float(self.cfg.sim_hard_torque_budget)).to(abs_tau.dtype), dim=1
        )

    def _pd_torque_for(
        self,
        q_target: torch.Tensor,
        kp_eff: torch.Tensor,
        kd_eff: torch.Tensor,
    ) -> torch.Tensor:
        q_current = self._asset.data.joint_pos[:, self._joint_ids]
        qd_current = self._asset.data.joint_vel[:, self._joint_ids]
        return kp_eff * (q_target - q_current) - kd_eff * qd_current

    @staticmethod
    def _ratio_over(tau: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.mean((torch.abs(tau) > threshold).to(tau.dtype), dim=1)

    def _record_raw_risk_debug(self, q_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kp_eff, kd_eff = self._actual_pd_gains()
        torque_budget = self._per_joint_torque_budget()
        hard_budget = torch.full_like(torque_budget, float(self.cfg.sim_hard_torque_budget))
        err_limit = hard_budget / torch.clamp(kp_eff, min=1.0e-6)
        self._update_safety_debug(
            kp_eff=kp_eff,
            kd_eff=kd_eff,
            torque_budget=torque_budget,
            err_limit=err_limit,
        )
        q_current = self._asset.data.joint_pos[:, self._joint_ids]
        self.last_q_error_raw_ref[:] = q_raw - q_current
        self.last_tau_est_raw_ref[:] = self._pd_torque_for(q_raw, kp_eff, kd_eff)
        self.last_tau_est_after_rate[:] = self.last_tau_est_raw_ref
        self.last_tau_est_after_accel[:] = self.last_tau_est_raw_ref
        self.last_tau_est_cmd_final[:] = self.last_tau_est_raw_ref
        self.last_tau_est[:] = self.last_tau_est_cmd_final
        self._update_torque_threshold_debug()
        return kp_eff, kd_eff

    def _performance_safe_torque_target(
        self,
        q_target: torch.Tensor,
        q_current: torch.Tensor,
        kp_eff: torch.Tensor,
    ) -> torch.Tensor:
        soft = float(self.cfg.fast_trot_soft_peak_torque_budget)
        hard = float(self.cfg.sim_hard_torque_budget)
        err = q_target - q_current
        abs_err = torch.abs(err)
        sign = torch.sign(err)
        soft_err = soft / torch.clamp(kp_eff, min=1.0e-6)
        hard_err = hard / torch.clamp(kp_eff, min=1.0e-6)
        denom = torch.clamp(hard_err - soft_err, min=1.0e-6)
        t = torch.clamp((abs_err - soft_err) / denom, min=0.0, max=1.0)
        compressed = soft_err + denom * (0.5 * t + 0.5 * t * t)
        limited_abs = torch.where(
            abs_err <= soft_err,
            abs_err,
            torch.where(abs_err < hard_err, compressed, hard_err),
        )
        return q_current + sign * limited_abs

    def _performance_soft_output_torque_target(
        self,
        q_target: torch.Tensor,
        q_current: torch.Tensor,
        q_ref: torch.Tensor,
        kp_eff: torch.Tensor,
        kd_eff: torch.Tensor,
        guard_strength: torch.Tensor | None = None,
        early_contact_guard_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Softly back off high-torque targets without turning the path into real_safe."""
        tau = self._pd_torque_for(q_target, kp_eff, kd_eff)
        abs_tau = torch.abs(tau)
        soft_start = torch.full_like(abs_tau, float(self.cfg.fast_trot_soft_output_start_torque))
        soft_full = torch.full_like(abs_tau, float(self.cfg.fast_trot_soft_output_full_torque))
        if guard_strength is not None:
            guard = guard_strength.unsqueeze(1).expand_as(abs_tau)
            soft_start = soft_start * (1.0 - guard) + float(self.cfg.fast_trot_guard_soft_start_torque) * guard
            soft_full = soft_full * (1.0 - guard) + float(self.cfg.fast_trot_guard_soft_full_torque) * guard
        if early_contact_guard_strength is not None:
            early_guard = early_contact_guard_strength.unsqueeze(1).expand_as(abs_tau)
            soft_start = soft_start * (1.0 - early_guard) + float(self.cfg.rear_early_contact_torque_soft_start) * early_guard
            soft_full = soft_full * (1.0 - early_guard) + float(self.cfg.rear_early_contact_torque_soft_full) * early_guard
        hard = float(self.cfg.sim_hard_torque_budget)
        soft_t = torch.clamp(
            (abs_tau - soft_start) / torch.clamp(soft_full - soft_start, min=1.0e-6),
            0.0,
            1.0,
        )
        soft_t = soft_t * soft_t * (3.0 - 2.0 * soft_t)
        hard_t = torch.clamp(
            (abs_tau - soft_full) / torch.clamp(hard - soft_full, min=1.0e-6),
            0.0,
            1.0,
        )
        hard_t = hard_t * hard_t * (3.0 - 2.0 * hard_t)
        # Keep most of the trajectory below 14 Nm; only compress strongly near the 17 Nm boundary.
        scale = 1.0 - 0.18 * soft_t - 0.32 * hard_t
        hard_scale = torch.where(abs_tau > hard, hard / torch.clamp(abs_tau, min=1.0e-6), torch.ones_like(abs_tau))
        scale = torch.minimum(scale, hard_scale)
        q_backoff = q_current + scale * (q_target - q_current)
        # Keep q_ref in the signature to make the safety contract explicit; CSV summary
        # reports q_ref-q_cmd drift instead of hiding it with a second hard projection.
        _ = q_ref
        return q_backoff

    def _estimate_pd_torque(self, q_target: torch.Tensor) -> None:
        kp_eff, kd_eff = self._actual_pd_gains()
        self._update_safety_debug(kp_eff=kp_eff, kd_eff=kd_eff)
        self.last_tau_est[:] = self._pd_torque_for(q_target, kp_eff, kd_eff)
        self.last_tau_est_cmd_final[:] = self.last_tau_est
        self._update_torque_threshold_debug()

    def _record_raw_target_rate(self, q_target: torch.Tensor) -> None:
        self.last_raw_target_rate[:] = (
            q_target - self._previous_raw_target
        ) / float(self._env.step_dt)
        self.last_rate_demand[:] = self.last_raw_target_rate
        self.last_accel_demand[:] = (
            self.last_rate_demand - self._previous_rate_demand
        ) / float(self._env.step_dt)
        self._previous_rate_demand[:] = self.last_rate_demand
        self._previous_raw_target[:] = q_target

    def _clamp_to_hard_joint_limits(self, q_sim: torch.Tensor) -> torch.Tensor:
        clamped, clip_mask = clamp_joint_targets(
            q_sim, self._hard_joint_lower, self._hard_joint_upper
        )
        self.last_joint_limit_clip_mask[:] = clip_mask
        self.last_joint_limit_clipping_ratio[:] = torch.mean(
            clip_mask.to(q_sim.dtype), dim=1
        )
        if torch.any(clip_mask):
            step = int(getattr(self._env, "common_step_counter", 0))
            interval_steps = max(
                1, round(float(self.cfg.joint_limit_warning_interval_sec) / float(self._env.step_dt))
            )
            if step - self._last_joint_limit_warning_step >= interval_steps:
                self._last_joint_limit_warning_step = step
                for joint_index in torch.nonzero(clip_mask[0], as_tuple=False).flatten().tolist():
                    print(
                        "[FANFAN JOINT LIMIT] "
                        f"joint={self._joint_names[joint_index]} "
                        f"before={float(q_sim[0, joint_index]):.6f} "
                        f"after={float(clamped[0, joint_index]):.6f} "
                        f"lower={float(self._hard_joint_lower[0, joint_index]):.6f} "
                        f"upper={float(self._hard_joint_upper[0, joint_index]):.6f}"
                    )
        return clamped

    def _set_direct_playback_output(self, q_policy: torch.Tensor) -> None:
        q_sim_unclamped = self.semantic_adapter.policy_to_sim(q_policy)
        q_sim = self._clamp_to_hard_joint_limits(q_sim_unclamped)
        self._record_raw_target_rate(q_sim)
        self.last_q_cpg_policy[:] = q_policy
        self.last_q_cpg[:] = q_sim_unclamped
        self.last_q_vmc_delta.zero_()
        self.last_q_ref_policy[:] = q_policy
        self.last_q_ref[:] = q_sim_unclamped
        self.last_delta_q_rl.zero_()
        self.last_q_raw_policy[:] = q_policy
        self.last_q_raw_reference[:] = q_sim
        self._deploy_q_raw[:] = q_sim
        self.last_q_after_joint_limit[:] = q_sim
        self.last_q_after_rate_limit[:] = q_sim
        self.last_q_after_accel_limit[:] = q_sim
        self.last_q_after_torque_clip[:] = q_sim
        self.last_q_before_delay[:] = q_sim
        self.last_q_after_delay[:] = q_sim
        self.last_q_cmd[:] = q_sim
        self.last_qdot_cmd.zero_()
        self.last_tau_est.zero_()
        self.last_rate_clip_mask.zero_()
        self.last_accel_clip_mask.zero_()
        self.last_torque_clip_mask.zero_()
        self.last_rate_clipping_ratio.zero_()
        self.last_accel_clipping_ratio.zero_()
        self.last_filter_error.zero_()
        self.last_filter_clipping_ratio.zero_()
        self.last_torque_clipping_ratio.zero_()
        self.last_over_8nm_ratio.zero_()
        self.last_over_12nm_ratio.zero_()
        self.last_over_17nm_ratio.zero_()
        self._update_safety_debug()
        self._filtered_residual.zero_()
        self._processed_actions[:] = q_sim

    def _sample_filter_params(self, env_ids: torch.Tensor) -> None:
        super()._sample_filter_params(env_ids)
        stage = get_wave_stage_number(int(getattr(self._env, "_fanfan_wave_stage", 1)))
        shape = (env_ids.numel(), 1)
        gain_range = tuple(stage["actuator_gain"])
        motor_range = tuple(stage["motor_strength"])
        self._kp_scale[env_ids] = self._uniform(gain_range, shape)
        self._kd_scale[env_ids] = self._uniform(gain_range, shape)
        self._motor_strength[env_ids] = self._uniform(motor_range, shape)
        if self.cfg.fixed_delay_steps is None:
            d0, d1 = stage["delay_steps"]
        else:
            d0 = d1 = int(self.cfg.fixed_delay_steps)
        shape = (env_ids.numel(), 1)
        self._motor_delay_steps[env_ids] = torch.randint(
            max(0, int(d0)),
            max(0, int(d1)) + 1,
            shape,
            device=self.device,
        )

    def _rear_lift_test_target(self) -> torch.Tensor:
        leg = str(self.cfg.rear_lift_test_leg).upper()
        if leg not in ("RR", "RL"):
            raise ValueError(f"rear_lift_test_leg must be RR or RL, got {leg!r}.")
        leg_index = 2 if leg == "RR" else 3
        q_policy = self.reference.default_joint_pos.clone()
        q_policy[:, leg_index * 3 + 1] = float(self.cfg.rear_lift_test_thigh)
        q_policy[:, leg_index * 3 + 2] = float(self.cfg.rear_lift_test_calf)

        dt = float(self._env.step_dt)
        state = self.last_rear_lift_phase
        state_step = self._rear_lift_state_step
        target_force = self._foot_normal_forces()[:, leg_index]
        elapsed = self._rear_lift_step * dt
        self.last_state_transition_reason.zero_()
        timed_durations = (
            float(self.cfg.rear_lift_test_settle_sec),
            float(self.cfg.rear_lift_pre_shift_sec),
            float(self.cfg.rear_lift_test_preload_sec),
        )
        for phase_id, duration in enumerate(timed_durations):
            advance = (state == phase_id) & (state_step * dt >= max(dt, duration))
            state[advance] += 1
            state_step[advance] = 0
            self.last_state_transition_reason[advance] = phase_id + 1

        force_low = target_force < float(self.cfg.rear_lift_force_drop_threshold_n)
        monitor_force = (state == 3) | (state == 4)
        previous_force_drop_steps = self._rear_lift_force_drop_steps.clone()
        first_drop = (
            monitor_force
            & force_low
            & (self.last_first_force_drop_time < 0.0)
        )
        self.last_first_force_drop_time[first_drop] = elapsed
        missed_window = (
            monitor_force
            & ~force_low
            & (previous_force_drop_steps > 0)
            & ~self.last_force_drop_success
        )
        self.last_missed_force_drop_window |= missed_window
        self._rear_lift_force_drop_steps = torch.where(
            monitor_force & force_low,
            self._rear_lift_force_drop_steps + 1,
            torch.where(
                monitor_force,
                torch.zeros_like(self._rear_lift_force_drop_steps),
                self._rear_lift_force_drop_steps,
            ),
        )
        self.last_force_below_threshold[:] = force_low
        self.last_force_below_timer[:] = (
            self._rear_lift_force_drop_steps.to(q_policy.dtype) * dt
        )
        confirm_steps = max(
            1, round(float(self.cfg.rear_lift_force_confirm_sec) / dt)
        )
        start_lift = monitor_force & (
            self._rear_lift_force_drop_steps >= confirm_steps
        )
        lift_from_unload = start_lift & (state == 3)
        lift_from_wait = start_lift & (state == 4)
        self.last_force_drop_success[start_lift] = True
        self.last_lift_entry_time[start_lift] = elapsed
        self.last_state_transition_reason[lift_from_unload] = 5
        self.last_state_transition_reason[lift_from_wait] = 6
        state[start_lift] = 5
        state_step[start_lift] = 0

        unload_complete = (state == 3) & ~start_lift & (
            state_step * dt >= max(dt, float(self.cfg.rear_lift_unload_sec))
        )
        state[unload_complete] = 4
        state_step[unload_complete] = 0
        self.last_state_transition_reason[unload_complete] = 4

        wait_timeout = (state == 4) & (
            state_step * dt >= float(self.cfg.rear_lift_force_drop_timeout_sec)
        )
        self.last_failure_reason[wait_timeout] = 1
        self.last_state_transition_reason[wait_timeout] = 7
        state[wait_timeout] = 6
        state_step[wait_timeout] = 0

        def smooth_state_progress(duration: float) -> torch.Tensor:
            progress = torch.clamp(
                state_step.to(q_policy.dtype) * dt / max(dt, duration), 0.0, 1.0
            )
            return progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)

        shift_progress = smooth_state_progress(float(self.cfg.rear_lift_pre_shift_sec))
        preload_progress = smooth_state_progress(float(self.cfg.rear_lift_test_preload_sec))
        unload_progress = smooth_state_progress(float(self.cfg.rear_lift_unload_sec))
        shift_gate = torch.where(
            state > 1, torch.ones_like(shift_progress), torch.where(state == 1, shift_progress, 0.0)
        )
        preload_gate = torch.where(
            state > 2,
            torch.ones_like(preload_progress),
            torch.where(state == 2, preload_progress, 0.0),
        )
        unload_gate = torch.where(
            state > 3,
            torch.ones_like(unload_progress),
            torch.where(state == 3, unload_progress, 0.0),
        )
        cycle = max(0.5, float(self.cfg.rear_lift_test_cycle_sec))
        lift_phase = torch.remainder(state_step.to(q_policy.dtype) * dt / cycle, 1.0)
        triangle = torch.where(
            lift_phase < 0.5, 2.0 * lift_phase, 2.0 * (1.0 - lift_phase)
        )
        lift_progress = triangle**3 * (
            triangle * (triangle * 6.0 - 15.0) + 10.0
        )
        lift_progress *= (state == 5).to(q_policy.dtype)

        shift_x = float(self.cfg.rear_lift_body_shift_x_m) * shift_gate
        shift_y = (
            (1.0 if leg == "RR" else -1.0)
            * float(self.cfg.rear_lift_body_shift_y_m)
            * shift_gate
        )
        self.last_body_shift_xy[:, 0] = shift_x
        self.last_body_shift_xy[:, 1] = shift_y
        support_preload = torch.zeros(self.num_envs, 4, device=self.device, dtype=q_policy.dtype)
        if leg == "RR":
            support_preload[:, 0] = float(self.cfg.rear_lift_same_front_preload_m)
            support_preload[:, 1] = float(self.cfg.rear_lift_diagonal_front_preload_m)
            support_preload[:, 3] = float(self.cfg.rear_lift_other_rear_preload_m)
        else:
            support_preload[:, 0] = float(self.cfg.rear_lift_diagonal_front_preload_m)
            support_preload[:, 1] = float(self.cfg.rear_lift_same_front_preload_m)
            support_preload[:, 2] = float(self.cfg.rear_lift_other_rear_preload_m)
        down_signs = torch.tensor(
            self.cfg.rear_lift_foot_down_signs,
            device=self.device,
            dtype=q_policy.dtype,
        )
        support_preload *= down_signs.unsqueeze(0)
        support_preload *= preload_gate.unsqueeze(1)
        unload_delta = float(self.cfg.rear_lift_target_unload_m) * unload_gate

        default_thigh = q_policy[:, 1::3]
        default_calf = q_policy[:, 2::3]
        x_default, z_default = self.reference._forward_sagittal(default_thigh, default_calf)
        x_target = x_default - shift_x.unsqueeze(1)
        z_target = z_default + support_preload
        z_target[:, leg_index] += unload_delta
        z_target[:, leg_index] += (
            float(self.cfg.rear_lift_test_height_m) * lift_progress
        )
        thigh_target, calf_target = self.reference._inverse_sagittal(x_target, z_target)
        q_policy[:, 1::3] = thigh_target
        q_policy[:, 2::3] = calf_target
        leg_length = torch.clamp(torch.abs(z_default), min=0.15)
        q_policy[:, 0::3] += -shift_y.unsqueeze(1) / leg_length

        self.reference.last_q_ref[:] = q_policy
        self.reference.last_leg_phase.zero_()
        self.reference.last_leg_phase[:, leg_index] = lift_phase
        self.reference.last_swing_mask.zero_()
        self.reference.last_swing_mask[:, leg_index] = lift_progress > 1.0e-5
        self.reference.last_active_swing_one_hot.zero_()
        self.reference.last_active_swing_one_hot[:, leg_index] = (
            lift_progress > 1.0e-5
        ).to(q_policy.dtype)
        self.reference.last_support_gate[:] = (
            ~self.reference.last_swing_mask
        ).to(q_policy.dtype)
        self.reference.last_preload_gate.zero_()
        self.reference.last_post_touchdown_gate.zero_()
        self.last_support_preload_delta_z[:] = support_preload
        self.last_target_leg_unload_delta_z[:] = unload_delta
        self.reference.last_predicted_foot_z = self.reference._forward_sagittal(
            q_policy[:, 1::3], q_policy[:, 2::3]
        )[1]
        self.reference.last_predicted_foot_lift = (
            self.reference.last_predicted_foot_z - self.reference.default_foot_z
        )
        self._rear_lift_state_step += 1
        self._rear_lift_step += 1
        return q_policy

    def _set_diagnostic_reference(self, q_policy: torch.Tensor) -> None:
        self.reference.last_q_ref[:] = q_policy
        self.reference.last_leg_phase.zero_()
        self.reference.last_swing_mask.zero_()
        self.reference.last_active_swing_one_hot.zero_()
        self.reference.last_support_gate.fill_(1.0)
        self.reference.last_preload_gate.zero_()
        self.reference.last_post_touchdown_gate.zero_()
        self.reference.last_predicted_foot_z = self.reference._forward_sagittal(
            q_policy[:, 1::3], q_policy[:, 2::3]
        )[1]
        self.reference.last_predicted_foot_lift = (
            self.reference.last_predicted_foot_z - self.reference.default_foot_z
        )

    def _press_sign_test_target(self) -> torch.Tensor:
        q_policy = self.reference.default_joint_pos.clone()
        dt = float(self._env.step_dt)
        segment = max(
            2.0 * dt,
            float(self.cfg.press_sign_rest_sec) + float(self.cfg.press_sign_hold_sec),
        )
        elapsed = self._rear_lift_step * dt
        test_index = min(7, int(elapsed / segment))
        segment_time = elapsed - test_index * segment
        leg_index = test_index // 2
        sign = 1.0 if test_index % 2 == 0 else -1.0
        applying = segment_time >= float(self.cfg.press_sign_rest_sec)
        delta = sign * float(self.cfg.press_sign_delta_m) if applying else 0.0
        forces = self._foot_normal_forces()
        if not applying:
            self.last_diagnostic_force_before[:] = forces[:, leg_index]
        self.last_diagnostic_leg.fill_(leg_index)
        self.last_diagnostic_delta_z.fill_(delta)
        self.last_diagnostic_force_after[:] = forces[:, leg_index]
        foot_z_delta = torch.zeros(self.num_envs, 4, device=self.device)
        foot_z_delta[:, leg_index] = delta
        q_policy = self._foot_target_to_policy(q_policy, foot_z_delta=foot_z_delta)
        self._set_diagnostic_reference(q_policy)
        self._rear_lift_step += 1
        return q_policy

    def _body_shift_sweep_target(self) -> torch.Tensor:
        q_policy = self.reference.default_joint_pos.clone()
        dt = float(self._env.step_dt)
        elapsed = self._rear_lift_step * dt
        settle = max(0.0, float(self.cfg.body_shift_sweep_settle_sec))
        hold = max(dt, float(self.cfg.body_shift_sweep_hold_sec))
        values = torch.linspace(
            -float(self.cfg.body_shift_sweep_extent_m),
            float(self.cfg.body_shift_sweep_extent_m),
            int(self.cfg.body_shift_sweep_points),
            device=self.device,
            dtype=q_policy.dtype,
        )
        if elapsed < settle:
            shift_x = torch.zeros(self.num_envs, device=self.device, dtype=q_policy.dtype)
            shift_y = torch.zeros_like(shift_x)
        else:
            point = int((elapsed - settle) / hold)
            point = min(point, values.numel() * values.numel() - 1)
            shift_x = values[point // values.numel()].expand(self.num_envs)
            shift_y = values[point % values.numel()].expand(self.num_envs)
        self.last_body_shift_xy[:, 0] = shift_x
        self.last_body_shift_xy[:, 1] = shift_y
        foot_x_delta = -shift_x.unsqueeze(1).expand(-1, 4)
        q_policy = self._foot_target_to_policy(
            q_policy,
            foot_x_delta=foot_x_delta,
            body_shift_y=shift_y,
        )
        self._set_diagnostic_reference(q_policy)
        self._rear_lift_step += 1
        return q_policy

    def _fast_trot_light_vmc_offsets(
        self,
        *,
        swing_mask: torch.Tensor,
        leg_phase: torch.Tensor,
        s_stance: torch.Tensor,
        touchdown_blend: torch.Tensor,
        guard_strength: torch.Tensor,
        warmup: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        profile = str(self.cfg.fast_trot_safety_profile)
        enabled = bool(self.cfg.fast_trot_enable_light_vmc) or profile in (
            "performance_soft_output_v2_light_vmc",
            "performance_soft_output_v2_light_vmc_balance",
            "performance_soft_output_v2_light_vmc_balance_v2",
            "performance_soft_output_v2_light_vmc_balance_v3",
            "performance_soft_output_v2_light_vmc_balance_v4",
        )
        if not enabled:
            self.last_light_vmc_weight.zero_()
            self.last_light_vmc_foot_z_offset.zero_()
            self.last_light_vmc_foot_x_offset.zero_()
            self.last_light_vmc_foot_y_offset.zero_()
            self.last_light_vmc_height_corr_z.zero_()
            self.last_light_vmc_roll_corr_z.zero_()
            self.last_light_vmc_pitch_corr_z.zero_()
            self.last_light_vmc_foot_x_corr.zero_()
            self.last_light_vmc_foot_y_corr.zero_()
            self.last_light_yaw_error.zero_()
            self.last_light_yaw_corr_hip_raw.zero_()
            self.last_light_yaw_corr_hip.zero_()
            self.last_light_yaw_hip_offset.zero_()
            self.last_light_yaw_hip_rate_limited.zero_()
            self.last_rear_preswing_unload_gate.zero_()
            self.last_rear_preswing_vmc_fade.fill_(1.0)
            self.last_rear_preswing_unload_z_offset.zero_()
            self.last_rear_touchdown_vmc_ramp_weight.zero_()
            self.last_rear_touchdown_kp_scale.fill_(1.0)
            self.last_phase_switch_vmc_weight_scale_applied.fill_(1.0)
            self.last_phase_switch_yaw_weight_scale_applied.fill_(1.0)
            self.last_phase_switch_kp_scale_applied.fill_(1.0)
            zeros = torch.zeros(self.num_envs, 4, device=device, dtype=dtype)
            return zeros, zeros, zeros

        touchdown_ramp = max(1.0e-6, float(self.cfg.light_vmc_touchdown_ramp))
        rear_touchdown_ramp = max(1.0e-6, float(self.cfg.rear_touchdown_vmc_ramp))
        preswing_ramp = max(1.0e-6, float(self.cfg.light_vmc_preswing_ramp))
        early_weight = self.reference._smootherstep01(torch.clamp(s_stance / touchdown_ramp, 0.0, 1.0))
        rear_early_weight = self.reference._smootherstep01(
            torch.clamp(s_stance / rear_touchdown_ramp, 0.0, 1.0)
        )
        rear_mask = torch.tensor((0.0, 0.0, 1.0, 1.0), device=device, dtype=dtype).unsqueeze(0)
        early_weight = early_weight * (1.0 - rear_mask) + rear_early_weight * rear_mask
        preswing_weight = self.reference._smootherstep01(torch.clamp((1.0 - s_stance) / preswing_ramp, 0.0, 1.0))
        stance_weight = (0.5 + 0.5 * early_weight) * preswing_weight * (~swing_mask).to(dtype)
        swing_touchdown_weight = 0.5 * touchdown_blend
        vmc_weight = torch.clamp(
            (stance_weight + swing_touchdown_weight)
            * float(self.cfg.light_vmc_max_weight)
            * warmup.unsqueeze(1),
            0.0,
            float(self.cfg.light_vmc_max_weight),
        )
        if guard_strength.numel() > 0:
            scale = 1.0 - guard_strength.unsqueeze(1) * (1.0 - float(self.cfg.light_vmc_phase_switch_weight_scale))
            vmc_weight *= scale
            self.last_phase_switch_vmc_weight_scale_applied[:] = torch.squeeze(scale[:, :1], dim=1)
        else:
            self.last_phase_switch_vmc_weight_scale_applied.fill_(1.0)
        if bool(self.cfg.rear_preswing_unload_enable):
            preswing_window = max(1.0e-6, float(self.cfg.rear_preswing_unload_window))
            fade_window = max(1.0e-6, float(self.cfg.rear_preswing_vmc_fade_window))
            rear_preswing_gate = self.reference._smootherstep01(
                torch.clamp((leg_phase - (1.0 - preswing_window)) / preswing_window, 0.0, 1.0)
            )
            rear_preswing_gate = rear_preswing_gate * rear_mask * (~swing_mask).to(dtype)
            rear_fade = 1.0 - self.reference._smootherstep01(
                torch.clamp((leg_phase - (1.0 - fade_window)) / fade_window, 0.0, 1.0)
            )
            rear_fade = rear_fade * rear_mask + (1.0 - rear_mask)
            rear_fade = torch.where(swing_mask, torch.ones_like(rear_fade), rear_fade)
            vmc_weight *= rear_fade
        else:
            rear_preswing_gate = torch.zeros(self.num_envs, 4, device=device, dtype=dtype)
            rear_fade = torch.ones(self.num_envs, 4, device=device, dtype=dtype)
        self.last_rear_preswing_unload_gate[:] = rear_preswing_gate
        self.last_rear_preswing_vmc_fade[:] = rear_fade
        self.last_rear_touchdown_vmc_ramp_weight[:] = early_weight * rear_mask

        base_roll, base_pitch, base_yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        base_lin_vel = self._asset.data.root_lin_vel_b
        base_ang_vel = self._asset.data.root_ang_vel_b
        height_corr = float(self.cfg.light_vmc_height_kp_z) * (
            float(self.cfg.light_vmc_target_base_height) - self._asset.data.root_pos_w[:, 2]
        )
        height_corr -= float(self.cfg.light_vmc_height_kd_z) * base_lin_vel[:, 2]
        height_corr = torch.clamp(
            height_corr,
            -float(self.cfg.light_vmc_height_corr_limit_m),
            float(self.cfg.light_vmc_height_corr_limit_m),
        )
        roll_corr = float(self.cfg.light_vmc_roll_kp_z) * (
            base_roll - float(self.cfg.light_vmc_target_roll)
        )
        roll_corr += float(self.cfg.light_vmc_roll_kd_z) * base_ang_vel[:, 0]
        roll_corr = torch.clamp(
            roll_corr,
            -float(self.cfg.light_vmc_roll_corr_limit_m),
            float(self.cfg.light_vmc_roll_corr_limit_m),
        )
        pitch_corr = float(self.cfg.light_vmc_pitch_kp_z) * (
            base_pitch - float(self.cfg.light_vmc_target_pitch)
        )
        pitch_corr += float(self.cfg.light_vmc_pitch_kd_z) * base_ang_vel[:, 1]
        pitch_corr = torch.clamp(
            pitch_corr,
            -float(self.cfg.light_vmc_pitch_corr_limit_m),
            float(self.cfg.light_vmc_pitch_corr_limit_m),
        )
        side = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=device, dtype=dtype)
        fore_aft = torch.tensor((1.0, 1.0, -1.0, -1.0), device=device, dtype=dtype)
        z_raw = (
            float(self.cfg.light_vmc_z_sign) * height_corr.unsqueeze(1)
            + side.unsqueeze(0) * float(self.cfg.light_vmc_roll_sign) * roll_corr.unsqueeze(1)
            + fore_aft.unsqueeze(0) * float(self.cfg.light_vmc_pitch_sign) * pitch_corr.unsqueeze(1)
        ) * vmc_weight

        if bool(self.cfg.light_vmc_enable_foot_placement):
            x_corr = float(self.cfg.light_vmc_vx_foot_k) * base_lin_vel[:, 0]
            x_corr += float(self.cfg.light_vmc_pitch_rate_foot_x_k) * base_ang_vel[:, 1]
            y_corr = float(self.cfg.light_vmc_vy_foot_k) * base_lin_vel[:, 1]
            y_corr += float(self.cfg.light_vmc_roll_rate_foot_y_k) * base_ang_vel[:, 0]
            x_corr = torch.clamp(
                x_corr,
                -float(self.cfg.light_vmc_foot_x_corr_limit_m),
                float(self.cfg.light_vmc_foot_x_corr_limit_m),
            )
            y_corr = torch.clamp(
                y_corr,
                -float(self.cfg.light_vmc_foot_y_corr_limit_m),
                float(self.cfg.light_vmc_foot_y_corr_limit_m),
            )
        else:
            x_corr = torch.zeros(self.num_envs, device=device, dtype=dtype)
            y_corr = torch.zeros_like(x_corr)
        x_raw = x_corr.unsqueeze(1) * vmc_weight
        y_raw = y_corr.unsqueeze(1) * vmc_weight

        z_limit = float(self.cfg.light_vmc_z_offset_rate_limit_m)
        xy_limit = float(self.cfg.light_vmc_xy_offset_rate_limit_m)
        z_offset = self.last_light_vmc_foot_z_offset + torch.clamp(
            z_raw - self.last_light_vmc_foot_z_offset,
            min=-z_limit,
            max=z_limit,
        )
        x_offset = self.last_light_vmc_foot_x_offset + torch.clamp(
            x_raw - self.last_light_vmc_foot_x_offset,
            min=-xy_limit,
            max=xy_limit,
        )
        y_offset = self.last_light_vmc_foot_y_offset + torch.clamp(
            y_raw - self.last_light_vmc_foot_y_offset,
            min=-xy_limit,
            max=xy_limit,
        )
        self.last_light_vmc_weight[:] = vmc_weight
        self.last_light_vmc_foot_z_offset[:] = z_offset
        self.last_light_vmc_foot_x_offset[:] = x_offset
        self.last_light_vmc_foot_y_offset[:] = y_offset
        self.last_light_vmc_height_corr_z[:] = height_corr
        self.last_light_vmc_roll_corr_z[:] = roll_corr
        self.last_light_vmc_pitch_corr_z[:] = pitch_corr
        self.last_light_vmc_foot_x_corr[:] = x_corr
        self.last_light_vmc_foot_y_corr[:] = y_corr
        if bool(self.cfg.enable_light_yaw_damping):
            invalid_yaw = ~self._light_vmc_target_yaw_valid
            if torch.any(invalid_yaw):
                self._light_vmc_target_yaw[invalid_yaw] = base_yaw[invalid_yaw]
                self._light_vmc_target_yaw_valid[invalid_yaw] = True
            yaw_error = torch.atan2(
                torch.sin(base_yaw - self._light_vmc_target_yaw),
                torch.cos(base_yaw - self._light_vmc_target_yaw),
            )
            yaw_corr = float(self.cfg.light_yaw_kp_hip) * yaw_error
            yaw_corr += float(self.cfg.light_yaw_kd_hip) * base_ang_vel[:, 2]
            yaw_corr = torch.clamp(
                yaw_corr,
                -float(self.cfg.light_yaw_hip_limit_rad),
                float(self.cfg.light_yaw_hip_limit_rad),
            )
            yaw_raw_corr = yaw_corr.clone()
            if guard_strength.numel() > 0:
                yaw_guard_scale = 1.0 - guard_strength * (
                    1.0 - float(self.cfg.light_yaw_phase_switch_weight_scale)
                )
                yaw_corr = yaw_corr * yaw_guard_scale
                self.last_phase_switch_yaw_weight_scale_applied[:] = yaw_guard_scale
            else:
                self.last_phase_switch_yaw_weight_scale_applied.fill_(1.0)
            side = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=device, dtype=dtype)
            yaw_raw = (
                side.unsqueeze(0)
                * float(self.cfg.light_yaw_sign)
                * yaw_corr.unsqueeze(1)
                * vmc_weight
                * warmup.unsqueeze(1)
            )
            yaw_limit = float(self.cfg.light_yaw_hip_rate_limit_rad)
            yaw_offset = self.last_light_yaw_hip_offset + torch.clamp(
                yaw_raw - self.last_light_yaw_hip_offset,
                min=-yaw_limit,
                max=yaw_limit,
            )
            self.last_light_yaw_error[:] = yaw_error
            self.last_light_yaw_corr_hip_raw[:] = yaw_raw_corr
            self.last_light_yaw_corr_hip[:] = yaw_corr
            self.last_light_yaw_hip_rate_limited[:] = yaw_offset - self.last_light_yaw_hip_offset
            self.last_light_yaw_hip_offset[:] = yaw_offset
        else:
            self.last_light_yaw_error.zero_()
            self.last_light_yaw_corr_hip_raw.zero_()
            self.last_light_yaw_corr_hip.zero_()
            self.last_light_yaw_hip_rate_limited.zero_()
            self.last_light_yaw_hip_offset.zero_()
            self.last_phase_switch_yaw_weight_scale_applied.fill_(1.0)
        return x_offset, y_offset, z_offset

    def _fast_diagonal_trot_target(self) -> torch.Tensor:
        q_policy = self.reference.default_joint_pos.clone()
        dt = float(self._env.step_dt)
        dtype = q_policy.dtype
        device = self.device
        warmup = torch.clamp(
            torch.full(
                (self.num_envs,),
                self._rear_lift_step * dt / max(float(self.cfg.fast_trot_warmup_sec), 1.0e-6),
                device=device,
                dtype=dtype,
            ),
            0.0,
            1.0,
        )
        frequency = torch.full(
            (self.num_envs,), float(self.cfg.fast_trot_step_hz), device=device, dtype=dtype
        )
        stride = torch.full(
            (self.num_envs,),
            float(self.cfg.fast_trot_stride_length_m),
            device=device,
            dtype=dtype,
        ) * warmup
        front_height = torch.full(
            (self.num_envs, 2),
            float(self.cfg.fast_trot_front_swing_height_m),
            device=device,
            dtype=dtype,
        )
        rear_height = torch.full(
            (self.num_envs, 2),
            float(self.cfg.fast_trot_rear_swing_height_m),
            device=device,
            dtype=dtype,
        )
        leg_height = torch.cat(
            (front_height[:, 0:1], front_height[:, 1:2], rear_height[:, 0:1], rear_height[:, 1:2]),
            dim=1,
        ) * warmup.unsqueeze(1)
        swing_height = torch.max(leg_height, dim=1).values
        self.reference.base_phase = torch.remainder(
            self.reference.base_phase + frequency * dt, 1.0
        )
        phase_a = self.reference.base_phase
        phase_b = torch.remainder(self.reference.base_phase + 0.5, 1.0)
        phase_to_switch = torch.minimum(
            torch.minimum(self.reference.base_phase, 1.0 - self.reference.base_phase),
            torch.abs(self.reference.base_phase - 0.5),
        )
        guard_window = max(1.0e-6, float(self.cfg.fast_trot_phase_switch_guard_window))
        guard_strength = self.reference._smootherstep01(
            torch.clamp((guard_window - phase_to_switch) / guard_window, 0.0, 1.0)
        )
        if str(self.cfg.fast_trot_safety_profile) not in (
            "performance_soft_output_v2_small_fix",
            "performance_soft_output_v2_light_vmc",
            "performance_soft_output_v2_light_vmc_balance",
            "performance_soft_output_v2_light_vmc_balance_v2",
            "performance_soft_output_v2_light_vmc_balance_v3",
            "performance_soft_output_v2_light_vmc_balance_v4",
        ):
            guard_strength.zero_()
        self.last_phase_to_switch[:] = phase_to_switch
        self.last_phase_switch_guard_strength[:] = guard_strength
        leg_phase = torch.zeros(self.num_envs, 4, device=device, dtype=dtype)
        leg_phase[:, [0, 3]] = phase_a.unsqueeze(1).expand(-1, 2)
        leg_phase[:, [1, 2]] = phase_b.unsqueeze(1).expand(-1, 2)
        swing_fraction = max(0.05, min(0.49, 1.0 - float(self.cfg.fast_trot_duty_factor)))
        swing_mask = leg_phase < swing_fraction
        pair_a_swing = torch.any(swing_mask[:, [0, 3]], dim=1)
        pair_b_swing = torch.any(swing_mask[:, [1, 2]], dim=1)
        self.last_active_swing_pair[:] = torch.where(
            pair_a_swing,
            torch.ones_like(self.last_active_swing_pair),
            torch.where(pair_b_swing, torch.full_like(self.last_active_swing_pair, 2), 0),
        )
        self.last_expected_support_pair[:] = torch.where(
            pair_a_swing,
            torch.full_like(self.last_expected_support_pair, 2),
            torch.where(pair_b_swing, torch.ones_like(self.last_expected_support_pair), 0),
        )

        s_swing = torch.clamp(leg_phase / swing_fraction, 0.0, 1.0)
        s_stance = torch.clamp((leg_phase - swing_fraction) / (1.0 - swing_fraction), 0.0, 1.0)
        advance = self.reference._smootherstep01(s_swing)
        peak_phase = min(0.80, max(0.20, float(self.cfg.fast_trot_swing_lift_peak_phase)))
        touchdown_phase = min(0.98, max(peak_phase + 0.05, float(self.cfg.fast_trot_touchdown_phase)))
        lift_up = self.reference._smootherstep01(torch.clamp(s_swing / peak_phase, 0.0, 1.0))
        lift_down = 1.0 - self.reference._smootherstep01(
            torch.clamp((s_swing - peak_phase) / max(touchdown_phase - peak_phase, 1.0e-6), 0.0, 1.0)
        )
        swing_shape = lift_up * lift_down * swing_mask * (s_swing < touchdown_phase)
        stance_progress = self.reference._smootherstep01(s_stance)
        early_stance = min(0.30, max(0.0, float(self.cfg.fast_trot_early_stance_blend)))
        touchdown_progress = torch.clamp(
            (s_swing - touchdown_phase) / max(1.0 - touchdown_phase, 1.0e-6),
            0.0,
            1.0,
        )
        touchdown_blend = self.reference._smootherstep01(touchdown_progress) * swing_mask
        early_stance_gate = torch.clamp(1.0 - s_stance / max(early_stance, 1.0e-6), 0.0, 1.0)
        early_stance_gate = self.reference._smootherstep01(early_stance_gate) * (~swing_mask)
        default_thigh = q_policy[:, 1::3]
        default_calf = q_policy[:, 2::3]
        x_default, z_default = self.reference._forward_sagittal(default_thigh, default_calf)
        support_gate = torch.maximum(
            (~swing_mask).to(dtype),
            torch.maximum(touchdown_blend, early_stance_gate),
        )
        profile = str(self.cfg.fast_trot_safety_profile)
        soft_output_profiles = (
            "performance_soft_output",
            "performance_soft_output_v2",
            "performance_soft_output_v2_small_fix",
            "performance_soft_output_v2_light_vmc",
            "performance_soft_output_v2_light_vmc_balance",
            "performance_soft_output_v2_light_vmc_balance_v2",
            "performance_soft_output_v2_light_vmc_balance_v3",
            "performance_soft_output_v2_light_vmc_balance_v4",
        )
        if profile in soft_output_profiles:
            ramp_in = max(1.0e-6, float(self.cfg.fast_trot_support_preload_ramp_in_phase))
            ramp_out = max(1.0e-6, float(self.cfg.fast_trot_support_preload_ramp_out_phase))
            ramp_in_gate = self.reference._smootherstep01(torch.clamp(s_stance / ramp_in, 0.0, 1.0))
            ramp_out_gate = self.reference._smootherstep01(torch.clamp((1.0 - s_stance) / ramp_out, 0.0, 1.0))
            preload_gate = ramp_in_gate * ramp_out_gate * (~swing_mask).to(dtype)
            if profile in (
                "performance_soft_output_v2",
                "performance_soft_output_v2_small_fix",
                "performance_soft_output_v2_light_vmc",
                "performance_soft_output_v2_light_vmc_balance",
                "performance_soft_output_v2_light_vmc_balance_v2",
                "performance_soft_output_v2_light_vmc_balance_v3",
                "performance_soft_output_v2_light_vmc_balance_v4",
            ):
                preload_gate = torch.clamp(
                    preload_gate,
                    max=float(self.cfg.fast_trot_support_preload_gate_max),
                )
            support_preload_gate = torch.maximum(preload_gate, touchdown_blend)
            if profile in (
                "performance_soft_output_v2",
                "performance_soft_output_v2_small_fix",
                "performance_soft_output_v2_light_vmc",
                "performance_soft_output_v2_light_vmc_balance",
                "performance_soft_output_v2_light_vmc_balance_v2",
                "performance_soft_output_v2_light_vmc_balance_v3",
                "performance_soft_output_v2_light_vmc_balance_v4",
            ):
                support_preload_gate = torch.clamp(
                    support_preload_gate,
                    max=float(self.cfg.fast_trot_support_preload_gate_max),
                )
        else:
            preload_phase = torch.remainder(leg_phase + swing_fraction, 1.0)
            preload_width = max(1.0e-6, float(self.cfg.fast_trot_preload_fraction))
            preload_gate = self.reference._smootherstep01(
                torch.clamp((preload_phase - (1.0 - preload_width)) / preload_width, 0.0, 1.0)
            )
            support_preload_gate = torch.maximum(support_gate, preload_gate * support_gate)
        support_preload = (
            -float(self.cfg.fast_trot_support_preload_z_m)
            * support_preload_gate
            * warmup.unsqueeze(1)
        )
        support_height_offset = (
            -float(self.cfg.fast_trot_global_support_height_offset_m)
            * (~swing_mask).to(dtype)
            * warmup.unsqueeze(1)
        )
        x_swing = x_default - 0.5 * stride.unsqueeze(1) + stride.unsqueeze(1) * advance
        x_stance = x_default + 0.5 * stride.unsqueeze(1) - stride.unsqueeze(1) * stance_progress
        z_swing = z_default + leg_height * swing_shape
        z_stance = z_default + support_height_offset + support_preload
        x_target = torch.where(swing_mask, x_swing, x_stance)
        if profile in soft_output_profiles:
            z_touchdown = z_swing * (1.0 - touchdown_blend) + z_stance * touchdown_blend
        else:
            z_touchdown = torch.where(touchdown_blend > 0.0, z_stance, z_swing)
        z_target = torch.where(swing_mask, z_touchdown, z_stance)
        vmc_x_offset, vmc_y_offset, vmc_z_offset = self._fast_trot_light_vmc_offsets(
            swing_mask=swing_mask,
            leg_phase=leg_phase,
            s_stance=s_stance,
            touchdown_blend=touchdown_blend,
            guard_strength=guard_strength,
            warmup=warmup,
            dtype=dtype,
        )
        rear_unload_offset = (
            float(self.cfg.rear_unload_sign)
            * float(self.cfg.rear_preswing_unload_z_m)
            * self.last_rear_preswing_unload_gate
            * warmup.unsqueeze(1)
        )
        self.last_rear_preswing_unload_z_offset[:] = rear_unload_offset
        x_target = x_target + vmc_x_offset
        z_target = z_target + vmc_z_offset + rear_unload_offset
        self.last_rear_late_swing_guard_active.zero_()
        self.last_rear_late_swing_height.zero_()
        self.last_rear_late_swing_height_error.zero_()
        self.last_rear_late_swing_descent_scale_applied.fill_(1.0)
        self.last_rear_early_contact_guard_active.zero_()
        self.last_rear_early_contact_relief_offset.zero_()
        if profile == "performance_soft_output_v2_light_vmc_balance_v4":
            rear_mask = torch.tensor((0.0, 0.0, 1.0, 1.0), device=device, dtype=dtype).unsqueeze(0)
            late_start = float(self.cfg.rear_late_swing_phase_start)
            late_end = max(late_start + 1.0e-6, float(self.cfg.rear_late_swing_phase_end))
            late_gate = self.reference._smootherstep01(
                torch.clamp((leg_phase - late_start) / (late_end - late_start), 0.0, 1.0)
            )
            late_gate *= 1.0 - self.reference._smootherstep01(
                torch.clamp((leg_phase - late_end) / 0.02, 0.0, 1.0)
            )
            late_gate = late_gate * swing_mask.to(dtype) * rear_mask
            rear_height = (z_target - z_default) * rear_mask
            min_height = float(self.cfg.rear_late_swing_min_height_m)
            height_error = torch.clamp(
                min_height + float(self.cfg.rear_late_swing_clearance_margin_m) - rear_height,
                min=0.0,
            ) * late_gate
            if bool(self.cfg.rear_late_swing_descent_soft_enable):
                descent_scale = 1.0 - late_gate * (1.0 - float(self.cfg.rear_late_swing_descent_scale))
                z_target = z_target + torch.clamp(rear_height, min=0.0) * (1.0 - descent_scale)
                self.last_rear_late_swing_descent_scale_applied[:] = descent_scale
            if bool(self.cfg.rear_late_swing_guard_enable):
                desired_clearance = (
                    float(self.cfg.rear_late_swing_clearance_sign)
                    * torch.clamp(height_error, max=0.006)
                )
                rate = float(self.cfg.rear_late_swing_guard_rate_limit_m)
                clearance_offset = self.last_rear_late_swing_clearance_offset + torch.clamp(
                    desired_clearance - self.last_rear_late_swing_clearance_offset,
                    min=-rate,
                    max=rate,
                )
                z_target = z_target + clearance_offset
                self.last_rear_late_swing_clearance_offset[:] = clearance_offset
                self.last_rear_late_swing_guard_active[:] = height_error > 1.0e-6
            else:
                self.last_rear_late_swing_clearance_offset.zero_()
            if bool(self.cfg.rear_early_contact_guard_enable):
                forces = self._foot_normal_forces().to(dtype)
                contact_start = float(self.cfg.rear_early_contact_phase_start)
                contact_end = max(contact_start + 1.0e-6, float(self.cfg.rear_early_contact_phase_end))
                contact_phase = (leg_phase >= contact_start) & (leg_phase <= contact_end)
                contact_guard = (
                    contact_phase
                    & (swing_mask | (touchdown_blend > 1.0e-6))
                    & (forces > float(self.cfg.rear_early_contact_force_threshold))
                    & (rear_mask > 0.5)
                )
                desired_relief = (
                    contact_guard.to(dtype)
                    * float(self.cfg.rear_early_contact_relief_sign)
                    * float(self.cfg.rear_early_contact_lift_relief_m)
                )
                z_target = z_target + desired_relief
                self.last_rear_early_contact_guard_active[:] = contact_guard
                self.last_rear_early_contact_relief_offset[:] = desired_relief
            self.last_rear_late_swing_height[:] = rear_height
            self.last_rear_late_swing_height_error[:] = height_error
        else:
            self.last_rear_late_swing_clearance_offset.zero_()
            self.last_rear_late_swing_descent_scale_applied.fill_(1.0)
            self.last_rear_early_contact_kp_scale.fill_(1.0)
            self.last_rear_touchdown_kp_ramp_weight.zero_()
        thigh_target, calf_target = self.reference._inverse_sagittal(x_target, z_target)
        q_policy[:, 1::3] = thigh_target
        q_policy[:, 2::3] = calf_target
        leg_length = torch.clamp(torch.abs(z_default), min=0.15)
        q_policy[:, 0::3] += -vmc_y_offset / leg_length
        q_policy[:, 0::3] += self.last_light_yaw_hip_offset

        self._apply_fast_trot_gains(
            swing_mask,
            touchdown_blend=touchdown_blend,
            early_stance_gate=early_stance_gate,
            preload_gate=preload_gate,
            phase_switch_guard_strength=guard_strength,
        )

        self.reference.last_q_ref[:] = q_policy
        self.reference.last_leg_phase[:] = leg_phase
        self.reference.last_swing_mask[:] = swing_mask
        self.reference.last_active_swing_one_hot[:] = swing_mask.to(dtype)
        self.reference.last_support_gate[:] = (~swing_mask).to(dtype)
        self.reference.last_preload_gate[:] = preload_gate
        self.reference.last_post_touchdown_gate[:] = early_stance_gate
        self.reference.last_frequency = frequency
        self.reference.last_stride = stride
        self.reference.last_swing_height = swing_height
        self.reference.last_duty_factor = torch.full_like(frequency, float(self.cfg.fast_trot_duty_factor))
        self.reference.last_warmup = warmup
        self.reference.last_predicted_foot_z = self.reference._forward_sagittal(
            q_policy[:, 1::3], q_policy[:, 2::3]
        )[1]
        self.reference.last_predicted_foot_lift = (
            self.reference.last_predicted_foot_z - self.reference.default_foot_z
        )
        self.last_support_preload_delta_z[:] = support_preload
        self.last_target_leg_unload_delta_z.zero_()
        self._rear_lift_step += 1
        return q_policy

    def _apply_fast_trot_gains(
        self,
        swing_mask: torch.Tensor,
        *,
        touchdown_blend: torch.Tensor | None = None,
        early_stance_gate: torch.Tensor | None = None,
        preload_gate: torch.Tensor | None = None,
        phase_switch_guard_strength: torch.Tensor | None = None,
    ) -> None:
        dtype = self.processed_actions.dtype
        kp = torch.zeros(self.num_envs, 12, device=self.device, dtype=dtype)
        kd = torch.zeros_like(kp)
        swing_kp = torch.tensor(
            (
                float(self.cfg.fast_trot_swing_hip_kp),
                float(self.cfg.fast_trot_swing_thigh_kp),
                float(self.cfg.fast_trot_swing_calf_kp),
            ),
            device=self.device,
            dtype=dtype,
        )
        touchdown_kp = torch.tensor(
            (
                float(self.cfg.fast_trot_touchdown_hip_kp),
                float(self.cfg.fast_trot_touchdown_thigh_kp),
                float(self.cfg.fast_trot_touchdown_calf_kp),
            ),
            device=self.device,
            dtype=dtype,
        )
        early_kp = torch.tensor(
            (
                float(self.cfg.fast_trot_early_stance_hip_kp),
                float(self.cfg.fast_trot_early_stance_thigh_kp),
                float(self.cfg.fast_trot_early_stance_calf_kp),
            ),
            device=self.device,
            dtype=dtype,
        )
        guard_kp = torch.tensor(
            (
                float(self.cfg.fast_trot_phase_switch_guard_hip_kp),
                float(self.cfg.fast_trot_phase_switch_guard_thigh_kp),
                float(self.cfg.fast_trot_phase_switch_guard_calf_kp),
            ),
            device=self.device,
            dtype=dtype,
        )
        support_kp = torch.tensor(
            (
                float(self.cfg.fast_trot_support_hip_kp),
                float(self.cfg.fast_trot_support_thigh_kp),
                float(self.cfg.fast_trot_support_calf_kp),
            ),
            device=self.device,
            dtype=dtype,
        )
        profile = str(self.cfg.fast_trot_safety_profile)
        self.last_guard_kp_scale.zero_()
        self.last_rear_touchdown_kp_scale.fill_(1.0)
        self.last_rear_early_contact_kp_scale.fill_(1.0)
        self.last_rear_touchdown_kp_ramp_weight.zero_()
        self.last_phase_switch_kp_scale_applied.fill_(1.0)
        for leg_index in range(4):
            cols = slice(leg_index * 3, leg_index * 3 + 3)
            leg_swing = swing_mask[:, leg_index].unsqueeze(1)
            if profile in (
                "performance_soft_output",
                "performance_soft_output_v2",
                "performance_soft_output_v2_small_fix",
                "performance_soft_output_v2_light_vmc",
                "performance_soft_output_v2_light_vmc_balance",
                "performance_soft_output_v2_light_vmc_balance_v2",
                "performance_soft_output_v2_light_vmc_balance_v3",
                "performance_soft_output_v2_light_vmc_balance_v4",
            ) and touchdown_blend is not None and early_stance_gate is not None:
                touchdown = touchdown_blend[:, leg_index].unsqueeze(1)
                early = early_stance_gate[:, leg_index].unsqueeze(1)
                stance = (~swing_mask[:, leg_index]).to(dtype).unsqueeze(1)
                leg_kp = torch.where(leg_swing, swing_kp.unsqueeze(0), support_kp.unsqueeze(0))
                leg_kp = leg_kp * (1.0 - early) + early_kp.unsqueeze(0) * early
                leg_kp = leg_kp * (1.0 - touchdown) + touchdown_kp.unsqueeze(0) * touchdown
                if preload_gate is not None:
                    preload = preload_gate[:, leg_index].unsqueeze(1) * stance
                    leg_kp = leg_kp * (1.0 - preload) + support_kp.unsqueeze(0) * preload
                if phase_switch_guard_strength is not None and profile in (
                "performance_soft_output_v2_small_fix",
                "performance_soft_output_v2_light_vmc_balance_v3",
                "performance_soft_output_v2_light_vmc_balance_v4",
                ):
                    guard = phase_switch_guard_strength.unsqueeze(1) * stance
                    leg_kp = leg_kp * (1.0 - guard) + guard_kp.unsqueeze(0) * guard
                    self.last_guard_kp_scale[:, cols] = guard.expand(-1, 3)
                    self.last_phase_switch_kp_scale_applied[:] = 1.0 - phase_switch_guard_strength * (
                        1.0 - float(self.cfg.fast_trot_phase_switch_kp_scale)
                    )
                if profile in (
                    "performance_soft_output_v2_light_vmc_balance_v3",
                    "performance_soft_output_v2_light_vmc_balance_v4",
                ) and leg_index >= 2:
                    rear_touchdown = torch.maximum(touchdown, early)
                    rear_touchdown = rear_touchdown * torch.clamp(
                        (~leg_swing).to(dtype) + touchdown, 0.0, 1.0
                    )
                    rear_limit_kp = torch.tensor(
                        (
                            float(self.cfg.rear_touchdown_hip_kp_limit),
                            float(self.cfg.rear_touchdown_thigh_kp_limit),
                            float(self.cfg.rear_touchdown_calf_kp_limit),
                        ),
                        device=self.device,
                        dtype=dtype,
                    ).unsqueeze(0)
                    rear_soft_kp = torch.minimum(
                        leg_kp * float(self.cfg.rear_touchdown_kp_scale),
                        rear_limit_kp,
                    )
                    leg_kp = leg_kp * (1.0 - rear_touchdown) + rear_soft_kp * rear_touchdown
                    self.last_rear_touchdown_kp_scale[:, leg_index] = torch.squeeze(
                        1.0 - rear_touchdown * (1.0 - float(self.cfg.rear_touchdown_kp_scale)),
                        dim=1,
                    )
                    self.last_rear_touchdown_kp_ramp_weight[:, leg_index] = torch.squeeze(
                        rear_touchdown, dim=1
                    )
                if profile == "performance_soft_output_v2_light_vmc_balance_v4" and leg_index >= 2:
                    early_contact = self.last_rear_early_contact_guard_active[:, leg_index].to(dtype).unsqueeze(1)
                    early_limit_kp = torch.tensor(
                        (
                            float(self.cfg.rear_early_contact_hip_kp_limit),
                            float(self.cfg.rear_early_contact_thigh_kp_limit),
                            float(self.cfg.rear_early_contact_calf_kp_limit),
                        ),
                        device=self.device,
                        dtype=dtype,
                    ).unsqueeze(0)
                    early_soft_kp = torch.minimum(
                        leg_kp * float(self.cfg.rear_early_contact_kp_scale),
                        early_limit_kp,
                    )
                    leg_kp = leg_kp * (1.0 - early_contact) + early_soft_kp * early_contact
                    self.last_rear_early_contact_kp_scale[:, leg_index] = torch.squeeze(
                        1.0 - early_contact * (1.0 - float(self.cfg.rear_early_contact_kp_scale)),
                        dim=1,
                    )
                leg_kd = torch.full((self.num_envs, 3), float(self.cfg.fast_trot_support_kd), device=self.device, dtype=dtype)
                leg_kd = torch.where(
                    leg_swing,
                    torch.full((self.num_envs, 3), float(self.cfg.fast_trot_swing_kd), device=self.device, dtype=dtype),
                    leg_kd,
                )
                leg_kd = leg_kd * (1.0 - touchdown) + float(self.cfg.fast_trot_touchdown_kd) * touchdown
                leg_kd = leg_kd * (1.0 - early) + float(self.cfg.fast_trot_early_stance_kd) * early
                if profile in (
                    "performance_soft_output_v2_light_vmc_balance_v3",
                    "performance_soft_output_v2_light_vmc_balance_v4",
                ) and leg_index >= 2:
                    rear_touchdown = torch.maximum(touchdown, early)
                    leg_kd = leg_kd * (1.0 - rear_touchdown) + float(self.cfg.rear_touchdown_kd) * rear_touchdown
                if profile == "performance_soft_output_v2_light_vmc_balance_v4" and leg_index >= 2:
                    early_contact = self.last_rear_early_contact_guard_active[:, leg_index].to(dtype).unsqueeze(1)
                    leg_kd = leg_kd * (1.0 - early_contact) + float(self.cfg.rear_early_contact_kd) * early_contact
                if phase_switch_guard_strength is not None and profile in (
                    "performance_soft_output_v2_small_fix",
                    "performance_soft_output_v2_light_vmc_balance_v3",
                ):
                    guard = phase_switch_guard_strength.unsqueeze(1) * stance
                    leg_kd = leg_kd * (1.0 - guard) + float(self.cfg.fast_trot_phase_switch_guard_kd) * guard
                kp[:, cols] = leg_kp
                kd[:, cols] = leg_kd
            else:
                kp[:, cols] = torch.where(leg_swing, swing_kp.unsqueeze(0), support_kp.unsqueeze(0))
                kd[:, cols] = torch.where(
                    leg_swing,
                    torch.full((self.num_envs, 3), float(self.cfg.fast_trot_swing_kd), device=self.device, dtype=dtype),
                    torch.full((self.num_envs, 3), float(self.cfg.fast_trot_support_kd), device=self.device, dtype=dtype),
                )
        self.last_debug_kp[:] = kp
        self.last_debug_kd[:] = kd
        self.debug_kp_override = kp
        self.debug_kd_override = kd
        self._asset.write_joint_stiffness_to_sim(kp, joint_ids=self._joint_ids)
        self._asset.write_joint_damping_to_sim(kd, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.cfg.action_mode == "reference_raw":
            self.reference.update(self._commands())
            self._set_direct_playback_output(self.reference.get_q_ref())
            return
        if self.cfg.action_mode == "csv_playback":
            if self._csv_playback is None:
                raise RuntimeError("CSV playback was not initialized.")
            q_policy = self._csv_playback.sample(self._csv_playback_time)
            self._csv_playback_time += float(self._env.step_dt)
            self._set_direct_playback_output(q_policy)
            return
        if self.cfg.action_mode == "joint_mapping_debug":
            q_ref_policy = self.reference.default_joint_pos.clone()
            joint_index = joint_mapping_index(
                self._joint_mapping_step,
                control_dt=float(self._env.step_dt),
                initial_hold_sec=float(self.cfg.joint_mapping_initial_hold_sec),
                active_hold_sec=float(self.cfg.joint_mapping_hold_sec),
                rest_sec=float(self.cfg.joint_mapping_rest_sec),
            )
            if joint_index >= 0:
                q_ref_policy[:, joint_index] += float(self.cfg.joint_mapping_delta)
            self._joint_mapping_index = joint_index
            self._joint_mapping_step += 1
            self._set_direct_playback_output(q_ref_policy)
            return
        if self.cfg.action_mode == "rear_lift_test":
            q_cpg_policy = self._rear_lift_test_target()
            q_vmc_delta = torch.zeros_like(q_cpg_policy)
        elif self.cfg.action_mode == "press_sign_test":
            q_cpg_policy = self._press_sign_test_target()
            q_vmc_delta = torch.zeros_like(q_cpg_policy)
        elif self.cfg.action_mode == "body_shift_sweep":
            q_cpg_policy = self._body_shift_sweep_target()
            q_vmc_delta = torch.zeros_like(q_cpg_policy)
        elif self.cfg.action_mode == "fast_diagonal_trot":
            q_cpg_policy = self._fast_diagonal_trot_target()
            q_vmc_delta = torch.zeros_like(q_cpg_policy)
        else:
            q_cpg_policy = self.reference.update(self._commands())
            q_vmc_delta = self._compute_vmc_delta(q_cpg_policy)
        q_ref_policy = q_cpg_policy + q_vmc_delta

        if self.cfg.action_mode != "reference_residual":
            delta = torch.zeros_like(actions)
            self._filtered_residual.zero_()
        else:
            alpha = float(self.cfg.residual_lowpass_alpha)
            self._filtered_residual.copy_(
                filter_residual(actions, self._filtered_residual, self._residual_scale, alpha)
            )
            delta = self._filtered_residual

        q_raw_policy = q_ref_policy + delta
        q_cpg_sim = self.semantic_adapter.policy_to_sim(q_cpg_policy)
        q_ref_sim = self.semantic_adapter.policy_to_sim(q_ref_policy)
        q_raw = self.semantic_adapter.policy_to_sim(q_raw_policy)
        q_before_joint_limit = q_raw.clone()
        if self.cfg.action_mode in (
            "reference_stage",
            "rear_lift_test",
            "press_sign_test",
            "body_shift_sweep",
            "fast_diagonal_trot",
        ):
            q_raw = self._clamp_to_hard_joint_limits(q_raw)
        elif self.cfg.clip is not None:
            q_raw = torch.clamp(q_raw, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._deploy_q_raw[:] = q_raw
        self._record_raw_target_rate(q_raw)
        self.last_q_cpg_policy[:] = q_cpg_policy
        self.last_q_cpg[:] = q_cpg_sim
        self.last_q_vmc_delta[:] = q_vmc_delta
        self.last_q_ref_policy[:] = q_ref_policy
        self.last_q_ref[:] = q_ref_sim
        self.last_delta_q_rl[:] = delta
        self.last_q_raw_policy[:] = q_raw_policy
        self.last_q_raw_reference[:] = q_raw
        self.last_q_after_joint_limit[:] = q_raw
        self.last_joint_limit_clip_mask[:] = torch.abs(q_raw - q_before_joint_limit) > 1.0e-6
        self.last_joint_limit_clipping_ratio[:] = torch.mean(
            self.last_joint_limit_clip_mask.to(q_raw.dtype), dim=1
        )
        self.last_joint_limit_margin[:] = torch.minimum(
            q_before_joint_limit - self._hard_joint_lower,
            self._hard_joint_upper - q_before_joint_limit,
        )
        self.last_joint_limit_warning[:] = torch.any(
            self.last_joint_limit_margin < float(self.cfg.joint_limit_warning_margin_rad),
            dim=1,
        )

        profile = str(self.cfg.fast_trot_safety_profile)
        monitor_only = self.cfg.action_mode == "fast_diagonal_trot" and profile == "monitor_only"
        if not self.cfg.enable_deploy_target_filter or monitor_only:
            self._processed_actions[:] = q_raw
            self.last_q_cmd[:] = q_raw
            self.last_q_after_rate_limit[:] = q_raw
            self.last_q_after_accel_limit[:] = q_raw
            self.last_q_after_torque_clip[:] = q_raw
            self.last_q_before_delay[:] = q_raw
            self.last_q_after_delay[:] = q_raw
            self.last_qdot_cmd.zero_()
            self._record_raw_risk_debug(q_raw)
            self.last_rate_clip_mask.zero_()
            self.last_accel_clip_mask.zero_()
            self.last_torque_clip_mask.zero_()
            self.last_rate_limit_delta.zero_()
            self.last_accel_limit_delta.zero_()
            self.last_torque_clip_delta.zero_()
            self.last_rate_clipping_ratio.zero_()
            self.last_accel_clipping_ratio.zero_()
            self.last_filter_error.zero_()
            self.last_filter_clipping_ratio.zero_()
            self.last_torque_clipping_ratio.zero_()
            return

        q_current = self._asset.data.joint_pos[:, self._joint_ids]
        uninit = ~self._initialized
        if torch.any(uninit):
            self._q_last_cmd[uninit] = q_current[uninit]
            self._qdot_last_cmd[uninit] = 0.0
            self._delay_buffer[uninit] = q_current[uninit].unsqueeze(1)
            self._initialized[uninit] = True

        dt = float(self._env.step_dt)
        kp_eff, kd_eff = self._actual_pd_gains()
        torque_budget = self._per_joint_torque_budget()
        if profile in (
            "performance_safe",
            "performance_soft_output",
            "performance_soft_output_v2",
            "performance_soft_output_v2_small_fix",
            "performance_soft_output_v2_light_vmc",
            "performance_soft_output_v2_light_vmc_balance",
            "performance_soft_output_v2_light_vmc_balance_v2",
            "performance_soft_output_v2_light_vmc_balance_v3",
            "performance_soft_output_v2_light_vmc_balance_v4",
        ):
            limit_budget = torch.full_like(torque_budget, float(self.cfg.sim_hard_torque_budget))
        else:
            limit_budget = torque_budget
        err_limit = (limit_budget / torch.clamp(kp_eff, min=1.0e-6)) * self._err_limit_mul
        self._update_safety_debug(
            kp_eff=kp_eff,
            kd_eff=kd_eff,
            torque_budget=limit_budget,
            err_limit=err_limit,
        )
        self.last_q_error_raw_ref[:] = q_raw - q_current
        self.last_tau_est_raw_ref[:] = self._pd_torque_for(q_raw, kp_eff, kd_eff)
        damping_scale = torch.sqrt(torch.clamp(self._kd_scale, min=0.5, max=2.0))
        rate_limit = (self._target_rate_limit / damping_scale) * self._target_rate_mul
        accel_limit = (self._target_accel_limit / damping_scale) * self._target_accel_mul

        qdot_desired = (q_raw - self._q_last_cmd) / dt
        if self.cfg.enable_target_rate_limit:
            if profile in (
                "performance_soft_output",
                "performance_soft_output_v2",
                "performance_soft_output_v2_small_fix",
                "performance_soft_output_v2_light_vmc",
                "performance_soft_output_v2_light_vmc_balance",
                "performance_soft_output_v2_light_vmc_balance_v2",
                "performance_soft_output_v2_light_vmc_balance_v3",
                "performance_soft_output_v2_light_vmc_balance_v4",
            ):
                max_step = rate_limit * dt
                target_step = q_raw - self._q_last_cmd
                step = torch.clamp(target_step, min=-max_step, max=max_step)
                q_after_rate = self._q_last_cmd + step
                crossed = (target_step * (q_raw - q_after_rate)) < 0.0
                q_after_rate = torch.where(crossed, q_raw, q_after_rate)
                qdot_rate = (q_after_rate - self._q_last_cmd) / dt
            else:
                qdot_rate = torch.clamp(qdot_desired, min=-rate_limit, max=rate_limit)
                q_after_rate = self._q_last_cmd + qdot_rate * dt
        else:
            qdot_rate = qdot_desired
            q_after_rate = self._q_last_cmd + qdot_rate * dt
        self.last_q_after_rate_limit[:] = q_after_rate
        self.last_rate_limit_delta[:] = q_after_rate - q_raw
        self.last_rate_clip_mask[:] = torch.abs(self.last_rate_limit_delta) > 1.0e-6
        self.last_rate_clipping_ratio[:] = torch.mean(self.last_rate_clip_mask.to(q_raw.dtype), dim=1)
        self.last_tau_est_after_rate[:] = self._pd_torque_for(q_after_rate, kp_eff, kd_eff)

        if self.cfg.enable_target_accel_limit:
            qdot_delta = torch.clamp(
                qdot_rate - self._qdot_last_cmd,
                min=-accel_limit * dt,
                max=accel_limit * dt,
            )
            qdot_cmd = self._qdot_last_cmd + qdot_delta
        else:
            qdot_cmd = qdot_rate
        q_after_accel = self._q_last_cmd + qdot_cmd * dt
        self.last_q_after_accel_limit[:] = q_after_accel
        self.last_accel_limit_delta[:] = q_after_accel - q_after_rate
        self.last_accel_clip_mask[:] = torch.abs(self.last_accel_limit_delta) > 1.0e-6
        self.last_accel_clipping_ratio[:] = torch.mean(self.last_accel_clip_mask.to(q_raw.dtype), dim=1)

        self.last_tau_est_after_accel[:] = self._pd_torque_for(q_after_accel, kp_eff, kd_eff)
        if self.cfg.enable_torque_target_limit:
            if profile == "performance_safe":
                q_after_torque = self._performance_safe_torque_target(q_after_accel, q_current, kp_eff)
            elif profile in (
                "performance_soft_output",
                "performance_soft_output_v2",
                "performance_soft_output_v2_small_fix",
                "performance_soft_output_v2_light_vmc",
                "performance_soft_output_v2_light_vmc_balance",
                "performance_soft_output_v2_light_vmc_balance_v2",
                "performance_soft_output_v2_light_vmc_balance_v3",
                "performance_soft_output_v2_light_vmc_balance_v4",
            ):
                guard_for_torque = (
                    self.last_phase_switch_guard_strength
                    if profile in (
                        "performance_soft_output_v2_small_fix",
                        "performance_soft_output_v2_light_vmc_balance_v3",
                        "performance_soft_output_v2_light_vmc_balance_v4",
                    )
                    else None
                )
                early_contact_guard = (
                    torch.max(self.last_rear_early_contact_guard_active.to(q_raw.dtype), dim=1).values
                    if profile == "performance_soft_output_v2_light_vmc_balance_v4"
                    else None
                )
                q_after_torque = self._performance_soft_output_torque_target(
                    q_after_accel,
                    q_current,
                    q_raw,
                    kp_eff,
                    kd_eff,
                    guard_strength=guard_for_torque,
                    early_contact_guard_strength=early_contact_guard,
                )
            else:
                q_after_torque = q_current + torch.clamp(
                    q_after_accel - q_current, min=-err_limit, max=err_limit
                )
        else:
            q_after_torque = q_after_accel
        self.last_q_after_torque_clip[:] = q_after_torque
        self.last_torque_clip_delta[:] = q_after_torque - q_after_accel
        self.last_torque_clip_mask[:] = torch.abs(self.last_torque_clip_delta) > 1.0e-6
        self.last_torque_clipping_ratio[:] = torch.mean(self.last_torque_clip_mask.to(q_raw.dtype), dim=1)

        actual_qdot_cmd = (q_after_torque - self._q_last_cmd) / dt
        self._q_last_cmd[:] = q_after_torque
        self._qdot_last_cmd[:] = actual_qdot_cmd
        self.last_q_before_delay[:] = q_after_torque
        self._delay_buffer = torch.roll(self._delay_buffer, shifts=1, dims=1)
        self._delay_buffer[:, 0] = q_after_torque
        if self.cfg.enable_action_delay:
            delay_idx = torch.clamp(
                self._motor_delay_steps.squeeze(-1), 0, self._delay_buffer.shape[1] - 1
            )
            q_final = self._delay_buffer[
                torch.arange(self.num_envs, device=self.device), delay_idx
            ]
        else:
            q_final = q_after_torque
        self._processed_actions = q_final
        self.last_q_after_delay[:] = q_final
        self.last_q_cmd[:] = q_final
        self.last_qdot_cmd[:] = actual_qdot_cmd
        self.last_tau_est_cmd_final[:] = self._pd_torque_for(q_final, kp_eff, kd_eff)
        self.last_tau_est[:] = self.last_tau_est_cmd_final
        self._update_torque_threshold_debug()
        self.last_filter_error[:] = torch.mean(torch.abs(q_raw - q_final), dim=1)
        self.last_filter_clipping_ratio[:] = torch.mean(
            (torch.abs(q_raw - q_final) > 1.0e-4).to(q_raw.dtype), dim=1
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.reference.reset(env_ids)
        self._csv_playback_time[env_ids] = 0.0
        self._filtered_residual[env_ids] = 0.0
        self._filtered_vmc_delta[env_ids] = 0.0
        self.last_q_cpg_policy[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_q_cpg[env_ids] = self.semantic_adapter.policy_to_sim(
            self.reference.default_joint_pos[env_ids]
        )
        self.last_q_vmc_delta[env_ids] = 0.0
        self.last_q_ref_policy[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_q_ref[env_ids] = self.semantic_adapter.policy_to_sim(
            self.reference.default_joint_pos[env_ids]
        )
        self._previous_raw_target[env_ids] = self.last_q_ref[env_ids]
        self.last_raw_target_rate[env_ids] = 0.0
        self.last_delta_q_rl[env_ids] = 0.0
        self.last_q_raw_policy[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_q_raw_reference[env_ids] = self.last_q_ref[env_ids]
        self.last_q_after_joint_limit[env_ids] = self.last_q_ref[env_ids]
        self.last_q_after_rate_limit[env_ids] = self.last_q_ref[env_ids]
        self.last_q_after_accel_limit[env_ids] = self.last_q_ref[env_ids]
        self.last_q_after_torque_clip[env_ids] = self.last_q_ref[env_ids]
        self.last_q_before_delay[env_ids] = self.last_q_ref[env_ids]
        self.last_q_after_delay[env_ids] = self.last_q_ref[env_ids]
        self.last_tau_est[env_ids] = 0.0
        self.last_tau_est_raw_ref[env_ids] = 0.0
        self.last_tau_est_after_rate[env_ids] = 0.0
        self.last_tau_est_after_accel[env_ids] = 0.0
        self.last_tau_est_cmd_final[env_ids] = 0.0
        self.last_q_error_raw_ref[env_ids] = 0.0
        self.last_rate_demand[env_ids] = 0.0
        self.last_accel_demand[env_ids] = 0.0
        self._previous_rate_demand[env_ids] = 0.0
        self.last_rate_limit_delta[env_ids] = 0.0
        self.last_accel_limit_delta[env_ids] = 0.0
        self.last_torque_clip_delta[env_ids] = 0.0
        self.last_joint_limit_margin[env_ids] = 0.0
        self.last_joint_limit_warning[env_ids] = False
        self.last_kp_actual[env_ids] = max(float(self.cfg.sim_kp), 1.0e-6)
        self.last_kd_actual[env_ids] = max(float(self.cfg.sim_kd), 0.0)
        self.last_torque_budget_per_joint[env_ids] = 0.0
        self.last_err_limit_per_joint[env_ids] = 0.0
        self.last_joint_limit_clip_mask[env_ids] = False
        self.last_rate_clip_mask[env_ids] = False
        self.last_accel_clip_mask[env_ids] = False
        self.last_torque_clip_mask[env_ids] = False
        self.last_joint_limit_clipping_ratio[env_ids] = 0.0
        self.last_rate_clipping_ratio[env_ids] = 0.0
        self.last_accel_clipping_ratio[env_ids] = 0.0
        self.last_filter_error[env_ids] = 0.0
        self.last_filter_clipping_ratio[env_ids] = 0.0
        self.last_torque_clipping_ratio[env_ids] = 0.0
        self.last_over_8nm_ratio[env_ids] = 0.0
        self.last_over_12nm_ratio[env_ids] = 0.0
        self.last_over_17nm_ratio[env_ids] = 0.0
        self.last_rear_lift_phase[env_ids] = 0
        self.last_support_preload_delta_z[env_ids] = 0.0
        self.last_target_leg_unload_delta_z[env_ids] = 0.0
        self._rear_lift_state_step[env_ids] = 0
        self._rear_lift_force_drop_steps[env_ids] = 0
        self.last_force_drop_success[env_ids] = False
        self.last_failure_reason[env_ids] = 0
        self.last_force_below_threshold[env_ids] = False
        self.last_force_below_timer[env_ids] = 0.0
        self.last_first_force_drop_time[env_ids] = -1.0
        self.last_lift_entry_time[env_ids] = -1.0
        self.last_missed_force_drop_window[env_ids] = False
        self.last_state_transition_reason[env_ids] = 0
        self.last_active_swing_pair[env_ids] = 0
        self.last_expected_support_pair[env_ids] = 0
        self.last_phase_switch_guard_strength[env_ids] = 0.0
        self.last_phase_to_switch[env_ids] = 0.0
        self.last_guard_kp_scale[env_ids] = 0.0
        self.last_light_vmc_weight[env_ids] = 0.0
        self.last_light_vmc_foot_z_offset[env_ids] = 0.0
        self.last_light_vmc_foot_x_offset[env_ids] = 0.0
        self.last_light_vmc_foot_y_offset[env_ids] = 0.0
        self.last_light_vmc_height_corr_z[env_ids] = 0.0
        self.last_light_vmc_roll_corr_z[env_ids] = 0.0
        self.last_light_vmc_pitch_corr_z[env_ids] = 0.0
        self.last_light_vmc_foot_x_corr[env_ids] = 0.0
        self.last_light_vmc_foot_y_corr[env_ids] = 0.0
        self._light_vmc_target_yaw[env_ids] = 0.0
        self._light_vmc_target_yaw_valid[env_ids] = False
        self.last_light_yaw_error[env_ids] = 0.0
        self.last_light_yaw_corr_hip_raw[env_ids] = 0.0
        self.last_light_yaw_corr_hip[env_ids] = 0.0
        self.last_light_yaw_hip_offset[env_ids] = 0.0
        self.last_light_yaw_hip_rate_limited[env_ids] = 0.0
        self.last_rear_preswing_unload_gate[env_ids] = 0.0
        self.last_rear_preswing_vmc_fade[env_ids] = 1.0
        self.last_rear_preswing_unload_z_offset[env_ids] = 0.0
        self.last_rear_touchdown_vmc_ramp_weight[env_ids] = 0.0
        self.last_rear_touchdown_kp_scale[env_ids] = 1.0
        self.last_rear_late_swing_guard_active[env_ids] = False
        self.last_rear_late_swing_clearance_offset[env_ids] = 0.0
        self.last_rear_late_swing_height[env_ids] = 0.0
        self.last_rear_late_swing_height_error[env_ids] = 0.0
        self.last_rear_late_swing_descent_scale_applied[env_ids] = 1.0
        self.last_rear_early_contact_guard_active[env_ids] = False
        self.last_rear_early_contact_relief_offset[env_ids] = 0.0
        self.last_rear_early_contact_kp_scale[env_ids] = 1.0
        self.last_rear_touchdown_kp_ramp_weight[env_ids] = 0.0
        self.last_phase_switch_vmc_weight_scale_applied[env_ids] = 1.0
        self.last_phase_switch_yaw_weight_scale_applied[env_ids] = 1.0
        self.last_phase_switch_kp_scale_applied[env_ids] = 1.0
        self.last_debug_kp[env_ids] = max(float(self.cfg.sim_kp), 1.0e-6)
        self.last_debug_kd[env_ids] = max(float(self.cfg.sim_kd), 0.0)
        self.last_body_shift_xy[env_ids] = 0.0
        self.last_diagnostic_leg[env_ids] = -1
        self.last_diagnostic_delta_z[env_ids] = 0.0
        self.last_diagnostic_force_before[env_ids] = 0.0
        self.last_diagnostic_force_after[env_ids] = 0.0
        if env_ids.numel() == self.num_envs:
            self._joint_mapping_step = 0
            self._joint_mapping_index = -1
            self._rear_lift_step = 0
        previous_reward_residual = getattr(self, "_previous_residual_for_reward", None)
        if previous_reward_residual is not None:
            previous_reward_residual[env_ids] = 0.0

    def get_debug_info(self) -> dict[str, torch.Tensor]:
        debug = dict(self.reference.get_debug_info())
        if self.cfg.action_mode == "fast_diagonal_trot":
            debug["duty_factor"] = torch.full(
                (self.num_envs,),
                float(self.cfg.fast_trot_duty_factor),
                device=self.device,
            )
        trunk_pos_w = self._asset.data.body_pos_w[:, self._trunk_body_ids, :]
        trunk_quat_w = self._asset.data.body_quat_w[:, self._trunk_body_ids, :]
        foot_from_trunk_w = self._asset.data.body_pos_w[:, self._foot_body_ids, :] - trunk_pos_w
        foot_from_trunk_b = quat_rotate_inverse(
            trunk_quat_w.expand(-1, foot_from_trunk_w.shape[1], -1).reshape(-1, 4),
            foot_from_trunk_w.reshape(-1, 3),
        ).reshape(self.num_envs, -1, 3)
        base_roll, base_pitch, base_yaw = euler_xyz_from_quat(
            self._asset.data.root_quat_w
        )
        foot_normal_force = self._foot_normal_forces()
        active_one_hot = self.reference.last_active_swing_one_hot
        active_leg = torch.where(
            torch.sum(active_one_hot, dim=1) > 0.0,
            torch.argmax(active_one_hot, dim=1),
            torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
        )
        debug.update(
            {
                "control_stage": torch.full(
                    (self.num_envs,), int(self.cfg.control_stage), device=self.device, dtype=torch.long
                ),
                "active_swing_leg": active_leg,
                "joint_mapping_index": torch.full(
                    (self.num_envs,), self._joint_mapping_index, device=self.device, dtype=torch.long
                ),
                "stance_mask": ~self.reference.last_swing_mask,
                "q_cpg_policy": self.last_q_cpg_policy,
                "q_cpg_simulator": self.last_q_cpg,
                "q_vmc_delta": self.last_q_vmc_delta,
                "policy_q_ref": self.last_q_ref_policy,
                "simulator_q_ref": self.last_q_ref,
                "q_after_joint_limit": self.last_q_after_joint_limit,
                "q_after_rate_limit": self.last_q_after_rate_limit,
                "q_after_accel_limit": self.last_q_after_accel_limit,
                "q_after_torque_clip": self.last_q_after_torque_clip,
                "q_before_delay": self.last_q_before_delay,
                "q_after_delay": self.last_q_after_delay,
                "final_q_cmd": self.last_q_cmd,
                "joint_limit_clipping_ratio": self.last_joint_limit_clipping_ratio,
                "rate_limit_clipping_ratio": self.last_rate_clipping_ratio,
                "acceleration_clipping_ratio": self.last_accel_clipping_ratio,
                "filter_clipping_ratio": self.last_filter_clipping_ratio,
                "torque_clipping_ratio": self.last_torque_clipping_ratio,
                "target_error_clipping_ratio": self.last_torque_clipping_ratio,
                "joint_limit_clip_mask": self.last_joint_limit_clip_mask,
                "rate_limit_clip_mask": self.last_rate_clip_mask,
                "acceleration_clip_mask": self.last_accel_clip_mask,
                "torque_clip_mask": self.last_torque_clip_mask,
                "tau_est_per_joint": self.last_tau_est,
                "tau_est_raw_ref": self.last_tau_est_raw_ref,
                "tau_est_after_rate": self.last_tau_est_after_rate,
                "tau_est_after_accel": self.last_tau_est_after_accel,
                "tau_est_cmd_final": self.last_tau_est_cmd_final,
                "q_error_raw_ref": self.last_q_error_raw_ref,
                "rate_demand": self.last_rate_demand,
                "accel_demand": self.last_accel_demand,
                "rate_limit_delta": self.last_rate_limit_delta,
                "accel_limit_delta": self.last_accel_limit_delta,
                "torque_clip_delta": self.last_torque_clip_delta,
                "joint_limit_margin": self.last_joint_limit_margin,
                "joint_limit_warning": self.last_joint_limit_warning,
                "kp_actual": self.last_kp_actual,
                "kd_actual": self.last_kd_actual,
                "torque_budget_per_joint": self.last_torque_budget_per_joint,
                "err_limit_per_joint": self.last_err_limit_per_joint,
                "joint_kp": (
                    self.debug_kp_override
                    if self.debug_kp_override is not None
                    else self.last_debug_kp
                ),
                "joint_kd": (
                    self.debug_kd_override
                    if self.debug_kd_override is not None
                    else self.last_debug_kd
                ),
                "raw_target_rate_per_joint": self.last_raw_target_rate,
                "raw_target_rate_max": torch.max(
                    torch.abs(self.last_raw_target_rate), dim=1
                ).values,
                "rate_demand_max": torch.max(torch.abs(self.last_rate_demand), dim=1).values,
                "accel_demand_max": torch.max(torch.abs(self.last_accel_demand), dim=1).values,
                "rate_limit_delta_max": torch.max(torch.abs(self.last_rate_limit_delta), dim=1).values,
                "accel_limit_delta_max": torch.max(torch.abs(self.last_accel_limit_delta), dim=1).values,
                "torque_clip_delta_max": torch.max(torch.abs(self.last_torque_clip_delta), dim=1).values,
                "tau_est_raw_ref_max": torch.max(torch.abs(self.last_tau_est_raw_ref), dim=1).values,
                "tau_est_after_rate_max": torch.max(torch.abs(self.last_tau_est_after_rate), dim=1).values,
                "tau_est_after_accel_max": torch.max(torch.abs(self.last_tau_est_after_accel), dim=1).values,
                "tau_est_cmd_final_max": torch.max(torch.abs(self.last_tau_est_cmd_final), dim=1).values,
                "q_error_raw_ref_max": torch.max(torch.abs(self.last_q_error_raw_ref), dim=1).values,
                "tau_est_max": torch.max(torch.abs(self.last_tau_est_cmd_final), dim=1).values,
                "tau_est_mean": torch.mean(torch.abs(self.last_tau_est_cmd_final), dim=1),
                "over_6nm_ratio": torch.mean((torch.abs(self.last_tau_est_cmd_final) > 6.0).to(self.last_tau_est.dtype), dim=1),
                "over_8nm_ratio": self.last_over_8nm_ratio,
                "over_10nm_ratio": torch.mean((torch.abs(self.last_tau_est_cmd_final) > 10.0).to(self.last_tau_est.dtype), dim=1),
                "over_12nm_ratio": self.last_over_12nm_ratio,
                "over_17nm_ratio": self.last_over_17nm_ratio,
                "over_8nm_raw_ratio": self._ratio_over(self.last_tau_est_raw_ref, 8.0),
                "over_12nm_raw_ratio": self._ratio_over(self.last_tau_est_raw_ref, 12.0),
                "over_17nm_raw_ratio": self._ratio_over(
                    self.last_tau_est_raw_ref, float(self.cfg.sim_hard_torque_budget)
                ),
                "over_8nm_cmd_ratio": self._ratio_over(self.last_tau_est_cmd_final, 8.0),
                "over_12nm_cmd_ratio": self._ratio_over(self.last_tau_est_cmd_final, 12.0),
                "over_17nm_cmd_ratio": self._ratio_over(
                    self.last_tau_est_cmd_final, float(self.cfg.sim_hard_torque_budget)
                ),
                "delay_steps": self._motor_delay_steps.squeeze(-1),
                "control_dt": torch.full((self.num_envs,), float(self._env.step_dt), device=self.device),
                "physics_dt": torch.full((self.num_envs,), float(self._env.cfg.sim.dt), device=self.device),
                "decimation": torch.full((self.num_envs,), float(self._env.cfg.decimation), device=self.device),
                "phase_increment_per_step": self.reference.last_frequency * float(self._env.step_dt),
                "phase_cycle_time": torch.where(
                    self.reference.last_frequency > 0.0,
                    1.0 / self.reference.last_frequency,
                    torch.zeros_like(self.reference.last_frequency),
                ),
                "predicted_foot_height": self.reference.last_predicted_foot_lift,
                "actual_foot_height": self._asset.data.body_pos_w[:, self._foot_body_ids, 2],
                "foot_sphere_bottom_z": self._asset.data.body_pos_w[:, self._foot_body_ids, 2]
                - float(self.cfg.diagnostic_foot_sphere_radius_m),
                "foot_sphere_radius_m": torch.full(
                    (self.num_envs,),
                    float(self.cfg.diagnostic_foot_sphere_radius_m),
                    device=self.device,
                ),
                "actual_foot_height_body": (
                    foot_from_trunk_b[:, :, 2]
                ),
                "base_height": self._asset.data.root_pos_w[:, 2],
                "base_rpy": torch.stack((base_roll, base_pitch, base_yaw), dim=1),
                "foot_normal_force": foot_normal_force,
                "foot_contact_state": foot_normal_force > float(
                    self.cfg.rear_lift_contact_force_threshold_n
                ),
                "rear_lift_phase": self.last_rear_lift_phase,
                "support_preload_delta_z": self.last_support_preload_delta_z,
                "target_leg_unload_delta_z": self.last_target_leg_unload_delta_z,
                "body_shift_xy": self.last_body_shift_xy,
                "diagnostic_leg": self.last_diagnostic_leg,
                "diagnostic_delta_z": self.last_diagnostic_delta_z,
                "diagnostic_force_before": self.last_diagnostic_force_before,
                "diagnostic_force_after": self.last_diagnostic_force_after,
                "force_drop_success": self.last_force_drop_success,
                "failure_reason": self.last_failure_reason,
                "force_below_threshold": self.last_force_below_threshold,
                "force_below_timer": self.last_force_below_timer,
                "first_force_drop_time": self.last_first_force_drop_time,
                "lift_entry_time": self.last_lift_entry_time,
                "missed_force_drop_window": self.last_missed_force_drop_window,
                "state_transition_reason": self.last_state_transition_reason,
                "active_swing_pair": self.last_active_swing_pair,
                "expected_support_pair": self.last_expected_support_pair,
                "phase_switch_guard_active": self.last_phase_switch_guard_strength > 1.0e-6,
                "phase_switch_guard_strength": self.last_phase_switch_guard_strength,
                "phase_to_switch": self.last_phase_to_switch,
                "phase_switch_vmc_weight_scale_applied": self.last_phase_switch_vmc_weight_scale_applied,
                "phase_switch_yaw_weight_scale_applied": self.last_phase_switch_yaw_weight_scale_applied,
                "phase_switch_kp_scale_applied": self.last_phase_switch_kp_scale_applied,
                "guard_kp_scale": self.last_guard_kp_scale,
                "light_vmc_enabled": torch.full(
                    (self.num_envs,),
                    bool(self.cfg.fast_trot_enable_light_vmc)
                    or str(self.cfg.fast_trot_safety_profile) in (
                        "performance_soft_output_v2_light_vmc",
                        "performance_soft_output_v2_light_vmc_balance",
                        "performance_soft_output_v2_light_vmc_balance_v2",
                        "performance_soft_output_v2_light_vmc_balance_v3",
                        "performance_soft_output_v2_light_vmc_balance_v4",
                    ),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "light_vmc_foot_placement_enabled": torch.full(
                    (self.num_envs,),
                    bool(self.cfg.light_vmc_enable_foot_placement),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "light_vmc_height_sign": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_vmc_z_sign),
                    device=self.device,
                ),
                "light_vmc_roll_sign": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_vmc_roll_sign),
                    device=self.device,
                ),
                "light_vmc_pitch_sign": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_vmc_pitch_sign),
                    device=self.device,
                ),
                "light_vmc_target_base_height": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_vmc_target_base_height),
                    device=self.device,
                ),
                "light_vmc_target_roll": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_vmc_target_roll),
                    device=self.device,
                ),
                "light_vmc_target_pitch": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_vmc_target_pitch),
                    device=self.device,
                ),
                "light_yaw_damping_enabled": torch.full(
                    (self.num_envs,),
                    bool(self.cfg.enable_light_yaw_damping),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "light_yaw_sign": torch.full(
                    (self.num_envs,),
                    float(self.cfg.light_yaw_sign),
                    device=self.device,
                ),
                "light_yaw_target": self._light_vmc_target_yaw,
                "light_yaw_error": self.last_light_yaw_error,
                "light_yaw_corr_hip_raw": self.last_light_yaw_corr_hip_raw,
                "light_yaw_corr_hip": self.last_light_yaw_corr_hip,
                "yaw_hip_offset": self.last_light_yaw_hip_offset,
                "yaw_hip_rate_limited": self.last_light_yaw_hip_rate_limited,
                "rear_preswing_unload_enabled": torch.full(
                    (self.num_envs,),
                    bool(self.cfg.rear_preswing_unload_enable),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "rear_unload_sign": torch.full(
                    (self.num_envs,),
                    float(self.cfg.rear_unload_sign),
                    device=self.device,
                ),
                "rear_preswing_unload_gate": self.last_rear_preswing_unload_gate,
                "rear_preswing_vmc_fade": self.last_rear_preswing_vmc_fade,
                "rear_preswing_unload_z_offset": self.last_rear_preswing_unload_z_offset,
                "rear_touchdown_vmc_ramp_weight": self.last_rear_touchdown_vmc_ramp_weight,
                "rear_touchdown_kp_scale": self.last_rear_touchdown_kp_scale,
                "rear_late_swing_guard_active": self.last_rear_late_swing_guard_active,
                "rear_late_swing_clearance_offset": self.last_rear_late_swing_clearance_offset,
                "rear_late_swing_clearance_sign": torch.full(
                    (self.num_envs,),
                    float(self.cfg.rear_late_swing_clearance_sign),
                    device=self.device,
                ),
                "rear_late_swing_height": self.last_rear_late_swing_height,
                "rear_late_swing_height_error": self.last_rear_late_swing_height_error,
                "rear_late_swing_descent_scale_applied": self.last_rear_late_swing_descent_scale_applied,
                "rear_early_contact_guard_active": self.last_rear_early_contact_guard_active,
                "rear_early_contact_relief_offset": self.last_rear_early_contact_relief_offset,
                "rear_early_contact_kp_scale": self.last_rear_early_contact_kp_scale,
                "rear_touchdown_kp_ramp_weight": self.last_rear_touchdown_kp_ramp_weight,
                "vmc_weight": self.last_light_vmc_weight,
                "vmc_height_corr_z": self.last_light_vmc_height_corr_z,
                "vmc_roll_corr_z": self.last_light_vmc_roll_corr_z,
                "vmc_pitch_corr_z": self.last_light_vmc_pitch_corr_z,
                "vmc_foot_x_corr": self.last_light_vmc_foot_x_corr,
                "vmc_foot_y_corr": self.last_light_vmc_foot_y_corr,
                "vmc_foot_z_offset": self.last_light_vmc_foot_z_offset,
                "vmc_foot_x_offset": self.last_light_vmc_foot_x_offset,
                "vmc_foot_y_offset": self.last_light_vmc_foot_y_offset,
                "base_lin_vel": self._asset.data.root_lin_vel_b,
                "base_ang_vel": self._asset.data.root_ang_vel_b,
                "global_support_height_offset_m": torch.full(
                    (self.num_envs,),
                    float(self.cfg.fast_trot_global_support_height_offset_m),
                    device=self.device,
                ),
            }
        )
        return debug


@configclass
class WaveResidualJointPositionActionCfg(DeployFilteredJointPositionActionCfg):
    class_type: type = WaveResidualJointPositionAction
    action_mode: str = "reference_residual"
    command_name: str = "base_velocity"
    reference_cfg: FanfanReferenceGaitCfg = FanfanReferenceGaitCfg()
    semantic_cfg: FanfanJointSemanticCfg = FanfanJointSemanticCfg()
    residual_scale_default: float = 0.08
    residual_scale_hip: float = 0.05
    residual_scale_thigh: float = 0.08
    residual_scale_calf: float = 0.10
    residual_lowpass_alpha: float = 0.30
    enable_target_rate_limit: bool = True
    enable_torque_target_limit: bool = True
    enable_target_accel_limit: bool = True
    enable_action_delay: bool = True
    fixed_delay_steps: int | None = None
    sim_kd: float = 5.0
    joint_mapping_delta: float = 0.10
    joint_mapping_hold_sec: float = 1.0
    joint_mapping_rest_sec: float = 1.0
    joint_mapping_initial_hold_sec: float = 2.0
    rear_lift_test_leg: str = "RR"
    rear_lift_test_thigh: float = 0.3491
    rear_lift_test_calf: float = -0.7854
    rear_lift_test_height_m: float = 0.030
    rear_lift_test_settle_sec: float = 1.5
    rear_lift_pre_shift_sec: float = 0.75
    rear_lift_test_preload_sec: float = 0.75
    rear_lift_unload_sec: float = 0.75
    rear_lift_test_cycle_sec: float = 2.0
    # RR unload naturally forms the FR+RL support pair; RL mirrors to FL+RR.
    # Do not force the opposite front leg into the primary support role.
    rear_lift_diagonal_front_preload_m: float = 0.0
    rear_lift_same_front_preload_m: float = 0.015
    rear_lift_other_rear_preload_m: float = 0.015
    rear_lift_foot_down_signs: tuple[float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    )
    rear_lift_target_unload_m: float = 0.012
    rear_lift_body_shift_x_m: float = 0.030
    rear_lift_body_shift_y_m: float = 0.010
    rear_lift_force_drop_threshold_n: float = 3.0
    rear_lift_force_confirm_sec: float = 0.20
    rear_lift_force_drop_timeout_sec: float = 1.0
    rear_lift_contact_force_threshold_n: float = 1.0
    press_sign_delta_m: float = 0.010
    press_sign_rest_sec: float = 1.0
    press_sign_hold_sec: float = 1.0
    body_shift_sweep_extent_m: float = 0.030
    body_shift_sweep_points: int = 7
    body_shift_sweep_settle_sec: float = 1.5
    body_shift_sweep_hold_sec: float = 0.75
    fast_trot_preset: str = "conservative"
    fast_trot_step_hz: float = 1.10
    fast_trot_duty_factor: float = 0.62
    fast_trot_stride_length_m: float = 0.020
    fast_trot_front_swing_height_m: float = 0.045
    fast_trot_rear_swing_height_m: float = 0.065
    fast_trot_warmup_sec: float = 2.0
    fast_trot_support_preload_z_m: float = 0.008
    fast_trot_preload_fraction: float = 0.12
    fast_trot_support_preload_ramp_in_phase: float = 0.15
    fast_trot_support_preload_ramp_out_phase: float = 0.15
    fast_trot_support_preload_gate_max: float = 1.0
    fast_trot_global_support_height_offset_m: float = 0.0
    fast_trot_phase_switch_guard_window: float = 0.04
    fast_trot_swing_lift_peak_phase: float = 0.45
    fast_trot_touchdown_phase: float = 0.82
    fast_trot_early_stance_blend: float = 0.12
    fast_trot_swing_hip_kp: float = 50.0
    fast_trot_swing_thigh_kp: float = 80.0
    fast_trot_swing_calf_kp: float = 80.0
    fast_trot_swing_kd: float = 4.5
    fast_trot_touchdown_hip_kp: float = 58.0
    fast_trot_touchdown_thigh_kp: float = 115.0
    fast_trot_touchdown_calf_kp: float = 125.0
    fast_trot_touchdown_kd: float = 5.0
    fast_trot_early_stance_hip_kp: float = 63.0
    fast_trot_early_stance_thigh_kp: float = 135.0
    fast_trot_early_stance_calf_kp: float = 145.0
    fast_trot_early_stance_kd: float = 5.1
    fast_trot_phase_switch_guard_hip_kp: float = 58.0
    fast_trot_phase_switch_guard_thigh_kp: float = 125.0
    fast_trot_phase_switch_guard_calf_kp: float = 135.0
    fast_trot_phase_switch_guard_kd: float = 6.2
    fast_trot_phase_switch_kp_scale: float = 0.75
    fast_trot_support_hip_kp: float = 70.0
    fast_trot_support_thigh_kp: float = 180.0
    fast_trot_support_calf_kp: float = 200.0
    fast_trot_support_kd: float = 5.0
    fast_trot_safety_profile: str = "monitor_only"
    fast_trot_continuous_warn_torque_budget: float = 8.0
    fast_trot_soft_peak_torque_budget: float = 12.0
    fast_trot_soft_output_start_torque: float = 10.0
    fast_trot_soft_output_full_torque: float = 14.0
    fast_trot_guard_soft_start_torque: float = 9.5
    fast_trot_guard_soft_full_torque: float = 13.5
    fast_trot_soft_output_max_ref_cmd_error_rad: float = 0.50
    fast_trot_enable_light_vmc: bool = False
    light_vmc_target_base_height: float = 0.290
    light_vmc_target_roll: float = 0.0
    light_vmc_target_pitch: float = 0.0
    light_vmc_height_kp_z: float = 0.45
    light_vmc_height_kd_z: float = 0.06
    light_vmc_height_corr_limit_m: float = 0.005
    light_vmc_roll_kp_z: float = 0.030
    light_vmc_roll_kd_z: float = 0.006
    light_vmc_roll_corr_limit_m: float = 0.004
    light_vmc_pitch_kp_z: float = 0.035
    light_vmc_pitch_kd_z: float = 0.006
    light_vmc_pitch_corr_limit_m: float = 0.004
    light_vmc_z_sign: float = 1.0
    light_vmc_roll_sign: float = 1.0
    light_vmc_pitch_sign: float = 1.0
    light_vmc_touchdown_ramp: float = 0.12
    light_vmc_preswing_ramp: float = 0.12
    light_vmc_max_weight: float = 1.0
    light_vmc_phase_switch_weight_scale: float = 0.6
    light_vmc_z_offset_rate_limit_m: float = 0.001
    light_vmc_xy_offset_rate_limit_m: float = 0.001
    light_vmc_enable_foot_placement: bool = True
    light_vmc_vx_foot_k: float = 0.025
    light_vmc_vy_foot_k: float = 0.020
    light_vmc_pitch_rate_foot_x_k: float = 0.005
    light_vmc_roll_rate_foot_y_k: float = 0.005
    light_vmc_foot_x_corr_limit_m: float = 0.006
    light_vmc_foot_y_corr_limit_m: float = 0.004
    enable_light_yaw_damping: bool = False
    light_yaw_kp_hip: float = 0.004
    light_yaw_kd_hip: float = 0.010
    light_yaw_hip_limit_rad: float = 0.010
    light_yaw_hip_rate_limit_rad: float = 0.002
    light_yaw_phase_switch_weight_scale: float = 0.40
    light_yaw_sign: float = 1.0
    rear_preswing_unload_enable: bool = False
    rear_preswing_unload_window: float = 0.14
    rear_preswing_unload_z_m: float = 0.003
    rear_preswing_vmc_fade_window: float = 0.14
    rear_unload_sign: float = 1.0
    rear_touchdown_vmc_ramp: float = 0.16
    rear_touchdown_kp_ramp: float = 0.18
    rear_touchdown_kp_scale: float = 0.75
    rear_touchdown_hip_kp_limit: float = 55.0
    rear_touchdown_thigh_kp_limit: float = 125.0
    rear_touchdown_calf_kp_limit: float = 135.0
    rear_touchdown_kd: float = 6.2
    rear_late_swing_guard_enable: bool = False
    rear_late_swing_phase_start: float = 0.28
    rear_late_swing_phase_end: float = 0.38
    rear_late_swing_clearance_margin_m: float = 0.003
    rear_late_swing_min_height_m: float = 0.003
    rear_late_swing_guard_rate_limit_m: float = 0.001
    rear_late_swing_clearance_sign: float = 1.0
    rear_late_swing_descent_soft_enable: bool = False
    rear_late_swing_descent_scale: float = 0.50
    rear_late_swing_descent_rate_limit_m: float = 0.001
    rear_early_contact_guard_enable: bool = False
    rear_early_contact_force_threshold: float = 10.0
    rear_early_contact_phase_start: float = 0.28
    rear_early_contact_phase_end: float = 0.40
    rear_early_contact_lift_relief_m: float = 0.002
    rear_early_contact_relief_sign: float = 1.0
    rear_early_contact_kp_scale: float = 0.60
    rear_early_contact_hip_kp_limit: float = 55.0
    rear_early_contact_thigh_kp_limit: float = 115.0
    rear_early_contact_calf_kp_limit: float = 115.0
    rear_early_contact_kd: float = 6.5
    rear_early_contact_torque_soft_start: float = 9.0
    rear_early_contact_torque_soft_full: float = 13.0
    sim_hard_torque_budget: float = 17.0
    diagnostic_foot_sphere_radius_m: float = 0.018
    joint_limit_warning_margin_rad: float = 0.02
    joint_limit_warning_interval_sec: float = 1.0
    csv_playback_path: str = "logs/reference_debug/fanfan_gait_playback.csv"
    control_stage: int = 1
    enable_vmc: bool = False
    vmc_mode: str = "off"
    vmc_roll_kp_m_per_rad: float = 0.020
    vmc_roll_kd_m_per_rad_s: float = 0.003
    vmc_pitch_kp_m_per_rad: float = 0.015
    vmc_pitch_kd_m_per_rad_s: float = 0.003
    vmc_body_height_target_m: float = 0.293
    vmc_height_kp: float = 0.10
    vmc_foot_z_limit_m: float = 0.006
    vmc_joint_delta_limit_rad: float = 0.03
    vmc_joint_rate_limit_rad_s: float = 0.5
    vmc_lowpass_alpha: float = 0.20
    vmc_stance_blend_fraction: float = 0.06
    full_vmc_provider: FullVmcProvider | None = None
