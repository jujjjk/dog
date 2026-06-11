from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from isaaclab.utils import configclass

from .cpg_cfg import CPGCfg
from .cpg_generator import QuadrupedCPG
from .deploy_actions import DeployFilteredJointPositionAction, DeployFilteredJointPositionActionCfg


class CPGFilteredJointPositionAction(DeployFilteredJointPositionAction):
    """Joint-position action term with optional CPG-only/residual target generation."""

    cfg: CPGFilteredJointPositionActionCfg

    def __init__(self, cfg: "CPGFilteredJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self.cpg_cfg = cfg.cpg_cfg if cfg.cpg_cfg is not None else CPGCfg()
        cpg_joint_order = tuple(self.cpg_cfg.joint_order)
        action_joint_order = tuple(self._joint_names)
        if set(cpg_joint_order) != set(action_joint_order):
            raise RuntimeError(
                "CPG/action joint names mismatch. "
                f"cpg={cpg_joint_order}, action={action_joint_order}."
            )

        self._cpg_from_action_ids = torch.tensor(
            [action_joint_order.index(name) for name in cpg_joint_order],
            device=self.device,
            dtype=torch.long,
        )
        self._action_from_cpg_ids = torch.tensor(
            [cpg_joint_order.index(name) for name in action_joint_order],
            device=self.device,
            dtype=torch.long,
        )
        self._hip_action_ids = torch.tensor(
            [idx for idx, name in enumerate(action_joint_order) if "hip" in name],
            device=self.device,
            dtype=torch.long,
        )
        self._hip_action_leg_ids = torch.tensor(
            [self.cpg_cfg.leg_order.index(action_joint_order[idx].split("_")[0]) for idx in self._hip_action_ids.tolist()],
            device=self.device,
            dtype=torch.long,
        )
        self._hip_gate_side_signs_cpg = torch.as_tensor(
            self.cpg_cfg.hip_gate_side_signs,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 4)
        self._hip_gate_side_signs_action = self._hip_gate_side_signs_cpg[:, self._hip_action_leg_ids]

        default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
        default_joint_pos_cpg_order = default_joint_pos[:, self._cpg_from_action_ids]
        limits = None
        if hasattr(self._asset.data, "joint_pos_limits") and self._asset.data.joint_pos_limits is not None:
            joint_limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
            joint_limits = joint_limits[:, self._cpg_from_action_ids]
            limits = (joint_limits[:, :, 0], joint_limits[:, :, 1])
        self._cpg = QuadrupedCPG(
            cfg=self.cpg_cfg,
            device=self.device,
            num_envs=self.num_envs,
            dt=float(self._env.step_dt),
            default_joint_pos=default_joint_pos_cpg_order,
            joint_limits=limits,
        )
        self.last_q_cpg = torch.zeros_like(self.processed_actions)
        self.last_delta_q_rl = torch.zeros_like(self.processed_actions)
        self.last_clipped_actions = torch.zeros_like(self.processed_actions)
        self.last_q_raw_cpg = torch.zeros_like(self.processed_actions)
        self.last_clip_count = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_balance_delta = torch.zeros(self.num_envs, len(self.cpg_cfg.leg_order), device=self.device)
        self.last_hip_residual_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_filter_error = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_saturation_count = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_gate_clamp_count = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_gate_clamp_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_stance_inward_violation = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_swing_over_outward_violation = torch.zeros(self.num_envs, device=self.device)
        self.last_hip_outward_before_gate = torch.zeros(self.num_envs, self._hip_action_ids.numel(), device=self.device)
        self.last_hip_outward_after_gate = torch.zeros_like(self.last_hip_outward_before_gate)

    @property
    def cpg(self) -> QuadrupedCPG:
        return self._cpg

    def _commands(self) -> torch.Tensor:
        try:
            return self._env.command_manager.get_command(self.cfg.command_name)
        except Exception:
            return torch.zeros(self.num_envs, 3, device=self.device)

    def _apply_phase_aware_hip_gate(
        self,
        q_raw: torch.Tensor,
        q_cpg: torch.Tensor,
        delta_q_rl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self.cpg_cfg, "enable_phase_aware_hip_gate", False)):
            self.last_hip_gate_clamp_count.zero_()
            self.last_hip_gate_clamp_ratio.zero_()
            self.last_hip_stance_inward_violation.zero_()
            self.last_hip_swing_over_outward_violation.zero_()
            self.last_hip_outward_before_gate.zero_()
            self.last_hip_outward_after_gate.zero_()
            return q_raw, delta_q_rl

        hip_ids = self._hip_action_ids
        default_hip = self._asset.data.default_joint_pos[:, self._joint_ids][:, hip_ids]
        side_sign = self._hip_gate_side_signs_action.to(device=q_raw.device, dtype=q_raw.dtype)
        phase01 = torch.remainder(self._cpg.last_leg_phase / (2.0 * math.pi), 1.0)
        phase01 = phase01[:, self._hip_action_leg_ids]
        swing_fraction = max(1.0 - float(self.cpg_cfg.duty_factor), 0.05)
        swing = phase01 < swing_fraction
        stance = ~swing
        moving = (self._cpg.last_frequency > 1.0e-6).unsqueeze(1)

        q_raw_hip = q_raw[:, hip_ids]
        outward_before = side_sign * (q_raw_hip - default_hip)
        outward_after = outward_before.clone()

        stance_min = float(getattr(self.cpg_cfg, "hip_gate_stance_min_outward", 0.0))
        swing_max = float(getattr(self.cpg_cfg, "hip_gate_swing_max_outward", 1.0e6))
        stance_low = stance & moving & (outward_after < stance_min)
        swing_high = swing & moving & (outward_after > swing_max)
        outward_after = torch.where(stance_low, torch.full_like(outward_after, stance_min), outward_after)
        outward_after = torch.where(swing_high, torch.full_like(outward_after, swing_max), outward_after)

        changed = torch.abs(outward_after - outward_before) > 1.0e-7
        q_gated = q_raw.clone()
        q_gated[:, hip_ids] = default_hip + side_sign * outward_after
        delta_gated = delta_q_rl.clone()
        delta_gated[:, hip_ids] = q_gated[:, hip_ids] - q_cpg[:, hip_ids]

        self.last_hip_outward_before_gate[:] = outward_before
        self.last_hip_outward_after_gate[:] = outward_after
        self.last_hip_gate_clamp_count[:] = torch.sum(changed.float(), dim=1)
        self.last_hip_gate_clamp_ratio[:] = self.last_hip_gate_clamp_count / max(float(hip_ids.numel()), 1.0)
        self.last_hip_stance_inward_violation[:] = torch.mean(
            torch.clamp(stance_min - outward_before, min=0.0) * stance.float() * moving.float(),
            dim=1,
        )
        self.last_hip_swing_over_outward_violation[:] = torch.mean(
            torch.clamp(outward_before - swing_max, min=0.0) * swing.float() * moving.float(),
            dim=1,
        )
        return q_gated, delta_gated

    def process_actions(self, actions: torch.Tensor):
        mode = self.cfg.action_mode
        if not bool(self.cpg_cfg.enable) or mode == "pure_rl" or self.cpg_cfg.mode == "off":
            super().process_actions(actions)
            self.last_q_cpg[:] = self._offset if isinstance(self._offset, torch.Tensor) else 0.0
            self.last_delta_q_rl.zero_()
            self.last_clipped_actions[:] = torch.clamp(actions, -1.0, 1.0)
            self.last_q_raw_cpg[:] = self._deploy_q_raw
            self.last_clip_count[:] = torch.sum(torch.abs(actions) > 1.0, dim=1)
            hip_actions = actions[:, self._hip_action_ids]
            self.last_hip_balance_delta.zero_()
            self.last_hip_residual_abs_mean.zero_()
            self.last_hip_filter_error[:] = torch.mean(
                torch.abs(self.last_q_raw_cpg[:, self._hip_action_ids] - self.last_q_cmd[:, self._hip_action_ids]),
                dim=1,
            )
            self.last_hip_saturation_count[:] = torch.sum(torch.abs(hip_actions) > 0.75, dim=1)
            self.last_hip_gate_clamp_count.zero_()
            self.last_hip_gate_clamp_ratio.zero_()
            self.last_hip_stance_inward_violation.zero_()
            self.last_hip_swing_over_outward_violation.zero_()
            self.last_hip_outward_before_gate.zero_()
            self.last_hip_outward_after_gate.zero_()
            return

        self._raw_actions[:] = actions
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        self.last_clipped_actions[:] = clipped_actions
        self.last_clip_count[:] = torch.sum(torch.abs(actions) > 1.0, dim=1)

        commands = self._commands()
        q_cpg = self._cpg.update(commands)[:, self._action_from_cpg_ids]
        residual_limit = self._cpg.residual_limits[:, self._action_from_cpg_ids]
        if mode == "cpg_only":
            delta_q_rl = torch.zeros_like(clipped_actions)
        else:
            delta_q_rl = torch.clamp(clipped_actions * residual_limit, min=-residual_limit, max=residual_limit)

        q_raw_unclipped = q_cpg + delta_q_rl
        if mode == "cpg_residual":
            q_raw_unclipped, delta_q_rl = self._apply_phase_aware_hip_gate(q_raw_unclipped, q_cpg, delta_q_rl)
        else:
            self.last_hip_gate_clamp_count.zero_()
            self.last_hip_gate_clamp_ratio.zero_()
            self.last_hip_stance_inward_violation.zero_()
            self.last_hip_swing_over_outward_violation.zero_()
            self.last_hip_outward_before_gate.zero_()
            self.last_hip_outward_after_gate.zero_()
        q_raw = q_raw_unclipped
        hip_clip_hit = torch.zeros(self.num_envs, self._hip_action_ids.numel(), dtype=torch.bool, device=self.device)
        if self.cfg.clip is not None:
            q_raw = torch.clamp(q_raw, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
            hip_clip_hit = torch.abs(q_raw_unclipped[:, self._hip_action_ids] - q_raw[:, self._hip_action_ids]) > 1.0e-6

        self._processed_actions[:] = q_raw
        self._deploy_q_raw[:] = q_raw
        self.last_q_cpg[:] = q_cpg
        self.last_delta_q_rl[:] = delta_q_rl
        self.last_q_raw_cpg[:] = q_raw
        self.last_hip_balance_delta[:] = getattr(self._cpg, "last_hip_balance_delta", self.last_hip_balance_delta)
        self.last_hip_residual_abs_mean[:] = torch.mean(torch.abs(delta_q_rl[:, self._hip_action_ids]), dim=1)
        hip_action_saturated = torch.abs(actions[:, self._hip_action_ids]) > 0.75
        self.last_hip_saturation_count[:] = torch.sum(hip_action_saturated | hip_clip_hit, dim=1)

        if not self.cfg.enable_deploy_target_filter:
            self.last_q_cmd[:] = self.processed_actions
            self.last_qdot_cmd.zero_()
            self.last_hip_filter_error.zero_()
            return

        # Reuse the deploy-like limiter from the parent class, but start from
        # the CPG/residual q_raw target instead of affine action scaling.
        q_current = self._asset.data.joint_pos[:, self._joint_ids]
        uninit = ~self._initialized
        if torch.any(uninit):
            self._q_last_cmd[uninit] = q_current[uninit]
            self._qdot_last_cmd[uninit] = 0.0
            self._delay_buffer[uninit] = q_current[uninit].unsqueeze(1)
            self._initialized[uninit] = True

        dt = float(self._env.step_dt)
        kp_eff = max(float(self.cfg.sim_kp), 1.0e-6) * self._kp_scale * self._motor_strength
        err_limit = (self._torque_budget / kp_eff) * self._err_limit_mul
        damping_scale = torch.sqrt(torch.clamp(self._kd_scale, min=0.5, max=2.0))
        rate_limit = (self._target_rate_limit / damping_scale) * self._target_rate_mul
        accel_limit = (self._target_accel_limit / damping_scale) * self._target_accel_mul

        q_safe = q_current + torch.clamp(q_raw - q_current, min=-err_limit, max=err_limit)
        qdot_raw = torch.clamp((q_safe - self._q_last_cmd) / dt, min=-rate_limit, max=rate_limit)
        qdot_delta = torch.clamp(qdot_raw - self._qdot_last_cmd, min=-accel_limit * dt, max=accel_limit * dt)
        qdot_cmd = self._qdot_last_cmd + qdot_delta
        q_cmd = self._q_last_cmd + qdot_cmd * dt
        q_cmd = q_current + torch.clamp(q_cmd - q_current, min=-err_limit, max=err_limit)

        self._q_last_cmd[:] = q_cmd
        self._qdot_last_cmd[:] = qdot_cmd
        self.last_q_cmd[:] = q_cmd
        self.last_qdot_cmd[:] = qdot_cmd
        self.last_hip_filter_error[:] = torch.mean(torch.abs(q_raw[:, self._hip_action_ids] - q_cmd[:, self._hip_action_ids]), dim=1)

        self._delay_buffer = torch.roll(self._delay_buffer, shifts=1, dims=1)
        self._delay_buffer[:, 0] = q_cmd
        delay_idx = torch.clamp(self._motor_delay_steps.squeeze(-1) - 1, min=0, max=self._delay_buffer.shape[1] - 1)
        self._processed_actions = self._delay_buffer[torch.arange(self.num_envs, device=self.device), delay_idx]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._cpg.reset(env_ids_tensor)
        self.last_q_cpg[env_ids_tensor] = self._asset.data.default_joint_pos[env_ids_tensor][:, self._joint_ids]
        self.last_delta_q_rl[env_ids_tensor] = 0.0
        self.last_hip_balance_delta[env_ids_tensor] = 0.0
        self.last_hip_residual_abs_mean[env_ids_tensor] = 0.0
        self.last_hip_filter_error[env_ids_tensor] = 0.0
        self.last_hip_saturation_count[env_ids_tensor] = 0.0
        self.last_hip_gate_clamp_count[env_ids_tensor] = 0.0
        self.last_hip_gate_clamp_ratio[env_ids_tensor] = 0.0
        self.last_hip_stance_inward_violation[env_ids_tensor] = 0.0
        self.last_hip_swing_over_outward_violation[env_ids_tensor] = 0.0
        self.last_hip_outward_before_gate[env_ids_tensor] = 0.0
        self.last_hip_outward_after_gate[env_ids_tensor] = 0.0


@configclass
class CPGFilteredJointPositionActionCfg(DeployFilteredJointPositionActionCfg):
    """Deploy-like joint target filter with optional CPG residual target generation."""

    class_type: type = CPGFilteredJointPositionAction
    action_mode: str = "cpg_residual"  # pure_rl, cpg_only, cpg_residual
    command_name: str = "base_velocity"
    cpg_cfg: CPGCfg | None = CPGCfg()
