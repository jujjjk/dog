from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.deploy_actions import (
    DeployFilteredJointPositionAction,
    DeployFilteredJointPositionActionCfg,
)

from .reference_gait import FanfanReferenceGait, FanfanReferenceGaitCfg
from .residual_math import filter_residual


class WaveResidualJointPositionAction(DeployFilteredJointPositionAction):
    """Wave-gait reference plus bounded residual, followed by deployment-like filtering."""

    cfg: WaveResidualJointPositionActionCfg

    def __init__(self, cfg: "WaveResidualJointPositionActionCfg", env):
        super().__init__(cfg, env)
        max_delay = max(0, int(cfg.sim_motor_delay_steps_range[1]))
        self._delay_buffer = torch.zeros(
            self.num_envs, max_delay + 1, self.action_dim, device=self.device
        )
        default_q = self._asset.data.default_joint_pos[:, self._joint_ids]
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self.reference = FanfanReferenceGait(
            cfg=cfg.reference_cfg,
            num_envs=self.num_envs,
            device=self.device,
            dt=float(self._env.step_dt),
            default_joint_pos=default_q,
            joint_limits=(limits[:, :, 0], limits[:, :, 1]),
        )
        self._residual_scale = self._make_residual_scale()
        self._filtered_residual = torch.zeros_like(self.processed_actions)
        self.last_q_ref = default_q.clone()
        self.last_delta_q_rl = torch.zeros_like(self.processed_actions)
        self.last_q_raw_reference = default_q.clone()
        self.last_filter_error = torch.zeros(self.num_envs, device=self.device)

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

    def _sample_filter_params(self, env_ids: torch.Tensor) -> None:
        super()._sample_filter_params(env_ids)
        d0, d1 = self.cfg.sim_motor_delay_steps_range
        shape = (env_ids.numel(), 1)
        self._motor_delay_steps[env_ids] = torch.randint(
            max(0, int(d0)),
            max(0, int(d1)) + 1,
            shape,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        q_ref = self.reference.update(self._commands())
        if self.cfg.action_mode == "reference_only":
            delta = torch.zeros_like(actions)
            self._filtered_residual.zero_()
        else:
            alpha = float(self.cfg.residual_lowpass_alpha)
            self._filtered_residual.copy_(
                filter_residual(actions, self._filtered_residual, self._residual_scale, alpha)
            )
            delta = self._filtered_residual

        q_raw = q_ref + delta
        if self.cfg.clip is not None:
            q_raw = torch.clamp(q_raw, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._deploy_q_raw[:] = q_raw
        self.last_q_ref[:] = q_ref
        self.last_delta_q_rl[:] = delta
        self.last_q_raw_reference[:] = q_raw

        if not self.cfg.enable_deploy_target_filter:
            self._processed_actions[:] = q_raw
            self.last_q_cmd[:] = q_raw
            self.last_qdot_cmd.zero_()
            self.last_filter_error.zero_()
            return

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
        self.last_filter_error[:] = torch.mean(torch.abs(q_raw - q_cmd), dim=1)

        self._delay_buffer = torch.roll(self._delay_buffer, shifts=1, dims=1)
        self._delay_buffer[:, 0] = q_cmd
        delay_idx = torch.clamp(self._motor_delay_steps.squeeze(-1), 0, self._delay_buffer.shape[1] - 1)
        self._processed_actions = self._delay_buffer[torch.arange(self.num_envs, device=self.device), delay_idx]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.reference.reset(env_ids)
        self._filtered_residual[env_ids] = 0.0
        self.last_q_ref[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_delta_q_rl[env_ids] = 0.0
        self.last_q_raw_reference[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_filter_error[env_ids] = 0.0
        previous_reward_residual = getattr(self, "_previous_residual_for_reward", None)
        if previous_reward_residual is not None:
            previous_reward_residual[env_ids] = 0.0


@configclass
class WaveResidualJointPositionActionCfg(DeployFilteredJointPositionActionCfg):
    class_type: type = WaveResidualJointPositionAction
    action_mode: str = "reference_residual"
    command_name: str = "base_velocity"
    reference_cfg: FanfanReferenceGaitCfg = FanfanReferenceGaitCfg()
    residual_scale_default: float = 0.08
    residual_scale_hip: float = 0.05
    residual_scale_thigh: float = 0.08
    residual_scale_calf: float = 0.10
    residual_lowpass_alpha: float = 0.30
