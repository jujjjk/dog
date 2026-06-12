from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.deploy_actions import (
    DeployFilteredJointPositionAction,
    DeployFilteredJointPositionActionCfg,
)

from .joint_semantics import FanfanJointSemanticAdapter, FanfanJointSemanticCfg
from .reference_gait import FanfanReferenceGait, FanfanReferenceGaitCfg
from .residual_math import filter_residual


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
            joint_limits=limits_policy,
        )
        self._residual_scale = self._make_residual_scale()
        self._filtered_residual = torch.zeros_like(self.processed_actions)
        self.last_q_ref_policy = self.reference.default_joint_pos.clone()
        self.last_q_ref = self.semantic_adapter.policy_to_sim(self.last_q_ref_policy)
        self.last_delta_q_rl = torch.zeros_like(self.processed_actions)
        self.last_q_raw_policy = self.last_q_ref_policy.clone()
        self.last_q_raw_reference = self.last_q_ref.clone()
        self.last_filter_error = torch.zeros(self.num_envs, device=self.device)
        self.last_filter_clipping_ratio = torch.zeros(self.num_envs, device=self.device)
        self.last_torque_clipping_ratio = torch.zeros(self.num_envs, device=self.device)

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
        q_ref_policy = self.reference.update(self._commands())
        if self.cfg.action_mode == "reference_only":
            delta = torch.zeros_like(actions)
            self._filtered_residual.zero_()
        else:
            alpha = float(self.cfg.residual_lowpass_alpha)
            self._filtered_residual.copy_(
                filter_residual(actions, self._filtered_residual, self._residual_scale, alpha)
            )
            delta = self._filtered_residual

        q_raw_policy = q_ref_policy + delta
        q_raw = self.semantic_adapter.policy_to_sim(q_raw_policy)
        if self.cfg.clip is not None:
            q_raw = torch.clamp(q_raw, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._deploy_q_raw[:] = q_raw
        self.last_q_ref_policy[:] = q_ref_policy
        self.last_q_ref[:] = self.semantic_adapter.policy_to_sim(q_ref_policy)
        self.last_delta_q_rl[:] = delta
        self.last_q_raw_policy[:] = q_raw_policy
        self.last_q_raw_reference[:] = q_raw

        if not self.cfg.enable_deploy_target_filter:
            self._processed_actions[:] = q_raw
            self.last_q_cmd[:] = q_raw
            self.last_qdot_cmd.zero_()
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
        kp_eff = max(float(self.cfg.sim_kp), 1.0e-6) * self._kp_scale * self._motor_strength
        err_limit = (self._torque_budget / kp_eff) * self._err_limit_mul
        damping_scale = torch.sqrt(torch.clamp(self._kd_scale, min=0.5, max=2.0))
        rate_limit = (self._target_rate_limit / damping_scale) * self._target_rate_mul
        accel_limit = (self._target_accel_limit / damping_scale) * self._target_accel_mul

        q_safe = q_current + torch.clamp(q_raw - q_current, min=-err_limit, max=err_limit)
        self.last_torque_clipping_ratio[:] = torch.mean(
            (torch.abs(q_safe - q_raw) > 1.0e-6).to(q_raw.dtype), dim=1
        )
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
        self.last_filter_clipping_ratio[:] = torch.mean(
            (torch.abs(q_raw - q_cmd) > 1.0e-4).to(q_raw.dtype), dim=1
        )

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
        self.last_q_ref_policy[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_q_ref[env_ids] = self.semantic_adapter.policy_to_sim(
            self.reference.default_joint_pos[env_ids]
        )
        self.last_delta_q_rl[env_ids] = 0.0
        self.last_q_raw_policy[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_q_raw_reference[env_ids] = self.last_q_ref[env_ids]
        self.last_filter_error[env_ids] = 0.0
        self.last_filter_clipping_ratio[env_ids] = 0.0
        self.last_torque_clipping_ratio[env_ids] = 0.0
        previous_reward_residual = getattr(self, "_previous_residual_for_reward", None)
        if previous_reward_residual is not None:
            previous_reward_residual[env_ids] = 0.0

    def get_debug_info(self) -> dict[str, torch.Tensor]:
        debug = dict(self.reference.get_debug_info())
        active_one_hot = self.reference.last_active_swing_one_hot
        active_leg = torch.where(
            torch.sum(active_one_hot, dim=1) > 0.0,
            torch.argmax(active_one_hot, dim=1),
            torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
        )
        debug.update(
            {
                "active_swing_leg": active_leg,
                "policy_q_ref": self.last_q_ref_policy,
                "simulator_q_ref": self.last_q_ref,
                "final_q_cmd": self.last_q_cmd,
                "filter_clipping_ratio": self.last_filter_clipping_ratio,
                "torque_clipping_ratio": self.last_torque_clipping_ratio,
                "predicted_foot_height": self.reference.last_predicted_foot_lift,
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
