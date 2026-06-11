from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class DeployFilteredJointPositionAction(JointPositionAction):
    """Approximate the real deployment target limiter in training.

    The RS01 can peak high briefly, but real walking should not depend on
    permanent 17 N*m torque.  This filter teaches PPO that position targets are
    rate/accel/error limited before reaching the motor controller.
    """

    cfg: DeployFilteredJointPositionActionCfg

    def __init__(self, cfg: DeployFilteredJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._q_last_cmd = torch.zeros_like(self.processed_actions)
        self._qdot_last_cmd = torch.zeros_like(self.processed_actions)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        max_delay = max(1, int(self.cfg.sim_motor_delay_steps_range[1]))
        self._delay_buffer = torch.zeros(self.num_envs, max_delay, self.action_dim, device=self.device)
        self._deploy_q_raw = torch.zeros_like(self.processed_actions)
        self.last_q_cmd = torch.zeros_like(self.processed_actions)
        self.last_qdot_cmd = torch.zeros_like(self.processed_actions)
        self._err_limit_mul = self._make_joint_type_mul(
            hip=float(self.cfg.hip_err_limit_mul),
            thigh=float(self.cfg.thigh_err_limit_mul),
            calf=float(self.cfg.calf_err_limit_mul),
        )
        self._target_rate_mul = self._make_joint_type_mul(
            hip=float(self.cfg.hip_target_rate_mul),
            thigh=float(self.cfg.thigh_target_rate_mul),
            calf=float(self.cfg.calf_target_rate_mul),
        )
        self._target_accel_mul = self._make_joint_type_mul(
            hip=float(self.cfg.hip_target_accel_mul),
            thigh=float(self.cfg.thigh_target_accel_mul),
            calf=float(self.cfg.calf_target_accel_mul),
        )

        self._sample_filter_params(torch.arange(self.num_envs, device=self.device))

    def _uniform(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        lo, hi = float(value_range[0]), float(value_range[1])
        return torch.empty(shape, device=self.device).uniform_(lo, hi)

    def _make_joint_type_mul(self, hip: float, thigh: float, calf: float) -> torch.Tensor:
        values = []
        for name in self._joint_names:
            if "_hip_joint" in name:
                values.append(hip)
            elif "_thigh_joint" in name:
                values.append(thigh)
            elif "_calf_joint" in name:
                values.append(calf)
            else:
                values.append(1.0)
        return torch.tensor(values, device=self.device, dtype=self.processed_actions.dtype).unsqueeze(0)

    def _sample_filter_params(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        if n == 0:
            return
        shape = (n, 1)
        self._target_rate_limit = getattr(
            self, "_target_rate_limit", torch.zeros(self.num_envs, 1, device=self.device)
        )
        self._target_accel_limit = getattr(
            self, "_target_accel_limit", torch.zeros(self.num_envs, 1, device=self.device)
        )
        self._torque_budget = getattr(self, "_torque_budget", torch.zeros(self.num_envs, 1, device=self.device))
        self._kp_scale = getattr(self, "_kp_scale", torch.ones(self.num_envs, 1, device=self.device))
        self._kd_scale = getattr(self, "_kd_scale", torch.ones(self.num_envs, 1, device=self.device))
        self._motor_strength = getattr(self, "_motor_strength", torch.ones(self.num_envs, 1, device=self.device))
        self._motor_delay_steps = getattr(
            self, "_motor_delay_steps", torch.ones(self.num_envs, 1, dtype=torch.long, device=self.device)
        )

        budget = self._uniform(self.cfg.sim_torque_budget_range, shape)
        peak_mask = torch.rand(shape, device=self.device) < float(self.cfg.sim_short_peak_prob)
        peak_budget = self._uniform(self.cfg.sim_short_peak_torque_range, shape)
        self._torque_budget[env_ids] = torch.where(peak_mask, peak_budget, budget)
        self._target_rate_limit[env_ids] = self._uniform(self.cfg.sim_target_rate_limit_range, shape)
        self._target_accel_limit[env_ids] = self._uniform(self.cfg.sim_target_accel_limit_range, shape)
        self._kp_scale[env_ids] = self._uniform(self.cfg.sim_kp_scale_range, shape)
        self._kd_scale[env_ids] = self._uniform(self.cfg.sim_kd_scale_range, shape)
        self._motor_strength[env_ids] = self._uniform(self.cfg.sim_motor_strength_scale_range, shape)

        d0, d1 = int(self.cfg.sim_motor_delay_steps_range[0]), int(self.cfg.sim_motor_delay_steps_range[1])
        self._motor_delay_steps[env_ids] = torch.randint(max(1, d0), max(1, d1) + 1, shape, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._deploy_q_raw[:] = self.processed_actions
        if not self.cfg.enable_deploy_target_filter:
            self.last_q_cmd[:] = self.processed_actions
            self.last_qdot_cmd.zero_()
            return

        q_raw = self.processed_actions
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
        qdot_raw = torch.clamp(
            (q_safe - self._q_last_cmd) / dt,
            min=-rate_limit,
            max=rate_limit,
        )
        qdot_delta = torch.clamp(
            qdot_raw - self._qdot_last_cmd,
            min=-accel_limit * dt,
            max=accel_limit * dt,
        )
        qdot_cmd = self._qdot_last_cmd + qdot_delta
        q_cmd = self._q_last_cmd + qdot_cmd * dt
        q_cmd = q_current + torch.clamp(q_cmd - q_current, min=-err_limit, max=err_limit)

        self._q_last_cmd[:] = q_cmd
        self._qdot_last_cmd[:] = qdot_cmd
        self.last_q_cmd[:] = q_cmd
        self.last_qdot_cmd[:] = qdot_cmd

        self._delay_buffer = torch.roll(self._delay_buffer, shifts=1, dims=1)
        self._delay_buffer[:, 0] = q_cmd
        delay_idx = torch.clamp(self._motor_delay_steps.squeeze(-1) - 1, min=0, max=self._delay_buffer.shape[1] - 1)
        self._processed_actions = self._delay_buffer[torch.arange(self.num_envs, device=self.device), delay_idx]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._raw_actions[env_ids_tensor] = 0.0
        q_current = self._asset.data.joint_pos[env_ids_tensor][:, self._joint_ids]
        self._q_last_cmd[env_ids_tensor] = q_current
        self._qdot_last_cmd[env_ids_tensor] = 0.0
        self._delay_buffer[env_ids_tensor] = q_current.unsqueeze(1)
        self.last_q_cmd[env_ids_tensor] = q_current
        self.last_qdot_cmd[env_ids_tensor] = 0.0
        self._initialized[env_ids_tensor] = False
        self._sample_filter_params(env_ids_tensor)


@configclass
class DeployFilteredJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with a deploy-like target filter before PD control."""

    class_type: type = DeployFilteredJointPositionAction

    enable_deploy_target_filter: bool = True
    sim_target_rate_limit_range: tuple[float, float] = (2.0, 3.0)
    sim_target_accel_limit_range: tuple[float, float] = (60.0, 120.0)
    sim_torque_budget_range: tuple[float, float] = (5.0, 10.0)
    sim_short_peak_torque_range: tuple[float, float] = (10.0, 14.0)
    sim_short_peak_prob: float = 0.05
    sim_motor_delay_steps_range: tuple[int, int] = (1, 3)
    sim_motor_strength_scale_range: tuple[float, float] = (0.7, 1.2)
    sim_kp: float = 40.0
    sim_kp_scale_range: tuple[float, float] = (0.8, 1.2)
    sim_kd_scale_range: tuple[float, float] = (0.7, 1.3)
    hip_err_limit_mul: float = 1.0
    thigh_err_limit_mul: float = 1.2
    calf_err_limit_mul: float = 1.4
    hip_target_rate_mul: float = 1.0
    thigh_target_rate_mul: float = 1.3
    calf_target_rate_mul: float = 1.6
    hip_target_accel_mul: float = 1.0
    thigh_target_accel_mul: float = 1.3
    calf_target_accel_mul: float = 1.6
