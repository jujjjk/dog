from __future__ import annotations

import torch

from isaaclab.actuators.actuator_net import ActuatorNetMLP
from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FanfanSafeActuatorNetMLP(ActuatorNetMLP):
    """带输入/输出保护和目标角平滑的 Fanfan actuator net。

    输入顺序保持 IsaacLab 原生实现：
    pos_vel = [qerr_t, qerr_t-1, qerr_t-2, dq_t, dq_t-1, dq_t-2]。
    """

    cfg: FanfanSafeActuatorNetMLPCfg

    def __init__(self, cfg: FanfanSafeActuatorNetMLPCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._target_smooth_enabled = bool(self.cfg.target_smooth_enable)
        self._last_q_cmd: torch.Tensor | None = None

        # 调试缓存：后续 reward 或 debug 打印可以直接读取这些属性。
        self.last_q_raw: torch.Tensor | None = None
        self.last_q_cmd: torch.Tensor | None = None
        self.last_target_error: torch.Tensor | None = None
        self.last_target_delta: torch.Tensor | None = None
        self.max_abs_target_error = torch.tensor(0.0, device=self._device)
        self.mean_abs_target_error = torch.tensor(0.0, device=self._device)
        self.max_abs_target_delta = torch.tensor(0.0, device=self._device)
        self.mean_abs_target_delta = torch.tensor(0.0, device=self._device)
        self.target_smooth_enabled = torch.tensor(float(self._target_smooth_enabled), device=self._device)

    def reset(self, env_ids):
        """重置目标角缓存，避免新 episode 继承旧目标角。"""
        super().reset(env_ids)
        if self._last_q_cmd is None:
            return
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            self._last_q_cmd[:] = torch.nan
        else:
            self._last_q_cmd[env_ids] = torch.nan

    def smooth_joint_targets(self, q_raw: torch.Tensor, q_current: torch.Tensor, env_ids=None) -> torch.Tensor:
        """把 policy 原始目标角拆成更小、更平滑的 actuator 目标角。"""
        self.last_q_raw = q_raw.detach().clone()
        self.target_smooth_enabled = torch.tensor(float(self._target_smooth_enabled), device=q_raw.device)

        if not self._target_smooth_enabled:
            self.last_q_cmd = q_raw.detach().clone()
            self.last_target_error = (q_raw - q_current).detach().clone()
            self.last_target_delta = torch.zeros_like(q_raw)
            self.max_abs_target_error = torch.max(torch.abs(self.last_target_error))
            self.mean_abs_target_error = torch.mean(torch.abs(self.last_target_error))
            self.max_abs_target_delta = torch.max(torch.abs(self.last_target_delta))
            self.mean_abs_target_delta = torch.mean(torch.abs(self.last_target_delta))
            return q_raw

        if self._last_q_cmd is None or self._last_q_cmd.shape != q_raw.shape:
            self._last_q_cmd = q_current.detach().clone()
        else:
            invalid_rows = ~torch.isfinite(self._last_q_cmd).all(dim=1)
            if torch.any(invalid_rows):
                self._last_q_cmd[invalid_rows] = q_current.detach()[invalid_rows]

        q_last = self._last_q_cmd
        qerr_limit = float(self.cfg.target_qerr_limit)
        max_delta = float(self.cfg.target_max_delta)
        alpha = float(self.cfg.target_lpf_alpha)

        q_safe = q_current + torch.clamp(q_raw - q_current, -qerr_limit, qerr_limit)
        delta = torch.clamp(q_safe - q_last, -max_delta, max_delta)
        q_limited = q_last + delta
        q_cmd = alpha * q_limited + (1.0 - alpha) * q_last

        self.last_q_cmd = q_cmd.detach().clone()
        self.last_target_error = (q_cmd - q_current).detach().clone()
        self.last_target_delta = (q_cmd - q_last).detach().clone()
        self.max_abs_target_error = torch.max(torch.abs(self.last_target_error))
        self.mean_abs_target_error = torch.mean(torch.abs(self.last_target_error))
        self.max_abs_target_delta = torch.max(torch.abs(self.last_target_delta))
        self.mean_abs_target_delta = torch.mean(torch.abs(self.last_target_delta))
        self._last_q_cmd = q_cmd.detach().clone()
        return q_cmd

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        q_raw = control_action.joint_positions
        q_cmd = self.smooth_joint_targets(q_raw, joint_pos)
        control_action.joint_positions = q_cmd

        # qerr 必须是 q_cmd - q_current，和采集训练脚本保持一致。
        self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
        self._joint_pos_error_history[:, 0] = q_cmd - joint_pos
        self._joint_vel_history = self._joint_vel_history.roll(1, 1)
        self._joint_vel_history[:, 0] = joint_vel
        self._joint_vel[:] = joint_vel

        pos_input = torch.cat([self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        pos_input = pos_input.view(self._num_envs * self.num_joints, -1)
        vel_input = torch.cat([self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        vel_input = vel_input.view(self._num_envs * self.num_joints, -1)

        # 原有 actuator-net 输入保护仍然保留。
        pos_input = torch.clamp(pos_input, -self.cfg.pos_error_clip, self.cfg.pos_error_clip)
        vel_input = torch.clamp(vel_input, -self.cfg.velocity_clip, self.cfg.velocity_clip)

        if self.cfg.input_order == "pos_vel":
            network_input = torch.cat([pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale], dim=1)
        elif self.cfg.input_order == "vel_pos":
            network_input = torch.cat([vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1)
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        with torch.inference_mode():
            net_effort = self.network(network_input).view(self._num_envs, self.num_joints) * self.cfg.torque_scale
        net_effort = torch.clamp(net_effort, -self.cfg.output_effort_clip, self.cfg.output_effort_clip)

        pd_effort = self.cfg.fallback_stiffness * (q_cmd - joint_pos)
        pd_effort -= self.cfg.fallback_damping * joint_vel
        pd_effort = torch.clamp(pd_effort, -self.cfg.output_effort_clip, self.cfg.output_effort_clip)

        blend = float(self.cfg.net_blend)
        self.computed_effort = (1.0 - blend) * pd_effort + blend * net_effort
        self.computed_effort = torch.clamp(
            self.computed_effort, -self.cfg.output_effort_clip, self.cfg.output_effort_clip
        )

        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class FanfanSafeActuatorNetMLPCfg(ActuatorNetMLPCfg):
    """Fanfan 专用安全 actuator net 配置。"""

    class_type: type = FanfanSafeActuatorNetMLP

    pos_error_clip: float = 0.25
    """送入 actuator net 前的 q_target - q_current 限幅，单位 rad。"""

    velocity_clip: float = 6.0
    """送入 actuator net 前的关节速度限幅，单位 rad/s。"""

    output_effort_clip: float = 6.0
    """actuator net 输出力矩限幅，单位 Nm。"""

    net_blend: float = 0.35
    """最终力矩中 actuator net 的比例；1.0 表示完全使用 actuator net。"""

    fallback_stiffness: float = 25.0
    """混合用保守 PD 的 kp。"""

    fallback_damping: float = 1.5
    """混合用保守 PD 的 kd。"""

    target_smooth_enable: bool = True
    """是否启用目标角拆分/平滑安全层。"""

    target_qerr_limit: float = 0.20
    """q_cmd 和 q_current 的最大瞬时误差，单位 rad。"""

    target_max_delta: float = 0.05
    """每个 control step 目标角最大变化量，单位 rad。"""

    target_lpf_alpha: float = 0.30
    """目标角低通滤波系数。"""
