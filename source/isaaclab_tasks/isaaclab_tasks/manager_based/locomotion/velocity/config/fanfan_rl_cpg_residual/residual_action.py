from __future__ import annotations

from collections.abc import Sequence
import os

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.deploy_actions import (
    DeployFilteredJointPositionAction,
    DeployFilteredJointPositionActionCfg,
)

from .joint_semantics import FanfanJointSemanticAdapter, FanfanJointSemanticCfg
from .curriculum_profiles import get_wave_stage_number
from .csv_playback import LoopingJointCsvPlayback, load_joint_csv
from .reference_gait import FanfanReferenceGait, FanfanReferenceGaitCfg
from .residual_math import clamp_joint_targets, filter_residual, joint_mapping_index


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
        self._residual_scale = self._make_residual_scale()
        self._filtered_residual = torch.zeros_like(self.processed_actions)
        self.last_q_ref_policy = self.reference.default_joint_pos.clone()
        self.last_q_ref = self.semantic_adapter.policy_to_sim(self.last_q_ref_policy)
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
        self._joint_mapping_step = 0
        self._joint_mapping_index = -1
        self._last_joint_limit_warning_step = -10**9
        self._csv_playback_time = torch.zeros(self.num_envs, device=self.device)
        self._csv_playback = None
        if cfg.action_mode == "csv_playback":
            csv_path = os.environ.get("FANFAN_CSV_PLAYBACK_PATH", cfg.csv_playback_path)
            times, values, value_space = load_joint_csv(csv_path)
            if value_space == "real":
                values = self.semantic_adapter.real_to_policy(values.to(self.device)).cpu()
            self._csv_playback = LoopingJointCsvPlayback(times, values, device=self.device)
            print(
                f"[FANFAN CSV PLAYBACK] loaded {values.shape[0]} frames from {csv_path}, "
                f"duration={self._csv_playback.duration:.3f}s, source={value_space}"
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
        else:
            q_ref_policy = self.reference.update(self._commands())

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
        q_ref_sim = self.semantic_adapter.policy_to_sim(q_ref_policy)
        q_raw = self.semantic_adapter.policy_to_sim(q_raw_policy)
        q_before_joint_limit = q_raw.clone()
        if self.cfg.clip is not None:
            q_raw = torch.clamp(q_raw, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._deploy_q_raw[:] = q_raw
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

        if not self.cfg.enable_deploy_target_filter:
            self._processed_actions[:] = q_raw
            self.last_q_cmd[:] = q_raw
            self.last_q_after_rate_limit[:] = q_raw
            self.last_q_after_accel_limit[:] = q_raw
            self.last_q_after_torque_clip[:] = q_raw
            self.last_q_before_delay[:] = q_raw
            self.last_q_after_delay[:] = q_raw
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
        kd_eff = max(float(self.cfg.sim_kd), 0.0) * self._kd_scale * self._motor_strength
        err_limit = (self._torque_budget / kp_eff) * self._err_limit_mul
        damping_scale = torch.sqrt(torch.clamp(self._kd_scale, min=0.5, max=2.0))
        rate_limit = (self._target_rate_limit / damping_scale) * self._target_rate_mul
        accel_limit = (self._target_accel_limit / damping_scale) * self._target_accel_mul

        qdot_desired = (q_raw - self._q_last_cmd) / dt
        if self.cfg.enable_target_rate_limit:
            qdot_rate = torch.clamp(qdot_desired, min=-rate_limit, max=rate_limit)
        else:
            qdot_rate = qdot_desired
        q_after_rate = self._q_last_cmd + qdot_rate * dt
        self.last_q_after_rate_limit[:] = q_after_rate
        self.last_rate_clip_mask[:] = torch.abs(qdot_rate - qdot_desired) > 1.0e-6
        self.last_rate_clipping_ratio[:] = torch.mean(self.last_rate_clip_mask.to(q_raw.dtype), dim=1)

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
        self.last_accel_clip_mask[:] = torch.abs(qdot_cmd - qdot_rate) > 1.0e-6
        self.last_accel_clipping_ratio[:] = torch.mean(self.last_accel_clip_mask.to(q_raw.dtype), dim=1)

        self.last_tau_est[:] = kp_eff * (q_after_accel - q_current) - kd_eff * self._asset.data.joint_vel[
            :, self._joint_ids
        ]
        if self.cfg.enable_torque_target_limit:
            q_after_torque = q_current + torch.clamp(
                q_after_accel - q_current, min=-err_limit, max=err_limit
            )
        else:
            q_after_torque = q_after_accel
        self.last_q_after_torque_clip[:] = q_after_torque
        self.last_torque_clip_mask[:] = torch.abs(q_after_torque - q_after_accel) > 1.0e-6
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
        self.last_q_ref_policy[env_ids] = self.reference.default_joint_pos[env_ids]
        self.last_q_ref[env_ids] = self.semantic_adapter.policy_to_sim(
            self.reference.default_joint_pos[env_ids]
        )
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
        if env_ids.numel() == self.num_envs:
            self._joint_mapping_step = 0
            self._joint_mapping_index = -1
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
                "joint_mapping_index": torch.full(
                    (self.num_envs,), self._joint_mapping_index, device=self.device, dtype=torch.long
                ),
                "stance_mask": ~self.reference.last_swing_mask,
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
                "tau_est_max": torch.max(torch.abs(self.last_tau_est), dim=1).values,
                "tau_est_mean": torch.mean(torch.abs(self.last_tau_est), dim=1),
                "over_6nm_ratio": torch.mean((torch.abs(self.last_tau_est) > 6.0).to(self.last_tau_est.dtype), dim=1),
                "over_8nm_ratio": torch.mean((torch.abs(self.last_tau_est) > 8.0).to(self.last_tau_est.dtype), dim=1),
                "over_10nm_ratio": torch.mean((torch.abs(self.last_tau_est) > 10.0).to(self.last_tau_est.dtype), dim=1),
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
    joint_limit_warning_interval_sec: float = 1.0
    csv_playback_path: str = "logs/reference_debug/fanfan_gait_playback.csv"
