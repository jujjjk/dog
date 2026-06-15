from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .curriculum_profiles import get_wave_stage_number


def _action(env, action_name: str = "joint_pos"):
    return env.action_manager.get_term(action_name)


def _stage_noise(env, value: torch.Tensor, noise_name: str) -> torch.Tensor:
    stage = get_wave_stage_number(int(getattr(env, "_fanfan_wave_stage", 1)))
    amplitude = float(stage["noise"][noise_name])
    if amplitude <= 0.0:
        return value
    return value + torch.empty_like(value).uniform_(-amplitude, amplitude)


def noisy_base_ang_vel(env) -> torch.Tensor:
    return _stage_noise(env, mdp.base_ang_vel(env), "base_ang_vel")


def noisy_projected_gravity(env) -> torch.Tensor:
    return _stage_noise(env, mdp.projected_gravity(env), "projected_gravity")


def reference_joint_pos(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _action(env, action_name).last_q_ref


def ordered_joint_pos_rel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def noisy_ordered_joint_pos_rel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _stage_noise(env, ordered_joint_pos_rel(env, asset_cfg), "joint_pos")


def ordered_joint_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def noisy_ordered_joint_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _stage_noise(env, ordered_joint_vel(env, asset_cfg), "joint_vel")


def reference_joint_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "joint_pos",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    return _action(env, action_name).last_q_ref - asset.data.joint_pos[:, joint_ids]


def reference_phase_features(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _action(env, action_name).reference.get_phase_features()


def active_swing_leg(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _action(env, action_name).reference.last_active_swing_one_hot


def last_residual_action(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _action(env, action_name).last_delta_q_rl


def normalized_foot_contact_forces(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces, dim=-1).max(dim=1)[0]
    return force_norm / torch.clamp(force_norm.sum(dim=1, keepdim=True), min=1.0)
