from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _action(env, action_name: str = "joint_pos"):
    return env.action_manager.get_term(action_name)


def q_ref_tracking_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "joint_pos",
    deadzone: float = 0.04,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    error = torch.abs(_action(env, action_name).last_q_ref - asset.data.joint_pos[:, joint_ids])
    return torch.sum(torch.clamp(error - deadzone, min=0.0) ** 2, dim=1)


def residual_magnitude_penalty(env, action_name: str = "joint_pos") -> torch.Tensor:
    return torch.sum(_action(env, action_name).last_delta_q_rl**2, dim=1)


def residual_rate_penalty(env, action_name: str = "joint_pos") -> torch.Tensor:
    action = _action(env, action_name)
    previous = getattr(action, "_previous_residual_for_reward", None)
    if previous is None:
        action._previous_residual_for_reward = action.last_delta_q_rl.clone()
        return torch.zeros(env.num_envs, device=action.last_delta_q_rl.device)
    penalty = torch.sum((action.last_delta_q_rl - previous) ** 2, dim=1)
    previous.copy_(action.last_delta_q_rl)
    return penalty


def wave_swing_contact_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.norm(forces, dim=-1).max(dim=1)[0] > threshold
    action = _action(env, action_name)
    return torch.sum(contact * action.reference.last_swing_mask, dim=1) * action.reference.last_walk_gate


def wave_stance_contact_loss_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.norm(forces, dim=-1).max(dim=1)[0] > threshold
    action = _action(env, action_name)
    expected_stance = ~action.reference.last_swing_mask
    return torch.sum((~contact) * expected_stance, dim=1) * action.reference.last_walk_gate


def filter_tracking_error(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _action(env, action_name).last_filter_error
