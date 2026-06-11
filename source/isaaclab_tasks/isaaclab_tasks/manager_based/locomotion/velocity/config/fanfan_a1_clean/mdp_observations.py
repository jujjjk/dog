from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def base_lin_vel_deploy_corrupted(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    enable_randomization: bool = True,
    zero_prob: float = 0.15,
    command_prob: float = 0.20,
    noise_std: tuple[float, float, float] = (0.08, 0.05, 0.02),
    bias_range: tuple[float, float] = (-0.05, 0.05),
    delay_steps_range: tuple[int, int] = (0, 3),
    scale_range: tuple[float, float] = (0.7, 1.2),
) -> torch.Tensor:
    """Base linear velocity observation that mimics deployment uncertainty.

    The real robot can run with zero, command, or estimator velocity sources.
    Training should therefore not let the policy depend on perfect base velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_vel = asset.data.root_lin_vel_b[:, :3]
    if not enable_randomization:
        return base_vel

    num_envs = base_vel.shape[0]
    max_delay = max(0, int(delay_steps_range[1]))
    hist_name = "_fanfan_base_lin_vel_history"
    hist = getattr(env, hist_name, None)
    if hist is None or hist.shape != (num_envs, max_delay + 1, 3) or hist.device != base_vel.device:
        hist = base_vel.unsqueeze(1).repeat(1, max_delay + 1, 1)
    hist = torch.roll(hist, shifts=1, dims=1)
    hist[:, 0] = base_vel
    setattr(env, hist_name, hist)

    d0, d1 = int(delay_steps_range[0]), int(delay_steps_range[1])
    delay = torch.randint(max(0, d0), max(0, d1) + 1, (num_envs,), device=base_vel.device)
    delayed_vel = hist[torch.arange(num_envs, device=base_vel.device), delay]

    cmd = env.command_manager.get_command(command_name)
    cmd_vel = torch.zeros_like(delayed_vel)
    cmd_vel[:, 0] = cmd[:, 0]
    cmd_vel[:, 1] = cmd[:, 1]

    mode = torch.rand(num_envs, 1, device=base_vel.device)
    obs = delayed_vel
    obs = torch.where(mode < float(zero_prob), torch.zeros_like(obs), obs)
    obs = torch.where(
        (mode >= float(zero_prob)) & (mode < float(zero_prob + command_prob)),
        cmd_vel,
        obs,
    )

    scale = torch.empty(num_envs, 1, device=base_vel.device).uniform_(float(scale_range[0]), float(scale_range[1]))
    bias = torch.empty_like(obs).uniform_(float(bias_range[0]), float(bias_range[1]))
    noise_std_tensor = torch.tensor(noise_std, device=base_vel.device, dtype=base_vel.dtype).unsqueeze(0)
    noise = torch.randn_like(obs) * noise_std_tensor
    return obs * scale + bias + noise
