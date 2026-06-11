from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _joint_target_for_rewards(
    env: ManagerBasedRLEnv,
    action_name: str,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return smoothed actuator targets when available, otherwise raw action targets."""
    action_term = env.action_manager.get_term(action_name)
    joint_target = action_term.processed_actions

    action_q_cmd = getattr(action_term, "last_q_cmd", None)
    if action_q_cmd is not None:
        joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
        return action_q_cmd[:, joint_ids]

    smoothed_target = None
    for actuator in asset.actuators.values():
        q_cmd = getattr(actuator, "last_q_cmd", None)
        if q_cmd is None:
            continue
        if smoothed_target is None:
            smoothed_target = joint_target.clone()
        smoothed_target[:, actuator.joint_indices] = q_cmd.to(device=joint_target.device, dtype=joint_target.dtype)

    if smoothed_target is not None:
        joint_target = smoothed_target

    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    return joint_target[:, joint_ids]


def action_saturation_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    threshold: float = 0.75,
) -> torch.Tensor:
    """Softly penalize raw policy actions before they hit hard clipping."""
    action_term = env.action_manager.get_term(action_name)
    raw = action_term.raw_actions
    sat = torch.clamp(torch.abs(raw) - threshold, min=0.0)
    return torch.sum(torch.square(sat), dim=1)


def residual_magnitude_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    delta = getattr(action_term, "last_delta_q_rl", None)
    if delta is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(delta), dim=1)


def residual_rate_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    delta = getattr(action_term, "last_delta_q_rl", None)
    if delta is None:
        return torch.zeros(env.num_envs, device=env.device)
    prev_name = "_fanfan_prev_delta_q_rl"
    prev = getattr(env, prev_name, torch.zeros_like(delta))
    penalty = torch.sum(torch.square(delta - prev), dim=1)
    setattr(env, prev_name, delta.detach().clone())
    return penalty


def action_acceleration_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    action = action_term.raw_actions
    prev1_name = "_fanfan_prev_action_1"
    prev2_name = "_fanfan_prev_action_2"
    prev1 = getattr(env, prev1_name, torch.zeros_like(action))
    prev2 = getattr(env, prev2_name, torch.zeros_like(action))
    accel = action - 2.0 * prev1 + prev2
    setattr(env, prev2_name, prev1.detach().clone())
    setattr(env, prev1_name, action.detach().clone())
    return torch.sum(torch.square(accel), dim=1)


def torque_rate_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    torque = asset.data.applied_torque[:, joint_ids]
    prev_name = "_fanfan_prev_applied_torque"
    prev = getattr(env, prev_name, torch.zeros_like(torque))
    penalty = torch.sum(torch.square(torque - prev), dim=1)
    setattr(env, prev_name, torque.detach().clone())
    return penalty


def soft_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    soft_torque_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    torque = asset.data.applied_torque[:, joint_ids]
    return torch.sum(torch.square(torch.clamp(torch.abs(torque) - soft_torque_limit, min=0.0)), dim=1)


def per_leg_residual_balance_penalty(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    delta = getattr(action_term, "last_delta_q_rl", None)
    if delta is None:
        delta = action_term.raw_actions
    leg_rms = torch.stack(
        [torch.mean(torch.square(delta[:, i : i + 3]), dim=1) for i in (0, 3, 6, 9)],
        dim=1,
    )
    return torch.var(leg_rms, dim=1)


def hip_residual_saturation_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    threshold_ratio: float = 0.85,
) -> torch.Tensor:
    """Diagnostic/stability penalty for hip residuals that live near their limit.

    This keeps long training runs from solving balance by permanently pinning
    the hip residual channel at its configured bound.
    """
    action_term = env.action_manager.get_term(action_name)
    delta = getattr(action_term, "last_delta_q_rl", None)
    cpg = getattr(action_term, "cpg", None)
    if delta is None or cpg is None:
        return torch.zeros(env.num_envs, device=env.device)
    hip_ids = torch.as_tensor((0, 3, 6, 9), device=delta.device)
    hip_delta = torch.abs(delta[:, hip_ids])
    hip_limits = cpg.residual_limits[:, hip_ids].to(device=delta.device, dtype=delta.dtype)
    threshold = float(threshold_ratio) * hip_limits
    return torch.sum(torch.square(torch.clamp(hip_delta - threshold, min=0.0)), dim=1)


def hip_filter_tracking_error_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    threshold: float = 0.015,
) -> torch.Tensor:
    """Diagnostic/stability penalty when the deploy-like filter removes hip motion.

    It compares raw CPG+residual hip targets against the filtered command so
    TensorBoard can reveal whether the target limiter is eating the balance
    correction.
    """
    action_term = env.action_manager.get_term(action_name)
    q_raw = getattr(action_term, "last_q_raw_cpg", None)
    q_cmd = getattr(action_term, "last_q_cmd", None)
    if q_raw is None or q_cmd is None:
        return torch.zeros(env.num_envs, device=env.device)
    hip_ids = torch.as_tensor((0, 3, 6, 9), device=q_raw.device)
    error = torch.abs(q_raw[:, hip_ids] - q_cmd[:, hip_ids])
    return torch.sum(torch.square(torch.clamp(error - threshold, min=0.0)), dim=1)


def hip_motion_diagnostic(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    source: str = "q_cmd",
    subtract_default: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a zero/small-weight hip motion diagnostic for long-run logging.

    Use this as a TensorBoard probe to confirm that CPG or filtered hip targets
    are actually moving without making it a meaningful optimization objective.
    """
    action_term = env.action_manager.get_term(action_name)
    q = getattr(action_term, "last_q_cmd", None) if source == "q_cmd" else getattr(action_term, "last_q_cpg", None)
    if q is None:
        return torch.zeros(env.num_envs, device=env.device)
    hip_ids = torch.as_tensor((0, 3, 6, 9), device=q.device)
    hip_q = q[:, hip_ids]
    if subtract_default:
        asset: Articulation = env.scene[asset_cfg.name]
        default_q = asset.data.default_joint_pos[:, asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)]
        hip_q = hip_q - default_q[:, hip_ids]
    return torch.sqrt(torch.mean(torch.square(hip_q), dim=1))


def joint_limit_margin_penalty(
    env: ManagerBasedRLEnv,
    margin: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    q = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.joint_pos_limits[:, joint_ids]
    lower_dist = q - limits[:, :, 0]
    upper_dist = limits[:, :, 1] - q
    near = torch.clamp(margin - torch.minimum(lower_dist, upper_dist), min=0.0)
    return torch.sum(torch.square(near), dim=1)


def cpg_phase_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    contact_threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    cpg = getattr(action_term, "cpg", None)
    if cpg is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
    phase01 = torch.remainder(cpg.last_leg_phase / (2.0 * math.pi), 1.0)
    swing = phase01 >= float(cpg.cfg.duty_factor)
    penalty = torch.sum((swing & in_contact).float(), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def phase_diagonal_support_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    contact_threshold: float = 1.0,
    stance_miss_cost: float = 0.5,
    swing_contact_cost: float = 0.5,
    transition_margin: float = 0.05,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Lightly penalize CPG phase/contact disagreement for diagonal trot support.

    The term is intentionally soft for long training: it ignores phase
    transitions and does not require a perfectly rigid two-feet-only contact
    pattern, but it discourages dragging all feet or losing the stance diagonal.
    """
    action_term = env.action_manager.get_term(action_name)
    cpg = getattr(action_term, "cpg", None)
    if cpg is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold

    phase01 = torch.remainder(cpg.last_leg_phase / (2.0 * math.pi), 1.0)
    swing_fraction = max(1.0 - float(cpg.cfg.duty_factor), 0.05)
    swing = phase01 < swing_fraction
    away_from_wrap = (phase01 > transition_margin) & (phase01 < 1.0 - transition_margin)
    away_from_switch = torch.abs(phase01 - swing_fraction) > transition_margin
    active = away_from_wrap & away_from_switch

    swing_bad = (swing & in_contact & active).float() * swing_contact_cost
    stance_bad = (~swing & ~in_contact & active).float() * stance_miss_cost
    penalty = torch.sum(swing_bad + stance_bad, dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def _cpg_diagonal_support_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str,
    transition_margin: float,
    command_name: str,
    command_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_term = env.action_manager.get_term(action_name)
    cpg = getattr(action_term, "cpg", None)
    if cpg is None:
        zero = torch.zeros(env.num_envs, device=env.device)
        return zero.bool(), zero.bool(), zero, zero

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.norm(foot_forces, dim=-1).max(dim=1)[0]
    force_proxy = force_norm / torch.clamp(torch.sum(force_norm, dim=1, keepdim=True), min=1.0e-6)
    diag_fr_rl_proxy = force_proxy[:, 0] + force_proxy[:, 3]
    diag_fl_rr_proxy = force_proxy[:, 1] + force_proxy[:, 2]

    phase01 = torch.remainder(cpg.last_leg_phase / (2.0 * math.pi), 1.0)
    swing_fraction = max(1.0 - float(cpg.cfg.duty_factor), 0.05)
    stance = phase01 >= swing_fraction
    away_from_wrap = (phase01 > transition_margin) & (phase01 < 1.0 - transition_margin)
    away_from_switch = torch.abs(phase01 - swing_fraction) > transition_margin
    active_phase = torch.all(away_from_wrap & away_from_switch, dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold

    fr_rl_stance_score = stance[:, 0].float() + stance[:, 3].float()
    fl_rr_stance_score = stance[:, 1].float() + stance[:, 2].float()
    expected_fr_rl = fr_rl_stance_score >= fl_rr_stance_score
    active = active_phase & moving
    return expected_fr_rl, active, diag_fr_rl_proxy, diag_fl_rr_proxy


def phase_diagonal_support_switch_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    margin: float = 0.05,
    transition_margin: float = 0.05,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Softly reward the expected trot diagonal for taking over load.

    This is a long-training stability term: it uses CPG phase and contact-force
    proxies to check whether FR+RL and FL+RR alternate support, without forcing
    a rigid two-foot contact pattern during transitions.
    """
    expected_fr_rl, active, diag_fr_rl_proxy, diag_fl_rr_proxy = _cpg_diagonal_support_state(
        env,
        sensor_cfg=sensor_cfg,
        action_name=action_name,
        transition_margin=transition_margin,
        command_name=command_name,
        command_threshold=command_threshold,
    )
    expected_proxy = torch.where(expected_fr_rl, diag_fr_rl_proxy, diag_fl_rr_proxy)
    other_proxy = torch.where(expected_fr_rl, diag_fl_rr_proxy, diag_fr_rl_proxy)
    return torch.clamp(float(margin) - (expected_proxy - other_proxy), min=0.0) * active.float()


def diagonal_support_accuracy_metric(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    expected_pair: str = "all",
    transition_margin: float = 0.05,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Zero-weight diagnostic for whether the expected diagonal wins load proxy.

    Set ``expected_pair`` to ``FR_RL`` or ``FL_RR`` to split TensorBoard curves
    and catch a one-sided support bias before hardware tests.
    """
    expected_fr_rl, active, diag_fr_rl_proxy, diag_fl_rr_proxy = _cpg_diagonal_support_state(
        env,
        sensor_cfg=sensor_cfg,
        action_name=action_name,
        transition_margin=transition_margin,
        command_name=command_name,
        command_threshold=command_threshold,
    )
    pair = str(expected_pair).strip().upper()
    if pair == "FR_RL":
        active = active & expected_fr_rl
        success = diag_fr_rl_proxy > diag_fl_rr_proxy
    elif pair == "FL_RR":
        active = active & ~expected_fr_rl
        success = diag_fl_rr_proxy > diag_fr_rl_proxy
    else:
        success = torch.where(expected_fr_rl, diag_fr_rl_proxy > diag_fl_rr_proxy, diag_fl_rr_proxy > diag_fr_rl_proxy)
    return success.float() * active.float()


def hip_gate_clamp_ratio_diagnostic(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Zero-weight diagnostic showing how often the phase-aware hip gate edits residuals."""
    action_term = env.action_manager.get_term(action_name)
    value = getattr(action_term, "last_hip_gate_clamp_ratio", None)
    if value is None:
        return torch.zeros(env.num_envs, device=env.device)
    return value


def base_roll_rate_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Lightly penalize large roll rate while preserving useful lateral weight shift."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll_rate = asset.data.root_ang_vel_b[:, 0]
    penalty = torch.square(roll_rate)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def excessive_foot_air_time_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that stay airborne for too long while the robot is commanded to move."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    penalty = torch.sum(torch.clamp(current_air_time - max_air_time, min=0.0), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize uneven swing/support timing across feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    current_air_time = torch.clamp(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], max=0.5)
    current_contact_time = torch.clamp(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], max=0.5)
    return torch.var(current_air_time, dim=1) + torch.var(current_contact_time, dim=1)


def moving_few_contacts_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contacts: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize moving with too few supporting feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    foot_force_norm = torch.norm(foot_forces, dim=-1)
    contacts = torch.max(foot_force_norm, dim=1)[0] > threshold
    contact_count = contacts.float().sum(dim=1)

    penalty = (contact_count < min_contacts).float()
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def moving_too_many_contacts_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contacts: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize dragging/shuffling with too many feet on the ground while moving.

    This term is intentionally simple: if the command asks the robot to move but
    all four feet stay in contact, the policy is probably avoiding swing phases
    instead of practicing a real gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    foot_force_norm = torch.norm(foot_forces, dim=-1)
    contacts = torch.max(foot_force_norm, dim=1)[0] > threshold
    contact_count = contacts.float().sum(dim=1)

    penalty = torch.clamp(contact_count - max_contacts, min=0.0)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def contact_foot_drag_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize foot horizontal velocity while the foot is in contact.

    This targets dragging/scraping: a foot can avoid airborne penalties by
    staying in contact, but if it slides horizontally under load it should be
    treated as poor foot-end trajectory quality.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
    foot_xy_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)

    penalty = torch.sum(foot_xy_vel * in_contact.float(), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def long_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_time: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that stay in stance too long while moving.

    This discourages a leg from becoming a passive crutch and helps reduce limp
    patterns where one rear foot never enters a clean swing phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    penalty = torch.sum(torch.clamp(contact_time - max_contact_time, min=0.0), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def debug_foot_height_contact_metric(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    metric: str,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Diagnostic foot metric for checking whether foot-link height is trustworthy.

    Use this as a zero-weight reward term.  Attach one foot per term so the
    logger can show FR/FL/RR/RL separately.  If the reward logger only displays
    weighted values, temporarily set the diagnostic term weight to a tiny value
    such as 1e-6 during manual debugging.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    if metric == "height":
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
        return torch.mean(foot_height, dim=1)

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
    if metric == "contact":
        return torch.mean(in_contact.float(), dim=1)

    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    if metric == "air_time":
        return torch.mean(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], dim=1)
    if metric == "contact_time":
        return torch.mean(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], dim=1)

    raise ValueError(f"Unsupported debug foot metric: {metric}")


def gait_phase_obs(env: ManagerBasedRLEnv, gait_period: float = 0.55) -> torch.Tensor:
    """Return sin/cos phase observation for a fixed low-speed trot cycle.

    The policy needs an explicit clock; otherwise it can only infer phase from
    past actions, which often collapses into symmetric but low-amplitude dragging.
    Shape: [num_envs, 2].
    """
    t = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    phase = torch.remainder(t / gait_period, 1.0)
    return torch.stack((torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)), dim=-1)


def phase_trot_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    gait_period: float = 0.55,
    swing_ratio: float = 0.45,
    base_clearance: float = 0.025,
    lift_height: float = 0.055,
    stance_contact_penalty: float = 0.025,
    contact_threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that miss a simple reference trot swing trajectory.

    This term is phase-based, so it does not wait for a foot to become airborne.
    FR/RL swing together, FL/RR swing half a cycle later.  During swing, the
    target foot height follows a small sine arc.  During stance, missing contact
    is only lightly penalized to avoid making the feet stick to the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    t = env.episode_length_buf.to(dtype=asset.data.body_pos_w.dtype) * env.step_dt
    base_phase = torch.remainder(t / gait_period, 1.0).unsqueeze(1)
    # Expected body order in the cfg is FR, FL, RR, RL.
    leg_offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=base_phase.device, dtype=base_phase.dtype).unsqueeze(0)
    leg_phase = torch.remainder(base_phase + leg_offsets, 1.0)
    swing_mask = leg_phase < swing_ratio

    swing_phase = torch.clamp(leg_phase / swing_ratio, min=0.0, max=1.0)
    desired_clearance = base_clearance + lift_height * torch.sin(math.pi * swing_phase)

    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    swing_penalty = torch.clamp(desired_clearance - foot_height, min=0.0) * swing_mask.float()

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
    stance_penalty = stance_contact_penalty * (~swing_mask & ~in_contact).float()

    penalty = torch.sum(swing_penalty + stance_penalty, dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def phase_trot_swing_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_period: float = 0.55,
    swing_ratio: float = 0.45,
    contact_threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize contact during the phase-defined swing window.

    This is intentionally phase-based rather than airborne-based: if the clock
    says a leg should swing but the foot remains on the ground, count it as
    dragging even if old air-time rewards are not active yet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    t = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    base_phase = torch.remainder(t / gait_period, 1.0).unsqueeze(1)
    # Expected order: FR, FL, RR, RL.  FR/RL swing together; FL/RR are half-cycle shifted.
    leg_offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=base_phase.device, dtype=base_phase.dtype).unsqueeze(0)
    leg_phase = torch.remainder(base_phase + leg_offsets, 1.0)
    swing_mask = leg_phase < swing_ratio

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
    penalty = torch.sum((swing_mask & in_contact).float(), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def phase_trot_calf_flexion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_calf_pos,
    gait_period: float = 0.55,
    swing_ratio: float = 0.45,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize insufficient calf folding during the phase-defined swing window."""
    asset: Articulation = env.scene[asset_cfg.name]

    t = env.episode_length_buf.to(dtype=asset.data.joint_pos.dtype) * env.step_dt
    base_phase = torch.remainder(t / gait_period, 1.0).unsqueeze(1)
    # Expected order: FR, FL, RR, RL.  Keep aligned with asset_cfg.joint_names.
    leg_offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=base_phase.device, dtype=base_phase.dtype).unsqueeze(0)
    leg_phase = torch.remainder(base_phase + leg_offsets, 1.0)
    swing_mask = leg_phase < swing_ratio

    calf_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target = torch.as_tensor(target_calf_pos, device=calf_pos.device, dtype=calf_pos.dtype)
    if target.ndim == 0:
        target = target.repeat(calf_pos.shape[1])
    target = target.unsqueeze(0)

    flexion_error = torch.clamp(calf_pos - target, min=0.0)
    penalty = torch.sum(flexion_error * swing_mask.float(), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def joint_target_tracking_error_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize large ``q_des - q`` errors that would demand high PD torque on hardware."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_target = _joint_target_for_rewards(env, action_name, asset, asset_cfg)

    error = torch.abs(joint_target - joint_pos)
    return torch.sum(torch.square(torch.clamp(error - threshold, min=0.0)), dim=1)


def estimated_pd_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    kp: float,
    kd: float,
    torque_limit: float,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize estimated real PD torque beyond a conservative motor limit.

    This mirrors deployment where commands are sent as position targets with fixed
    gains.  It is intentionally independent from the simulator actuator gains so
    the policy learns to avoid target errors that would trip the real driver.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    joint_target = _joint_target_for_rewards(env, action_name, asset, asset_cfg)

    torque_est = kp * (joint_target - joint_pos) - kd * joint_vel
    return torch.sum(torch.square(torch.clamp(torch.abs(torque_est) - torque_limit, min=0.0)), dim=1)


def applied_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    torque_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize actual actuator torque only when it approaches a hard limit.

    For the RS01 setup, the simulator hard limit can be the 17 N*m peak torque,
    while separate rewards should discourage long-term use above the 6 N*m
    continuous torque rating.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    applied_torque = asset.data.applied_torque[:, joint_ids]
    return torch.sum(torch.square(torch.clamp(torch.abs(applied_torque) - torque_limit, min=0.0)), dim=1)


def continuous_torque_penalty(
    env: ManagerBasedRLEnv,
    torque_reference: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize torque relative to the RS01 continuous rating.

    The RS01 can peak at 17 N*m, but the rated continuous load is 6 N*m.
    This term keeps policies from treating peak torque as a normal operating
    point while still allowing short high-torque transients.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    applied_torque = asset.data.applied_torque[:, joint_ids]
    normalized_torque = applied_torque / torque_reference
    return torch.mean(torch.square(normalized_torque), dim=1)


def low_speed_high_torque_penalty(
    env: ManagerBasedRLEnv,
    torque_reference: float,
    velocity_reference: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low-speed high-torque states that resemble motor stall.

    This is an engineering safety term for RS01 deployment: large torque near
    zero joint speed is more likely to heat the motor or trigger protection.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    applied_torque = asset.data.applied_torque[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    torque_ratio = torch.abs(applied_torque) / torque_reference
    low_speed_weight = torch.clamp(1.0 - torch.abs(joint_vel) / velocity_reference, min=0.0, max=1.0)
    return torch.mean(torch.square(torque_ratio) * low_speed_weight, dim=1)


def swing_foot_clearance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_clearance: float,
    contact_threshold: float = 1.0,
    min_air_time: float = 0.02,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize airborne feet that do not lift high enough while moving."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    foot_force_norm = torch.norm(foot_forces, dim=-1)
    in_contact = torch.max(foot_force_norm, dim=1)[0] > contact_threshold
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    swing_mask = torch.logical_and(~in_contact, current_air_time > min_air_time)

    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance_error = torch.clamp(target_clearance - foot_height, min=0.0)
    penalty = torch.sum(clearance_error * swing_mask.float(), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def swing_calf_flexion_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_calf_pos,
    contact_threshold: float = 1.0,
    min_air_time: float = 0.02,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize swing legs whose calf joints are not folded enough.

    真机开环数据已经说明电机基本能跟目标，拖地主要来自目标小腿收腿幅度不够。
    这个项直接约束摆动相的小腿角度，避免策略学成低幅度小碎步。
    ``sensor_cfg.body_names`` 和 ``asset_cfg.joint_names`` 必须使用同样的腿顺序。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    foot_force_norm = torch.norm(foot_forces, dim=-1)
    in_contact = torch.max(foot_force_norm, dim=1)[0] > contact_threshold
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    swing_mask = torch.logical_and(~in_contact, current_air_time > min_air_time)

    calf_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target = torch.as_tensor(target_calf_pos, device=calf_pos.device, dtype=calf_pos.dtype)
    if target.ndim == 0:
        target = target.repeat(calf_pos.shape[1])
    target = target.unsqueeze(0)

    # Fanfan 小腿关节为负向收腿。若摆动相实际角度比目标更“直”，就惩罚。
    flexion_error = torch.clamp(calf_pos - target, min=0.0)
    penalty = torch.sum(flexion_error * swing_mask.float(), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def rear_calf_fold_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.12,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["RR_calf_joint", "RL_calf_joint"]),
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize rear calves folding beyond default by ``threshold`` radians.

    When ``sensor_cfg`` is provided, this only applies while the rear foot is in
    contact.  That avoids fighting the swing-phase calf flexion needed to lift
    the foot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    fold = torch.clamp((q_default - threshold) - q, min=0.0)
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
        fold = fold * in_contact.float()
    penalty = torch.sum(torch.square(fold), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def rear_thigh_low_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.10,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["RR_thigh_joint", "RL_thigh_joint"]),
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize rear thighs dropping below default by ``threshold`` radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    low = torch.clamp((q_default - threshold) - q, min=0.0)
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
        low = low * in_contact.float()
    penalty = torch.sum(torch.square(low), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def joint_fold_below_default_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints moving below their default angle by a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    fold = torch.clamp((q_default - threshold) - q, min=0.0)
    penalty = torch.sum(torch.square(fold), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def front_rear_posture_balance_penalty(
    env: ManagerBasedRLEnv,
    front_asset_cfg: SceneEntityCfg,
    rear_asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Penalize front legs doing all the crouch/swing work while rear legs stay stiff.

    This compares joint displacement from each joint's own default pose, so it
    still allows the front and rear default stand angles to be different.
    """
    asset: Articulation = env.scene[front_asset_cfg.name]
    q_front = asset.data.joint_pos[:, front_asset_cfg.joint_ids]
    q_rear = asset.data.joint_pos[:, rear_asset_cfg.joint_ids]
    d_front = q_front - asset.data.default_joint_pos[:, front_asset_cfg.joint_ids]
    d_rear = q_rear - asset.data.default_joint_pos[:, rear_asset_cfg.joint_ids]

    front_mean = torch.mean(d_front, dim=1)
    rear_mean = torch.mean(d_rear, dim=1)
    penalty = torch.square(front_mean - rear_mean)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def phase_trot_contact_pattern_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_period: float = 0.55,
    swing_ratio: float = 0.45,
    contact_threshold: float = 1.0,
    stance_miss_cost: float = 0.5,
    swing_contact_cost: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Penalize feet that disagree with the expected diagonal trot contact phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    t = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    base_phase = torch.remainder(t / gait_period, 1.0).unsqueeze(1)
    # Expected order: FR, FL, RR, RL. FR/RL swing together; FL/RR shift by half cycle.
    leg_offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=base_phase.device, dtype=base_phase.dtype).unsqueeze(0)
    leg_phase = torch.remainder(base_phase + leg_offsets, 1.0)
    swing_mask = leg_phase < swing_ratio

    foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
    swing_bad = (swing_mask & in_contact).float() * swing_contact_cost
    stance_bad = (~swing_mask & ~in_contact).float() * stance_miss_cost
    penalty = torch.sum(swing_bad + stance_bad, dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def rear_leg_length_penalty(
    env: ManagerBasedRLEnv,
    thigh_threshold: float = 0.10,
    calf_threshold: float = 0.12,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["RR_thigh_joint", "RR_calf_joint", "RL_thigh_joint", "RL_calf_joint"],
    ),
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Approximate rear short-leg/crouch penalty from thigh and calf folding."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    thresholds = torch.tensor(
        [thigh_threshold, calf_threshold, thigh_threshold, calf_threshold],
        device=q.device,
        dtype=q.dtype,
    ).unsqueeze(0)
    fold = torch.clamp((q_default - thresholds) - q, min=0.0)
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        in_contact = torch.norm(foot_forces, dim=-1).max(dim=1)[0] > contact_threshold
        contact_mask = torch.stack((in_contact[:, 0], in_contact[:, 0], in_contact[:, 1], in_contact[:, 1]), dim=1)
        fold = fold * contact_mask.float()
    penalty = torch.sum(torch.square(fold), dim=1)
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return penalty


def rear_swing_foot_clearance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_clearance: float = 0.075,
    contact_threshold: float = 1.0,
    min_air_time: float = 0.025,
    command_name: str = "base_velocity",
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Stronger clearance term for rear swing feet."""
    return swing_foot_clearance_penalty(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        target_clearance=target_clearance,
        contact_threshold=contact_threshold,
        min_air_time=min_air_time,
        command_name=command_name,
        command_threshold=command_threshold,
    )


def power_penalty(
    env: ManagerBasedRLEnv,
    torque_reference: float,
    velocity_reference: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint mechanical power normalized by RS01 continuous capability."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    applied_torque = asset.data.applied_torque[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    power = torch.abs(applied_torque * joint_vel) / max(torque_reference * velocity_reference, 1.0e-6)
    return torch.mean(power, dim=1)


class GaitReward(ManagerTermBase):
    """Reward a trot-like quadruped gait using diagonal sync and side/front-back anti-sync."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two synchronized foot pairs.")
        self.synced_feet_pairs = [
            self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0],
            self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0],
        ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sync_reward = self._sync_reward(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward *= self._sync_reward(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])

        async_reward = self._async_reward(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward *= self._async_reward(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward *= self._async_reward(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward *= self._async_reward(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])

        cmd = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.1, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )

    def _sync_reward(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)
