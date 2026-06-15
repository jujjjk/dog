from __future__ import annotations

from isaaclab.envs import mdp

from .curriculum_profiles import WAVE_CURRICULUM_STAGES, get_wave_stage, reference_scales


def _set_event_params(env, term_name: str, **params) -> None:
    cfg = env.event_manager.get_term_cfg(term_name)
    cfg.params.update(params)
    env.event_manager.set_term_cfg(term_name, cfg)


def wave_curriculum(env, env_ids, command_name="base_velocity", num_steps_per_iter=24, stages=None):
    iteration = int(getattr(env, "common_step_counter", 0)) // max(1, int(num_steps_per_iter))
    stage = get_wave_stage(iteration, WAVE_CURRICULUM_STAGES if stages is None else stages)
    command = env.command_manager.get_term(command_name)
    command.cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    command.cfg.ranges.lin_vel_y = (0.0, 0.0)
    command.cfg.ranges.ang_vel_z = (0.0, 0.0)
    command.cfg.rel_standing_envs = float(stage["standing"])
    for name, key in (("wave_swing_contact", "swing_contact"), ("wave_stance_loss", "stance_loss")):
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = float(stage[key])
        env.reward_manager.set_term_cfg(name, cfg)

    _set_event_params(env, "add_base_mass", mass_distribution_params=tuple(stage["mass_delta"]))
    _set_event_params(
        env,
        "rs01_joint_properties",
        friction_distribution_params=tuple(stage["joint_friction"]),
        armature_distribution_params=tuple(stage["armature"]),
    )
    _set_event_params(
        env,
        "rs01_actuator_gains",
        stiffness_distribution_params=tuple(stage["actuator_gain"]),
        damping_distribution_params=tuple(stage["actuator_gain"]),
    )
    tilt = float(stage["reset_tilt"])
    _set_event_params(
        env,
        "reset_base",
        pose_range={
            **env.event_manager.get_term_cfg("reset_base").params["pose_range"],
            "roll": (-tilt, tilt),
            "pitch": (-tilt, tilt),
        },
    )
    env._fanfan_wave_stage = int(stage["stage"])
    stride_min, frequency_min, swing_min = reference_scales(stage["lin_vel_x"][0])
    stride_max, frequency_max, swing_max = reference_scales(stage["lin_vel_x"][1])
    return {
        "iteration": float(iteration),
        "stage": float(stage["stage"]),
        "cmd_x_min": float(stage["lin_vel_x"][0]),
        "cmd_x_max": float(stage["lin_vel_x"][1]),
        "standing_ratio": float(stage["standing"]),
        "reference_stride_scale_min": stride_min,
        "reference_stride_scale_max": stride_max,
        "reference_frequency_scale_min": frequency_min,
        "reference_frequency_scale_max": frequency_max,
        "reference_swing_scale_min": swing_min,
        "reference_swing_scale_max": swing_max,
        "mass_delta_max": max(abs(float(v)) for v in stage["mass_delta"]),
        "friction_min": float(stage["joint_friction"][0]),
        "friction_max": float(stage["joint_friction"][1]),
        "motor_strength_min": float(stage["motor_strength"][0]),
        "motor_strength_max": float(stage["motor_strength"][1]),
        "delay_max": float(stage["delay_steps"][1]),
        "noise_level": float(stage["noise_level"]),
        "push_enabled": float(stage["push_enabled"]),
    }


def stage_gated_push(env, env_ids, velocity_range, minimum_stage: int = 4):
    if int(getattr(env, "_fanfan_wave_stage", 1)) >= minimum_stage:
        mdp.push_by_setting_velocity(env, env_ids, velocity_range)
