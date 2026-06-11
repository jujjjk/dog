from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import mdp


WAVE_CURRICULUM_STAGES = (
    {"stage": 1, "start_iter": 0, "end_iter": 10_000, "lin_vel_x": (0.00, 0.05), "standing": 0.35,
     "swing_contact": -0.10, "stance_loss": -0.05},
    {"stage": 2, "start_iter": 10_000, "end_iter": 30_000, "lin_vel_x": (0.03, 0.10), "standing": 0.20,
     "swing_contact": -0.10, "stance_loss": -0.05},
    {"stage": 3, "start_iter": 30_000, "end_iter": 60_000, "lin_vel_x": (0.05, 0.15), "standing": 0.10,
     "swing_contact": -0.30, "stance_loss": -0.15},
    {"stage": 4, "start_iter": 60_000, "end_iter": None, "lin_vel_x": (0.05, 0.15), "standing": 0.05,
     "swing_contact": -0.60, "stance_loss": -0.30},
)


def get_wave_stage(iteration: int, stages: Sequence[dict] = WAVE_CURRICULUM_STAGES) -> dict:
    for stage in stages:
        if iteration >= stage["start_iter"] and (stage["end_iter"] is None or iteration < stage["end_iter"]):
            return stage
    return stages[-1]


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
    env._fanfan_wave_stage = int(stage["stage"])
    return {"iteration": float(iteration), "stage": float(stage["stage"])}


def stage_gated_push(env, env_ids, velocity_range, minimum_stage: int = 4):
    if int(getattr(env, "_fanfan_wave_stage", 1)) >= minimum_stage:
        mdp.push_by_setting_velocity(env, env_ids, velocity_range)
