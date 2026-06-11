from __future__ import annotations

from collections.abc import Sequence


AUTO_SPEED_CURRICULUM_STAGES = (
    {
        "stage": 1,
        "start_iter": 0,
        "end_iter": 10_000,
        "lin_vel_x": (0.0, 0.08),
        "rel_standing_envs": 0.50,
    },
    {
        "stage": 2,
        "start_iter": 10_000,
        "end_iter": 30_000,
        "lin_vel_x": (0.0, 0.15),
        "rel_standing_envs": 0.35,
    },
    {
        "stage": 3,
        "start_iter": 30_000,
        "end_iter": 60_000,
        "lin_vel_x": (0.03, 0.25),
        "rel_standing_envs": 0.25,
    },
    {
        "stage": 4,
        "start_iter": 60_000,
        "end_iter": None,
        "lin_vel_x": (0.05, 0.35),
        "rel_standing_envs": 0.15,
    },
)


def get_speed_curriculum_stage(iteration: int, stages: Sequence[dict] | None = None) -> dict:
    """Return the automatic speed curriculum stage for a learner iteration."""
    stages = AUTO_SPEED_CURRICULUM_STAGES if stages is None else stages
    iteration = max(0, int(iteration))
    for stage in stages:
        start_iter = int(stage["start_iter"])
        end_iter = stage["end_iter"]
        if iteration >= start_iter and (end_iter is None or iteration < int(end_iter)):
            return stage
    return stages[-1]


def auto_speed_curriculum(
    env,
    env_ids,
    command_name: str = "base_velocity",
    enabled: bool = True,
    num_steps_per_iter: int = 24,
    stages: Sequence[dict] | None = None,
    print_on_stage_change: bool = True,
) -> dict[str, float] | None:
    """Update the velocity command distribution from low-speed standing to walking.

    IsaacLab calls curriculum terms at environment reset.  RSL-RL advances one
    learner iteration after ``num_steps_per_iter`` environment steps, so the
    current iteration is approximated from ``env.common_step_counter``.  The
    returned state is logged by the curriculum manager.
    """
    if not enabled:
        return None

    num_steps_per_iter = max(1, int(num_steps_per_iter))
    iteration = int(getattr(env, "common_step_counter", 0)) // num_steps_per_iter
    stage = get_speed_curriculum_stage(iteration, stages)

    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    command_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_term.cfg.ranges.heading = (0.0, 0.0)
    command_term.cfg.rel_standing_envs = float(stage["rel_standing_envs"])
    command_term.cfg.rel_heading_envs = 0.0
    command_term.cfg.heading_command = False

    stage_id = int(stage["stage"])
    previous_stage = getattr(env, "_fanfan_auto_speed_stage", None)
    if print_on_stage_change and previous_stage != stage_id:
        env._fanfan_auto_speed_stage = stage_id
        print(
            "[AUTO_SPEED_CURRICULUM] "
            f"iter={iteration} stage={stage_id} "
            f"lin_vel_x={tuple(stage['lin_vel_x'])} "
            f"rel_standing_envs={float(stage['rel_standing_envs']):.2f}"
        )

    return {
        "iteration": float(iteration),
        "stage": float(stage_id),
        "lin_vel_x_min": float(stage["lin_vel_x"][0]),
        "lin_vel_x_max": float(stage["lin_vel_x"][1]),
        "rel_standing_envs": float(stage["rel_standing_envs"]),
    }
