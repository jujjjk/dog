# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    action_term = None
    robot = None
    debug_period_steps = None
    try:
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        if getattr(action_term.cfg, "action_mode", "") in {
            "reference_raw",
            "reference_stage",
            "joint_mapping_debug",
            "csv_playback",
        }:
            robot = env.unwrapped.scene["robot"]
            debug_period_steps = max(1, round(0.5 / float(env.unwrapped.step_dt)))
            print(
                "[FANFAN DEAD GAIT PLAYER] zero_agent action is ignored; "
                f"action_mode={action_term.cfg.action_mode} generates joint targets internally."
            )
            print(
                f"[FANFAN DEAD GAIT PLAYER] control_dt={env.unwrapped.step_dt:.6f}s "
                f"physics_dt={env.unwrapped.cfg.sim.dt:.6f}s "
                f"decimation={env.unwrapped.cfg.decimation}"
            )
    except (AttributeError, KeyError, ValueError):
        action_term = None

    step = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            if action_term is not None and step % debug_period_steps == 0:
                debug = action_term.get_debug_info()
                active_index = int(debug["active_swing_leg"][0])
                active_name = ("FR", "FL", "RR", "RL")[active_index] if 0 <= active_index < 4 else "STANCE"
                roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
                joint_pos = robot.data.joint_pos[:, action_term._joint_ids]
                max_error = torch.max(torch.abs(debug["simulator_q_ref"] - joint_pos))
                print(
                    f"[FANFAN DEAD GAIT PLAYER] t={step * env.unwrapped.step_dt:.2f}s "
                    f"stage={int(debug['control_stage'][0])} "
                    f"phase={float(action_term.reference.base_phase[0]):.3f} active={active_name} "
                    f"frequency={float(debug['frequency'][0]):.3f}Hz "
                    f"stride={float(debug['stride'][0]):.4f}m "
                    f"swing={float(debug['swing_height'][0]):.4f}m "
                    f"phase_step={float(debug['phase_increment_per_step'][0]):.5f} "
                    f"cycle={float(debug['phase_cycle_time'][0]):.3f}s "
                    f"max_error={float(max_error):.4f}rad "
                    f"tau_max={float(debug['tau_est_max'][0]):.2f}Nm "
                    f"clips={float(debug['joint_limit_clipping_ratio'][0]):.2f}/"
                    f"{float(debug['rate_limit_clipping_ratio'][0]):.2f}/"
                    f"{float(debug['torque_clipping_ratio'][0]):.2f} "
                    f"rpy=({float(roll[0]):.3f},{float(pitch[0]):.3f},{float(yaw[0]):.3f})"
                )
            step += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
