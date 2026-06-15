"""Run and record deterministic Fanfan reference-gait debug tasks."""

import argparse
import csv
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record layered Fanfan reference-gait diagnostics.")
parser.add_argument(
    "--task",
    default="Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceRaw-v0",
    help="Registered Fanfan reference debug task.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds.")
parser.add_argument("--delay_steps", type=int, choices=(0, 1, 2), default=None)
parser.add_argument("--csv_path", type=str, default=None, help="CSV input for CsvPlayback-v0.")
parser.add_argument("--output", type=str, default=None, help="CSV path; defaults under logs/reference_debug.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg


LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)


def _vector_columns(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index}" for index in range(count)]


def _row_vector(tensor: torch.Tensor) -> list[float]:
    return tensor[0].detach().cpu().flatten().tolist()


def _scalar(tensor: torch.Tensor) -> float:
    return float(tensor[0].detach().cpu())


def main():
    if args_cli.csv_path:
        os.environ["FANFAN_CSV_PLAYBACK_PATH"] = str(Path(args_cli.csv_path).expanduser().resolve())
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.delay_steps is not None:
        env_cfg.actions.joint_pos.fixed_delay_steps = int(args_cli.delay_steps)
        env_cfg.actions.joint_pos.enable_action_delay = args_cli.delay_steps > 0

    mode = str(env_cfg.actions.joint_pos.action_mode)
    output_path = Path(args_cli.output or f"logs/reference_debug/{mode}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    action_term = base_env.action_manager.get_term("joint_pos")
    robot = base_env.scene["robot"]
    env.reset()

    header = [
        "time",
        "mode",
        "phase",
        "active_swing_leg",
        "joint_mapping_joint",
        "cmd_x",
        "stride",
        "frequency",
        "swing_height",
        "duty_factor",
        "warmup",
        "control_dt",
        "physics_dt",
        "decimation",
        "phase_increment_per_step",
        "phase_cycle_time",
        "delay_steps",
        "joint_limit_clip_ratio",
        "rate_limit_clip_ratio",
        "acceleration_clip_ratio",
        "torque_clip_ratio",
        "filter_clip_ratio",
        "max_abs_q_ref_minus_rate",
        "max_abs_q_ref_minus_torque",
        "max_abs_q_ref_minus_final",
        "tau_est_max",
        "tau_est_mean",
        "over_6nm_ratio",
        "over_8nm_ratio",
        "over_10nm_ratio",
        "roll",
        "pitch",
        "yaw",
    ]
    for prefix in (
        "swing_mask",
        "stance_mask",
        "joint_limit_clip_mask",
        "rate_limit_clip_mask",
        "acceleration_clip_mask",
        "torque_clip_mask",
        "q_ref_semantic",
        "q_ref_isaac",
        "q_after_joint_limit",
        "q_after_rate_limit",
        "q_after_accel_limit",
        "q_after_torque_clip",
        "q_before_delay",
        "q_after_delay",
        "q_cmd_final",
        "joint_pos",
        "q_ref_error",
        "q_cmd_error",
        "tau_est",
        "base_ang_vel",
    ):
        header.extend(
            _vector_columns(
                prefix,
                4 if prefix in ("swing_mask", "stance_mask") else (3 if prefix == "base_ang_vel" else 12),
            )
        )

    print(
        f"[REFERENCE_DEBUG] mode={mode} output={output_path} "
        f"control_dt={base_env.step_dt:.6f}s physics_dt={base_env.cfg.sim.dt:.6f}s "
        f"decimation={base_env.cfg.decimation}"
    )
    if mode == "joint_mapping_debug":
        print("[JOINT_MAPPING] sequence:", " -> ".join(JOINT_NAMES), "(+0.1 rad each)")

    step = 0
    max_steps = max(1, round(float(args_cli.duration) / float(base_env.step_dt)))
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        while simulation_app.is_running() and step < max_steps:
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=base_env.device)
                env.step(actions)
                debug = action_term.get_debug_info()

                active_index = int(_scalar(debug["active_swing_leg"]))
                active_name = LEG_NAMES[active_index] if 0 <= active_index < 4 else "STANCE"
                mapping_index = int(_scalar(debug["joint_mapping_index"]))
                mapping_name = JOINT_NAMES[mapping_index] if 0 <= mapping_index < 12 else "DEFAULT_POSE"
                joint_pos = robot.data.joint_pos[:, action_term._joint_ids]
                q_ref_error = debug["simulator_q_ref"] - joint_pos
                q_cmd_error = debug["final_q_cmd"] - joint_pos
                roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
                command = base_env.command_manager.get_command("base_velocity")

                row = [
                    step * float(base_env.step_dt),
                    mode,
                    _scalar(action_term.reference.base_phase),
                    active_name,
                    mapping_name,
                    float(command[0, 0].detach().cpu()),
                    _scalar(debug["stride"]),
                    _scalar(debug["frequency"]),
                    _scalar(debug["swing_height"]),
                    _scalar(debug["duty_factor"]),
                    _scalar(debug["warmup"]),
                    _scalar(debug["control_dt"]),
                    _scalar(debug["physics_dt"]),
                    _scalar(debug["decimation"]),
                    _scalar(debug["phase_increment_per_step"]),
                    _scalar(debug["phase_cycle_time"]),
                    int(_scalar(debug["delay_steps"])),
                    _scalar(debug["joint_limit_clipping_ratio"]),
                    _scalar(debug["rate_limit_clipping_ratio"]),
                    _scalar(debug["acceleration_clipping_ratio"]),
                    _scalar(debug["torque_clipping_ratio"]),
                    _scalar(debug["filter_clipping_ratio"]),
                    float(torch.max(torch.abs(debug["simulator_q_ref"] - debug["q_after_rate_limit"])).detach().cpu()),
                    float(torch.max(torch.abs(debug["simulator_q_ref"] - debug["q_after_torque_clip"])).detach().cpu()),
                    float(torch.max(torch.abs(debug["simulator_q_ref"] - debug["final_q_cmd"])).detach().cpu()),
                    _scalar(debug["tau_est_max"]),
                    _scalar(debug["tau_est_mean"]),
                    _scalar(debug["over_6nm_ratio"]),
                    _scalar(debug["over_8nm_ratio"]),
                    _scalar(debug["over_10nm_ratio"]),
                    float(roll[0].detach().cpu()),
                    float(pitch[0].detach().cpu()),
                    float(yaw[0].detach().cpu()),
                ]
                row += _row_vector(debug["swing_mask"].to(torch.float32))
                row += _row_vector(debug["stance_mask"].to(torch.float32))
                row += _row_vector(debug["joint_limit_clip_mask"].to(torch.float32))
                row += _row_vector(debug["rate_limit_clip_mask"].to(torch.float32))
                row += _row_vector(debug["acceleration_clip_mask"].to(torch.float32))
                row += _row_vector(debug["torque_clip_mask"].to(torch.float32))
                row += _row_vector(debug["policy_q_ref"])
                row += _row_vector(debug["simulator_q_ref"])
                row += _row_vector(debug["q_after_joint_limit"])
                row += _row_vector(debug["q_after_rate_limit"])
                row += _row_vector(debug["q_after_accel_limit"])
                row += _row_vector(debug["q_after_torque_clip"])
                row += _row_vector(debug["q_before_delay"])
                row += _row_vector(debug["q_after_delay"])
                row += _row_vector(debug["final_q_cmd"])
                row += _row_vector(joint_pos)
                row += _row_vector(q_ref_error)
                row += _row_vector(q_cmd_error)
                row += _row_vector(debug["tau_est_per_joint"])
                row += _row_vector(robot.data.root_ang_vel_b)
                writer.writerow(row)

                if step % max(1, round(1.0 / float(base_env.step_dt))) == 0:
                    max_error = torch.max(torch.abs(debug["simulator_q_ref"] - debug["final_q_cmd"]))
                    print(
                        f"[REFERENCE_DEBUG] t={step * base_env.step_dt:6.2f}s "
                        f"phase={_scalar(action_term.reference.base_phase):.3f} active={active_name} "
                        f"tau_max={_scalar(debug['tau_est_max']):.2f}Nm "
                        f"max|q_ref-q_cmd|={float(max_error.detach().cpu()):.4f}rad "
                        f"clips(j/r/a/t)={_scalar(debug['joint_limit_clipping_ratio']):.2f}/"
                        f"{_scalar(debug['rate_limit_clipping_ratio']):.2f}/"
                        f"{_scalar(debug['acceleration_clipping_ratio']):.2f}/"
                        f"{_scalar(debug['torque_clipping_ratio']):.2f}"
                    )
                step += 1
        csv_file.flush()

    env.close()
    print(f"[REFERENCE_DEBUG] wrote {step} rows to {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
