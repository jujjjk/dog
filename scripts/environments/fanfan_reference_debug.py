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
parser.add_argument("--rear_leg", choices=("RR", "RL"), default=None)
parser.add_argument("--rear_thigh", type=float, default=None)
parser.add_argument("--rear_calf", type=float, default=None)
parser.add_argument("--rear_lift_height", type=float, default=0.030)
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
    if args_cli.rear_leg is not None:
        env_cfg.actions.joint_pos.rear_lift_test_leg = args_cli.rear_leg
    if args_cli.rear_thigh is not None:
        env_cfg.actions.joint_pos.rear_lift_test_thigh = args_cli.rear_thigh
        env_cfg.scene.robot.init_state.joint_pos["RR_thigh_joint"] = args_cli.rear_thigh
        env_cfg.scene.robot.init_state.joint_pos["RL_thigh_joint"] = args_cli.rear_thigh
    if args_cli.rear_calf is not None:
        env_cfg.actions.joint_pos.rear_lift_test_calf = args_cli.rear_calf
        env_cfg.scene.robot.init_state.joint_pos["RR_calf_joint"] = args_cli.rear_calf
        env_cfg.scene.robot.init_state.joint_pos["RL_calf_joint"] = args_cli.rear_calf
    env_cfg.actions.joint_pos.rear_lift_test_height_m = float(args_cli.rear_lift_height)

    mode = str(env_cfg.actions.joint_pos.action_mode)
    default_output_name = args_cli.task.removeprefix("Isaac-Velocity-Flat-").removesuffix("-v0")
    output_path = Path(
        args_cli.output or f"logs/reference_debug/{default_output_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    action_term = base_env.action_manager.get_term("joint_pos")
    robot = base_env.scene["robot"]
    env.reset()

    header = [
        "time",
        "mode",
        "stage",
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
        "raw_target_rate_max",
        "over_6nm_ratio",
        "over_8nm_ratio",
        "over_10nm_ratio",
        "roll",
        "pitch",
        "yaw",
    ]
    for prefix in (
        "leg_phase",
        "swing_mask",
        "stance_mask",
        "preload_gate",
        "post_touchdown_gate",
        "support_gate",
        "joint_limit_clip_mask",
        "rate_limit_clip_mask",
        "acceleration_clip_mask",
        "torque_clip_mask",
        "q_cpg_semantic",
        "q_cpg_isaac",
        "q_vmc_delta",
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
        "raw_target_rate",
        "base_ang_vel",
        "predicted_foot_height",
        "actual_foot_height",
        "actual_foot_height_body",
    ):
        header.extend(
            _vector_columns(
                prefix,
                4
                if prefix in (
                    "leg_phase",
                    "swing_mask",
                    "stance_mask",
                    "preload_gate",
                    "post_touchdown_gate",
                    "support_gate",
                    "predicted_foot_height",
                    "actual_foot_height",
                    "actual_foot_height_body",
                )
                else (3 if prefix == "base_ang_vel" else 12),
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
    clip_ratio_sum = torch.zeros(4)
    predicted_lift_max = torch.full((4,), float("-inf"))
    actual_height_min = torch.full((4,), float("inf"))
    actual_height_max = torch.full((4,), float("-inf"))
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
                clip_ratio_sum += torch.tensor(
                    [
                        _scalar(debug["joint_limit_clipping_ratio"]),
                        _scalar(debug["rate_limit_clipping_ratio"]),
                        _scalar(debug["acceleration_clipping_ratio"]),
                        _scalar(debug["torque_clipping_ratio"]),
                    ]
                )
                predicted = debug["predicted_foot_height"][0].detach().cpu()
                actual = debug["actual_foot_height_body"][0].detach().cpu()
                predicted_lift_max = torch.maximum(predicted_lift_max, predicted)
                actual_height_min = torch.minimum(actual_height_min, actual)
                actual_height_max = torch.maximum(actual_height_max, actual)

                row = [
                    step * float(base_env.step_dt),
                    mode,
                    int(_scalar(debug["control_stage"])),
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
                    _scalar(debug["raw_target_rate_max"]),
                    _scalar(debug["over_6nm_ratio"]),
                    _scalar(debug["over_8nm_ratio"]),
                    _scalar(debug["over_10nm_ratio"]),
                    float(roll[0].detach().cpu()),
                    float(pitch[0].detach().cpu()),
                    float(yaw[0].detach().cpu()),
                ]
                row += _row_vector(debug["leg_phase"])
                row += _row_vector(debug["swing_mask"].to(torch.float32))
                row += _row_vector(debug["stance_mask"].to(torch.float32))
                row += _row_vector(debug["preload_gate"])
                row += _row_vector(debug["post_touchdown_gate"])
                row += _row_vector(debug["support_gate"])
                row += _row_vector(debug["joint_limit_clip_mask"].to(torch.float32))
                row += _row_vector(debug["rate_limit_clip_mask"].to(torch.float32))
                row += _row_vector(debug["acceleration_clip_mask"].to(torch.float32))
                row += _row_vector(debug["torque_clip_mask"].to(torch.float32))
                row += _row_vector(debug["q_cpg_policy"])
                row += _row_vector(debug["q_cpg_simulator"])
                row += _row_vector(debug["q_vmc_delta"])
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
                row += _row_vector(debug["raw_target_rate_per_joint"])
                row += _row_vector(robot.data.root_ang_vel_b)
                row += _row_vector(debug["predicted_foot_height"])
                row += _row_vector(debug["actual_foot_height"])
                row += _row_vector(debug["actual_foot_height_body"])
                writer.writerow(row)

                if step % max(1, round(0.1 / float(base_env.step_dt))) == 0:
                    max_error = torch.max(torch.abs(debug["simulator_q_ref"] - debug["final_q_cmd"]))
                    leg_phase = ",".join(f"{value:.2f}" for value in _row_vector(debug["leg_phase"]))
                    swing_mask = "".join(
                        str(int(value)) for value in _row_vector(debug["swing_mask"].to(torch.float32))
                    )
                    preload = ",".join(f"{value:.2f}" for value in _row_vector(debug["preload_gate"]))
                    post = ",".join(
                        f"{value:.2f}" for value in _row_vector(debug["post_touchdown_gate"])
                    )
                    q_ref = ",".join(f"{value:.3f}" for value in _row_vector(debug["simulator_q_ref"]))
                    q_cmd = ",".join(f"{value:.3f}" for value in _row_vector(debug["final_q_cmd"]))
                    q_actual = ",".join(f"{value:.3f}" for value in _row_vector(joint_pos))
                    clamp_flags = (
                        f"j={int(torch.any(debug['joint_limit_clip_mask'][0]).item())},"
                        f"r={int(torch.any(debug['rate_limit_clip_mask'][0]).item())},"
                        f"a={int(torch.any(debug['acceleration_clip_mask'][0]).item())},"
                        f"t={int(torch.any(debug['torque_clip_mask'][0]).item())}"
                    )
                    print(
                        f"[REFERENCE_DEBUG] t={step * base_env.step_dt:6.2f}s "
                        f"stage={int(_scalar(debug['control_stage']))} "
                        f"phase={_scalar(action_term.reference.base_phase):.3f} active={active_name} "
                        f"leg_phase=[{leg_phase}] swing={swing_mask} "
                        f"preload=[{preload}] post=[{post}] "
                        f"q_ref=[{q_ref}] q_target=[{q_cmd}] q_actual=[{q_actual}] "
                        f"clamp({clamp_flags}) "
                        f"max|q_ref-q_cmd|={float(max_error.detach().cpu()):.4f}rad"
                    )
                    if mode == "rear_lift_test":
                        rear_leg = str(action_term.cfg.rear_lift_test_leg).upper()
                        rear_index = 2 if rear_leg == "RR" else 3
                        thigh_id = rear_index * 3 + 1
                        calf_id = rear_index * 3 + 2
                        print(
                            "[REAR_LIFT] "
                            f"leg={rear_leg} "
                            f"pred={_row_vector(debug['predicted_foot_height'])[rear_index]:.4f}m "
                            f"actual_body_z={_row_vector(debug['actual_foot_height_body'])[rear_index]:.4f}m "
                            f"q_ref(thigh/calf)="
                            f"{_row_vector(debug['simulator_q_ref'])[thigh_id]:.3f}/"
                            f"{_row_vector(debug['simulator_q_ref'])[calf_id]:.3f} "
                            f"q_cmd="
                            f"{_row_vector(debug['final_q_cmd'])[thigh_id]:.3f}/"
                            f"{_row_vector(debug['final_q_cmd'])[calf_id]:.3f} "
                            f"q_pos={_row_vector(joint_pos)[thigh_id]:.3f}/"
                            f"{_row_vector(joint_pos)[calf_id]:.3f} "
                            f"q_err={_row_vector(q_ref_error)[thigh_id]:.3f}/"
                            f"{_row_vector(q_ref_error)[calf_id]:.3f} "
                            f"tau={_row_vector(debug['tau_est_per_joint'])[thigh_id]:.2f}/"
                            f"{_row_vector(debug['tau_est_per_joint'])[calf_id]:.2f}Nm"
                        )
                step += 1
        csv_file.flush()

    env.close()
    print(f"[REFERENCE_DEBUG] wrote {step} rows to {output_path}")
    if step > 0:
        mean_clips = clip_ratio_sum / step
        actual_lift = actual_height_max - actual_height_min
        lift_ratio = actual_lift / torch.clamp(predicted_lift_max, min=1.0e-6)
        print(
            "[REFERENCE_SUMMARY] "
            f"mean_clips(j/r/a/t)={mean_clips[0]:.3f}/{mean_clips[1]:.3f}/"
            f"{mean_clips[2]:.3f}/{mean_clips[3]:.3f} "
            f"predicted_lift_max={predicted_lift_max.tolist()} "
            f"actual_lift={actual_lift.tolist()} "
            f"actual/predicted={lift_ratio.tolist()}"
        )
        if "Stage1-Safe" in args_cli.task and (
            mean_clips[1] > 0.10
            or mean_clips[3] > 0.10
            or torch.min(lift_ratio) < 0.60
        ):
            print(
                "[REFERENCE_SAFE_BUDGET_WARNING] Current trajectory exceeds the "
                "5 rad/s / 6 N.m safety profile; filtered motion must not be "
                "interpreted as proof that the gait itself is executable."
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
