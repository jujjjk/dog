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
parser.add_argument("--body_shift_x", type=float, default=None)
parser.add_argument("--body_shift_y", type=float, default=None)
parser.add_argument(
    "--target_unload_z",
    type=float,
    choices=(0.012, 0.018, 0.024, 0.030),
    default=None,
)
parser.add_argument("--main_support_push_z", type=float, default=None)
parser.add_argument("--front_support_push_z", type=float, default=None)
parser.add_argument("--rear_support_push_z", type=float, default=None)
parser.add_argument(
    "--foot_down_signs",
    type=str,
    default=None,
    help="Comma-separated FR,FL,RR,RL z signs reported by PressSignTest.",
)
parser.add_argument(
    "--support_kp_level",
    choices=("mid", "high", "very_high"),
    default=None,
    help="Simulation-only thigh/calf stiffness sweep; damping is fixed at 5.0.",
)
parser.add_argument(
    "--trot_preset",
    choices=("conservative", "balanced", "fast"),
    default=None,
    help="Preset parameters for FastDiagonalTrot-Reference-v0.",
)
parser.add_argument(
    "--fast_trot_profile",
    choices=("conservative", "balanced", "fast"),
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--fast_trot_support_preload_z",
    type=float,
    choices=(0.006, 0.008, 0.010),
    default=None,
    help="Positive magnitude in meters; applied as negative foot-z support preload.",
)
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
PAIR_NAMES = ("STANCE", "FR+RL", "FL+RR")
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
    if args_cli.body_shift_x is not None:
        env_cfg.actions.joint_pos.rear_lift_body_shift_x_m = float(args_cli.body_shift_x)
    if args_cli.body_shift_y is not None:
        env_cfg.actions.joint_pos.rear_lift_body_shift_y_m = abs(float(args_cli.body_shift_y))
    if args_cli.target_unload_z is not None:
        env_cfg.actions.joint_pos.rear_lift_target_unload_m = float(args_cli.target_unload_z)
    if args_cli.main_support_push_z is not None:
        push = abs(float(args_cli.main_support_push_z))
        env_cfg.actions.joint_pos.rear_lift_same_front_preload_m = push
        env_cfg.actions.joint_pos.rear_lift_other_rear_preload_m = push
    if args_cli.front_support_push_z is not None:
        env_cfg.actions.joint_pos.rear_lift_same_front_preload_m = abs(
            float(args_cli.front_support_push_z)
        )
    if args_cli.rear_support_push_z is not None:
        env_cfg.actions.joint_pos.rear_lift_other_rear_preload_m = abs(
            float(args_cli.rear_support_push_z)
        )
    if args_cli.foot_down_signs:
        signs = tuple(float(value) for value in args_cli.foot_down_signs.split(","))
        if len(signs) != 4 or any(value not in (-1.0, 1.0) for value in signs):
            raise ValueError("--foot_down_signs must contain four comma-separated +/-1 values.")
        env_cfg.actions.joint_pos.rear_lift_foot_down_signs = signs
    trot_preset = args_cli.trot_preset or args_cli.fast_trot_profile
    if trot_preset == "conservative":
        env_cfg.actions.joint_pos.fast_trot_step_hz = 1.10
        env_cfg.actions.joint_pos.fast_trot_duty_factor = 0.62
        env_cfg.actions.joint_pos.fast_trot_stride_length_m = 0.020
        env_cfg.actions.joint_pos.fast_trot_front_swing_height_m = 0.045
        env_cfg.actions.joint_pos.fast_trot_rear_swing_height_m = 0.065
        env_cfg.actions.joint_pos.fast_trot_support_preload_z_m = 0.008
    elif trot_preset == "balanced":
        env_cfg.actions.joint_pos.fast_trot_step_hz = 1.15
        env_cfg.actions.joint_pos.fast_trot_duty_factor = 0.61
        env_cfg.actions.joint_pos.fast_trot_stride_length_m = 0.022
        env_cfg.actions.joint_pos.fast_trot_front_swing_height_m = 0.048
        env_cfg.actions.joint_pos.fast_trot_rear_swing_height_m = 0.067
        env_cfg.actions.joint_pos.fast_trot_support_preload_z_m = 0.009
    elif trot_preset == "fast":
        env_cfg.actions.joint_pos.fast_trot_step_hz = 1.20
        env_cfg.actions.joint_pos.fast_trot_duty_factor = 0.60
        env_cfg.actions.joint_pos.fast_trot_stride_length_m = 0.024
        env_cfg.actions.joint_pos.fast_trot_front_swing_height_m = 0.050
        env_cfg.actions.joint_pos.fast_trot_rear_swing_height_m = 0.070
        env_cfg.actions.joint_pos.fast_trot_support_preload_z_m = 0.010
    if "FastDiagonalTrot" in args_cli.task:
        env_cfg.actions.joint_pos.fast_trot_preset = trot_preset or "conservative"
        kp_level = args_cli.support_kp_level or "high"
        kp_profiles = {
            "mid": (160.0, 180.0),
            "high": (180.0, 200.0),
            "very_high": (220.0, 220.0),
        }
        support_thigh_kp, support_calf_kp = kp_profiles[kp_level]
        env_cfg.actions.joint_pos.fast_trot_support_thigh_kp = support_thigh_kp
        env_cfg.actions.joint_pos.fast_trot_support_calf_kp = support_calf_kp
    if args_cli.fast_trot_support_preload_z is not None:
        env_cfg.actions.joint_pos.fast_trot_support_preload_z_m = float(
            args_cli.fast_trot_support_preload_z
        )

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
    support_kp_level = args_cli.support_kp_level or ("high" if mode == "fast_diagonal_trot" else "default")
    if args_cli.support_kp_level and mode != "fast_diagonal_trot":
        kp_profiles = {
            "mid": (140.0, 160.0),
            "high": (180.0, 200.0),
            "very_high": (220.0, 220.0),
        }
        thigh_kp, calf_kp = kp_profiles[args_cli.support_kp_level]
        stiffness = robot.data.default_joint_stiffness.clone()
        damping = robot.data.default_joint_damping.clone()
        target_leg = str(action_term.cfg.rear_lift_test_leg).upper()
        target_leg_index = 2 if target_leg == "RR" else 3
        support_leg_indices = [0, 3] if target_leg == "RR" else [1, 2]
        thigh_ids = [
            int(action_term._joint_ids[index * 3 + 1]) for index in support_leg_indices
        ]
        calf_ids = [
            int(action_term._joint_ids[index * 3 + 2]) for index in support_leg_indices
        ]
        support_ids = thigh_ids + calf_ids
        stiffness[:, thigh_ids] = thigh_kp
        stiffness[:, calf_ids] = calf_kp
        damping[:, support_ids] = 5.0
        robot.write_joint_stiffness_to_sim(stiffness)
        robot.write_joint_damping_to_sim(damping)
        action_term.debug_kp_override = stiffness[:, action_term._joint_ids].clone()
        action_term.debug_kd_override = damping[:, action_term._joint_ids].clone()
        print(
            f"[SUPPORT_KP_SWEEP] level={support_kp_level} support_legs="
            f"{','.join(LEG_NAMES[index] for index in support_leg_indices)} "
            f"thigh={thigh_kp:.1f} calf={calf_kp:.1f} kd=5.0"
        )

    header = [
        "time",
        "mode",
        "stage",
        "base_phase",
        "active_swing_leg",
        "active_swing_pair",
        "support_pair",
        "rear_lift_phase",
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
        "q_error_max",
        "unsafe_torque",
        "tau_est_mean",
        "raw_target_rate_max",
        "over_6nm_ratio",
        "over_8nm_ratio",
        "over_10nm_ratio",
        "roll",
        "pitch",
        "yaw",
        "base_height",
        "base_roll",
        "base_pitch",
        "base_yaw",
        "target_leg_unload_delta_z",
        "body_shift_x",
        "body_shift_y",
        "diagnostic_leg",
        "diagnostic_delta_z",
        "diagnostic_force_before",
        "diagnostic_force_after",
        "target_rear_leg",
        "target_normal_force",
        "main_support_leg",
        "main_support_force",
        "target_unload_z",
        "support_push_z_FR",
        "support_push_z_FL",
        "support_push_z_RR",
        "support_push_z_RL",
        "support_kp_level",
        "force_drop_success",
        "failure_reason",
        "force_below_threshold",
        "force_below_timer",
        "first_force_drop_time",
        "lift_entry_time",
        "missed_force_drop_window",
        "state_transition_reason",
        "actual_support_pair",
    ]
    for prefix in (
        "leg_phase",
        "swing_mask",
        "support_mask",
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
        "q_ref",
        "q_after_joint_limit",
        "q_after_rate_limit",
        "q_after_accel_limit",
        "q_after_torque_clip",
        "q_before_delay",
        "q_after_delay",
        "q_cmd_final",
        "q_actual",
        "q_error",
        "q_cmd_error",
        "tau_est",
        "kp",
        "kd",
        "raw_target_rate",
        "base_ang_vel",
        "predicted_foot_height",
        "foot_world_z",
        "foot_body_z",
        "foot_contact_state",
        "foot_normal_force",
        "support_preload_delta_z",
    ):
        header.extend(
            _vector_columns(
                prefix,
                4
                if prefix in (
                    "leg_phase",
                    "swing_mask",
                    "support_mask",
                    "preload_gate",
                    "post_touchdown_gate",
                    "support_gate",
                    "predicted_foot_height",
                    "foot_world_z",
                    "foot_body_z",
                    "foot_contact_state",
                    "foot_normal_force",
                    "support_preload_delta_z",
                )
                else (3 if prefix == "base_ang_vel" else 12),
            )
        )

    print(
        f"[REFERENCE_DEBUG] mode={mode} output={output_path} "
        f"control_dt={base_env.step_dt:.6f}s physics_dt={base_env.cfg.sim.dt:.6f}s "
        f"decimation={base_env.cfg.decimation}"
    )
    print(
        f"[CSV_SCHEMA] columns={len(header)} "
        f"first_fields={header[:12]}"
    )
    print(
        "[LEG_INDEX_MAP] "
        + " ".join(
            f"{leg}:joints={list(action_term._joint_ids[index * 3:index * 3 + 3])},"
            f"body={action_term._foot_body_ids[index]},"
            f"contact={action_term._contact_foot_ids[index]}"
            for index, leg in enumerate(LEG_NAMES)
        )
    )
    if mode == "joint_mapping_debug":
        print("[JOINT_MAPPING] sequence:", " -> ".join(JOINT_NAMES), "(+0.1 rad each)")

    step = 0
    max_steps = max(1, round(float(args_cli.duration) / float(base_env.step_dt)))
    clip_ratio_sum = torch.zeros(4)
    predicted_lift_max = torch.full((4,), float("-inf"))
    actual_height_min = torch.full((4,), float("inf"))
    actual_height_max = torch.full((4,), float("-inf"))
    rear_lift_world_min = float("inf")
    rear_lift_world_max = float("-inf")
    rear_lift_force_min = float("inf")
    rear_lift_airborne_samples = 0
    rear_lift_samples = 0
    rear_lift_phase_forces = {}
    rear_lift_max_phase = 0
    press_samples = {}
    shift_samples = {}
    csv_schema_checked = False
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
                active_pair_index = int(_scalar(debug["active_swing_pair"]))
                active_pair_name = PAIR_NAMES[active_pair_index] if 0 <= active_pair_index < 3 else "UNKNOWN"
                support_pair_index = int(_scalar(debug["expected_support_pair"]))
                support_pair_name = PAIR_NAMES[support_pair_index] if 0 <= support_pair_index < 3 else "UNKNOWN"
                mapping_index = int(_scalar(debug["joint_mapping_index"]))
                mapping_name = JOINT_NAMES[mapping_index] if 0 <= mapping_index < 12 else "DEFAULT_POSE"
                joint_pos = robot.data.joint_pos[:, action_term._joint_ids]
                q_ref_error = debug["simulator_q_ref"] - joint_pos
                q_cmd_error = debug["final_q_cmd"] - joint_pos
                q_error_max = float(torch.max(torch.abs(q_ref_error)).detach().cpu())
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
                if mode == "rear_lift_test" and int(_scalar(debug["rear_lift_phase"])) == 5:
                    rear_leg = str(action_term.cfg.rear_lift_test_leg).upper()
                    rear_index = 2 if rear_leg == "RR" else 3
                    if _row_vector(debug["predicted_foot_height"])[rear_index] > 0.005:
                        world_z = _row_vector(debug["actual_foot_height"])[rear_index]
                        normal_force = _row_vector(debug["foot_normal_force"])[rear_index]
                        rear_lift_world_min = min(rear_lift_world_min, world_z)
                        rear_lift_world_max = max(rear_lift_world_max, world_z)
                        rear_lift_force_min = min(rear_lift_force_min, normal_force)
                        rear_lift_airborne_samples += int(
                            not bool(_row_vector(debug["foot_contact_state"])[rear_index])
                        )
                        rear_lift_samples += 1
                diagnostic_leg = int(_scalar(debug["diagnostic_leg"]))
                diagnostic_delta = _scalar(debug["diagnostic_delta_z"])
                target_rear_leg = str(action_term.cfg.rear_lift_test_leg).upper()
                target_rear_index = 2 if target_rear_leg == "RR" else 3
                foot_forces = _row_vector(debug["foot_normal_force"])
                support_indices = [
                    index for index in range(4) if index != target_rear_index
                ]
                main_support_index = max(
                    support_indices, key=lambda index: foot_forces[index]
                )
                actual_support_indices = sorted(
                    range(4), key=lambda index: foot_forces[index], reverse=True
                )[:2]
                actual_support_pair = "+".join(
                    LEG_NAMES[index] for index in actual_support_indices
                )
                support_push = _row_vector(debug["support_preload_delta_z"])
                force_drop_success = bool(_scalar(debug["force_drop_success"]))
                failure_code = int(_scalar(debug["failure_reason"]))
                failure_reason = {
                    0: "",
                    1: "force_drop_timeout",
                }.get(failure_code, f"unknown_{failure_code}")
                transition_code = int(_scalar(debug["state_transition_reason"]))
                state_transition_reason = {
                    0: "",
                    1: "default_complete",
                    2: "pre_shift_complete",
                    3: "preload_complete",
                    4: "unload_complete_wait",
                    5: "force_drop_confirmed_during_unload",
                    6: "force_drop_confirmed_during_wait",
                    7: "force_drop_timeout",
                }.get(transition_code, f"unknown_{transition_code}")
                if mode == "rear_lift_test":
                    rear_phase = int(_scalar(debug["rear_lift_phase"]))
                    rear_lift_max_phase = max(rear_lift_max_phase, rear_phase)
                    rear_lift_phase_forces.setdefault(rear_phase, []).append(
                        _row_vector(debug["foot_normal_force"])
                    )
                if mode == "press_sign_test" and diagnostic_leg >= 0 and abs(diagnostic_delta) > 1.0e-6:
                    key = (diagnostic_leg, round(diagnostic_delta, 4))
                    press_samples.setdefault(key, []).append(
                        (
                            _scalar(debug["diagnostic_force_before"]),
                            _row_vector(debug["foot_normal_force"])[diagnostic_leg],
                        )
                    )
                if mode == "body_shift_sweep":
                    shift = _row_vector(debug["body_shift_xy"])
                    key = (round(shift[0], 4), round(shift[1], 4))
                    shift_samples.setdefault(key, []).append(
                        _row_vector(debug["foot_normal_force"])
                        + [
                            _scalar(debug["base_height"]),
                            _row_vector(debug["base_rpy"])[0],
                            _row_vector(debug["base_rpy"])[1],
                        ]
                    )

                row = [
                    step * float(base_env.step_dt),
                    mode,
                    int(_scalar(debug["control_stage"])),
                    _scalar(action_term.reference.base_phase),
                    active_name,
                    active_pair_name,
                    support_pair_name,
                    int(_scalar(debug["rear_lift_phase"])),
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
                    q_error_max,
                    int(_scalar(debug["tau_est_max"]) > 17.0),
                    _scalar(debug["tau_est_mean"]),
                    _scalar(debug["raw_target_rate_max"]),
                    _scalar(debug["over_6nm_ratio"]),
                    _scalar(debug["over_8nm_ratio"]),
                    _scalar(debug["over_10nm_ratio"]),
                    float(roll[0].detach().cpu()),
                    float(pitch[0].detach().cpu()),
                    float(yaw[0].detach().cpu()),
                    _scalar(debug["base_height"]),
                    _row_vector(debug["base_rpy"])[0],
                    _row_vector(debug["base_rpy"])[1],
                    _row_vector(debug["base_rpy"])[2],
                    _scalar(debug["target_leg_unload_delta_z"]),
                    _row_vector(debug["body_shift_xy"])[0],
                    _row_vector(debug["body_shift_xy"])[1],
                    diagnostic_leg,
                    diagnostic_delta,
                    _scalar(debug["diagnostic_force_before"]),
                    _scalar(debug["diagnostic_force_after"]),
                    target_rear_leg,
                    foot_forces[target_rear_index],
                    LEG_NAMES[main_support_index],
                    foot_forces[main_support_index],
                    float(action_term.cfg.rear_lift_target_unload_m),
                    support_push[0],
                    support_push[1],
                    support_push[2],
                    support_push[3],
                    support_kp_level,
                    int(force_drop_success),
                    failure_reason,
                    int(bool(_scalar(debug["force_below_threshold"]))),
                    _scalar(debug["force_below_timer"]),
                    _scalar(debug["first_force_drop_time"]),
                    _scalar(debug["lift_entry_time"]),
                    int(bool(_scalar(debug["missed_force_drop_window"]))),
                    state_transition_reason,
                    actual_support_pair,
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
                row += _row_vector(debug["joint_kp"])
                row += _row_vector(debug["joint_kd"])
                row += _row_vector(debug["raw_target_rate_per_joint"])
                row += _row_vector(robot.data.root_ang_vel_b)
                row += _row_vector(debug["predicted_foot_height"])
                row += _row_vector(debug["actual_foot_height"])
                row += _row_vector(debug["actual_foot_height_body"])
                row += _row_vector(debug["foot_contact_state"].to(torch.float32))
                row += _row_vector(debug["foot_normal_force"])
                row += _row_vector(debug["support_preload_delta_z"])
                if len(row) != len(header):
                    raise RuntimeError(
                        f"CSV schema mismatch: header has {len(header)} columns, "
                        f"row has {len(row)} columns."
                    )
                if not csv_schema_checked:
                    print(
                        f"[CSV_SCHEMA_CHECK] ok columns={len(header)} "
                        f"first_fields={header[:12]} "
                        f"last_fields={header[-12:]}"
                    )
                    csv_schema_checked = True
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
                        phase_name = (
                            "DEFAULT_POSE",
                            "PRE_SHIFT",
                            "PRELOAD",
                            "UNLOAD",
                            "WAIT_FORCE_DROP",
                            "LIFT",
                            "FAILED",
                        )[
                            int(_scalar(debug["rear_lift_phase"]))
                        ]
                        print(
                            "[REAR_LIFT] "
                            f"leg={rear_leg} phase={phase_name} "
                            f"body_shift={_row_vector(debug['body_shift_xy'])[0]:.3f}/"
                            f"{_row_vector(debug['body_shift_xy'])[1]:.3f}m "
                            f"base_z={_scalar(debug['base_height']):.4f}m "
                            f"pred={_row_vector(debug['predicted_foot_height'])[rear_index]:.4f}m "
                            f"world_z={_row_vector(debug['actual_foot_height'])[rear_index]:.4f}m "
                            f"actual_body_z={_row_vector(debug['actual_foot_height_body'])[rear_index]:.4f}m "
                            f"normal_force={_row_vector(debug['foot_normal_force'])[rear_index]:.2f}N "
                            f"contact={int(_row_vector(debug['foot_contact_state'].to(torch.float32))[rear_index])} "
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
        if mode == "rear_lift_test" and rear_lift_samples > 0:
            world_lift = rear_lift_world_max - rear_lift_world_min
            airborne_ratio = rear_lift_airborne_samples / rear_lift_samples
            print(
                "[REAR_LIFT_SUMMARY] "
                f"world_z_span={world_lift:.4f}m "
                f"min_normal_force={rear_lift_force_min:.2f}N "
                f"airborne_ratio={airborne_ratio:.3f}"
            )
            if world_lift < 0.010:
                print(
                    "[REAR_LIFT_SUPPORT_WARNING] Body motion or insufficient "
                    "support transfer is cancelling the commanded lift; world "
                    "foot clearance stayed below 10 mm."
                )
        if mode == "rear_lift_test" and rear_lift_samples == 0:
            rear_leg = str(action_term.cfg.rear_lift_test_leg).upper()
            target_index = 2 if rear_leg == "RR" else 3
            latest = rear_lift_phase_forces.get(rear_lift_max_phase, [])
            if latest:
                tail = latest[len(latest) // 2 :]
                mean_forces = [
                    sum(row[index] for row in tail) / len(tail) for index in range(4)
                ]
                support_indices = [
                    index for index in range(4) if index != target_index
                ]
                support_index = max(
                    support_indices, key=lambda index: mean_forces[index]
                )
                print(
                    f"[REAR_LIFT_FORCE_GATE] stopped_at_phase={rear_lift_max_phase} "
                    f"target={rear_leg} force={mean_forces[target_index]:.2f}N "
                    f"actual_main_support={LEG_NAMES[support_index]} "
                    f"force={mean_forces[support_index]:.2f}N threshold="
                    f"{action_term.cfg.rear_lift_force_drop_threshold_n:.2f}N"
                )
            if rear_lift_max_phase == 6:
                print(
                    "[REAR_LIFT_FORCE_GATE_FAILURE] failure_reason=force_drop_timeout; "
                    "target-foot force did not remain below threshold within "
                    f"{action_term.cfg.rear_lift_force_drop_timeout_sec:.2f} s."
                )
            else:
                print(
                    "[REAR_LIFT_FORCE_GATE_WARNING] LIFT was intentionally blocked "
                    "because target-foot contact force did not remain below the threshold."
                )
        if mode == "rear_lift_test" and 3 in rear_lift_phase_forces:
            unload_rows = rear_lift_phase_forces[3]
            tail = unload_rows[len(unload_rows) // 2 :]
            mean_forces = [
                sum(row[index] for row in tail) / len(tail) for index in range(4)
            ]
            support_pair = sorted(
                range(4), key=lambda index: mean_forces[index], reverse=True
            )[:2]
            print(
                "[REAR_LIFT_SUPPORT_PAIR] UNLOAD actual support pair="
                f"{LEG_NAMES[support_pair[0]]}+{LEG_NAMES[support_pair[1]]} "
                f"forces={mean_forces[support_pair[0]]:.2f}/"
                f"{mean_forces[support_pair[1]]:.2f}N"
            )
        if mode == "press_sign_test":
            print("[PRESS_SIGN_SUMMARY] positive force_delta means stronger ground press")
            by_leg = {}
            for (leg_index, delta), samples in sorted(press_samples.items()):
                tail = samples[len(samples) // 2 :]
                before = sum(item[0] for item in tail) / len(tail)
                after = sum(item[1] for item in tail) / len(tail)
                force_delta = after - before
                by_leg.setdefault(leg_index, []).append((delta, force_delta))
                print(
                    f"  leg={LEG_NAMES[leg_index]} applied_delta_z={delta:+.3f}m "
                    f"normal_force_before={before:.2f}N "
                    f"normal_force_after={after:.2f}N force_delta={force_delta:+.2f}N"
                )
            for leg_index, results in sorted(by_leg.items()):
                best = max(results, key=lambda item: item[1])
                print(
                    f"[PRESS_SIGN_RESULT] leg={LEG_NAMES[leg_index]} "
                    f"downward_press_sign={'+' if best[0] > 0 else '-'} "
                    f"delta_force={best[1]:+.2f}N"
                )
                if best[1] < 1.0:
                    print(
                        f"[PRESS_SIGN_WARNING] {LEG_NAMES[leg_index]} force did not "
                        "increase meaningfully; verify foot/contact index mapping."
                    )
        if mode == "body_shift_sweep" and shift_samples:
            target_leg = str(action_term.cfg.rear_lift_test_leg).upper()
            target_index = 2 if target_leg == "RR" else 3
            support_index = 1 if target_leg == "RR" else 0
            candidates = []
            for shift, samples in shift_samples.items():
                tail = samples[len(samples) // 2 :]
                mean = [sum(row[index] for row in tail) / len(tail) for index in range(7)]
                tilt = max(abs(mean[5]), abs(mean[6]))
                stable_penalty = 0 if tilt <= 0.05236 else 1
                candidates.append(
                    (
                        stable_penalty,
                        mean[target_index],
                        -mean[support_index],
                        abs(mean[5]) + abs(mean[6]),
                        shift,
                        mean,
                    )
                )
            best = min(candidates)
            shift = best[4]
            mean = best[5]
            print(
                f"[BODY_SHIFT_RESULT] unload={target_leg} shift_x/y={shift[0]:+.3f}/"
                f"{shift[1]:+.3f}m forces(FR/FL/RR/RL)="
                f"{mean[0]:.2f}/{mean[1]:.2f}/{mean[2]:.2f}/{mean[3]:.2f}N "
                f"base_z={mean[4]:.4f} roll/pitch={mean[5]:+.4f}/{mean[6]:+.4f}rad"
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
