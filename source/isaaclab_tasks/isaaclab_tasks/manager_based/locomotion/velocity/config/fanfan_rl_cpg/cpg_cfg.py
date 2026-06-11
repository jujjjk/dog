from __future__ import annotations

from isaaclab.utils import configclass


FANFAN_POLICY_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


@configclass
class CPGJointSineCfg:
    hip_amp: float = 0.025
    thigh_amp: float = 0.18
    calf_amp: float = 0.0
    thigh_phase_shift: float = 0.0
    calf_phase_shift: float = 0.45
    swing_lift_thigh_amp: float = 0.0
    swing_lift_calf_amp: float = 0.60
    stance_calf_amp: float = 0.08
    stride_sign: float = -1.0
    enable_hip_balance: bool = True
    hip_stance_widen_amp: float = 0.020
    hip_swing_relax_amp: float = 0.008
    hip_balance_signs: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    hip_balance_use_stance_mask: bool = True
    hip_balance_smooth_shape: str = "sin"
    hip_balance_max_abs: float = 0.06


@configclass
class CPGFilterCfg:
    use_rate_limit: bool = True
    max_delta_per_step: float = 0.03
    use_lowpass: bool = True
    lowpass_alpha: float = 0.35
    use_joint_limit_clip: bool = True
    use_torque_clip: bool = True
    use_action_delay: bool = True


@configclass
class CPGCfg:
    enable: bool = True
    mode: str = "joint_sine"  # off, joint_sine, foot_ik
    gait: str = "trot"

    leg_order: tuple[str, str, str, str] = ("FR", "FL", "RR", "RL")
    joint_order: tuple[str, ...] = FANFAN_POLICY_JOINT_ORDER
    joint_signs: tuple[float, ...] = (1.0,) * 12
    joint_offsets: tuple[float, ...] = (0.0,) * 12

    phase_offsets: dict[str, dict[str, float]] = {
        "trot": {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5},
        "pace": {"FR": 0.0, "RR": 0.0, "FL": 0.5, "RL": 0.5},
        "bound": {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5},
        "walk": {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75},
    }

    freq_mode: str = "linear"
    freq_min: float = 0.8
    freq_max: float = 1.8
    freq_default: float = 1.2
    k_freq: float = 3.0
    k_step: float = 0.5

    duty_factor: float = 0.60
    step_height: float = 0.030
    step_length_min: float = 0.015
    step_length_max: float = 0.060
    nominal_body_height: float = 0.293
    standing_cmd_threshold: float = 0.03
    foot_ik_thigh_length: float = 0.15606
    foot_ik_calf_length: float = 0.14894
    foot_ik_reach_margin: float = 0.010

    residual_limit: float = 0.06
    residual_limit_hip: float = 0.03
    residual_limit_thigh: float = 0.06
    residual_limit_calf: float = 0.06
    enable_phase_aware_hip_gate: bool = True
    hip_gate_stance_min_outward: float = 0.008
    hip_gate_swing_max_outward: float = 0.035
    hip_gate_side_signs: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)

    initial_phase_random: bool = True
    motor_profile_path: str = "config/motor_profile_fitted.yaml"
    randomization_profile_path: str = "config/randomization_profile.yaml"

    joint_sine: CPGJointSineCfg = CPGJointSineCfg()
    filter: CPGFilterCfg = CPGFilterCfg()
