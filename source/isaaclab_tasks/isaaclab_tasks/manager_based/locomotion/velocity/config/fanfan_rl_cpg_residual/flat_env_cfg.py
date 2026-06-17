from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.flat_env_cfg import (
    FanfanA1CleanFlatEnvCfg,
)

from . import mdp_observations as wave_obs
from . import mdp_rewards as wave_rew
from .curriculum import WAVE_CURRICULUM_STAGES, stage_gated_push, wave_curriculum
from .joint_semantics import SIM_JOINT_NAMES
from .reference_gait import FanfanSmallHighFreqReferenceGaitCfg
from .residual_action import WaveResidualJointPositionActionCfg
from .urdf_model import make_heavy_fanfan_cfg


JOINT_NAMES = list(SIM_JOINT_NAMES)
FOOT_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
    preserve_order=True,
)
JOINT_CFG = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
HEAVY_FANFAN_CFG, HEAVY_FANFAN_MODEL = make_heavy_fanfan_cfg()
SMALL_HIGH_FREQ_REAR_THIGH = 0.3491
SMALL_HIGH_FREQ_REAR_CALF = -0.7854
SMALL_HIGH_FREQ_INITIAL_BASE_HEIGHT = 0.300


def _set_rear_stand_pose(robot_cfg, thigh: float, calf: float) -> None:
    robot_cfg.init_state.pos = (0.0, 0.0, SMALL_HIGH_FREQ_INITIAL_BASE_HEIGHT)
    robot_cfg.init_state.joint_pos.update(
        {
            "RR_thigh_joint": float(thigh),
            "RL_thigh_joint": float(thigh),
            "RR_calf_joint": float(calf),
            "RL_calf_joint": float(calf),
        }
    )


@configclass
class WavePolicyCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=wave_obs.noisy_base_ang_vel)
    projected_gravity = ObsTerm(func=wave_obs.noisy_projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=wave_obs.noisy_ordered_joint_pos_rel,
        params={"asset_cfg": JOINT_CFG},
    )
    joint_vel = ObsTerm(
        func=wave_obs.noisy_ordered_joint_vel,
        params={"asset_cfg": JOINT_CFG},
    )
    actions = ObsTerm(func=wave_obs.last_residual_action)
    q_ref = ObsTerm(func=wave_obs.reference_joint_pos)
    q_ref_error = ObsTerm(func=wave_obs.reference_joint_error)
    phase = ObsTerm(func=wave_obs.reference_phase_features)
    active_leg = ObsTerm(func=wave_obs.active_swing_leg)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class WaveCriticCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=wave_obs.ordered_joint_pos_rel, params={"asset_cfg": JOINT_CFG})
    joint_vel = ObsTerm(func=wave_obs.ordered_joint_vel, params={"asset_cfg": JOINT_CFG})
    actions = ObsTerm(func=wave_obs.last_residual_action)
    q_ref = ObsTerm(func=wave_obs.reference_joint_pos)
    q_ref_error = ObsTerm(func=wave_obs.reference_joint_error)
    phase = ObsTerm(func=wave_obs.reference_phase_features)
    active_leg = ObsTerm(func=wave_obs.active_swing_leg)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    foot_contact = ObsTerm(func=wave_obs.normalized_foot_contact_forces, params={"sensor_cfg": FOOT_CFG})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class WaveObservationsCfg:
    policy: WavePolicyCfg = WavePolicyCfg()
    critic: WaveCriticCfg = WaveCriticCfg()


@configclass
class FanfanRlCpgResidualFlatEnvCfg(FanfanA1CleanFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HEAVY_FANFAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos = WaveResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=JOINT_NAMES,
            preserve_order=True,
            use_default_offset=False,
            scale=1.0,
            clip={
                "FR_hip_joint": (-0.16, 0.08), "FL_hip_joint": (-0.08, 0.16),
                "RR_hip_joint": (-0.16, 0.08), "RL_hip_joint": (-0.08, 0.16),
                "FR_thigh_joint": (-1.5708, 0.6458), "FL_thigh_joint": (-1.5708, 0.6458),
                "RR_thigh_joint": (-0.10, 0.60), "RL_thigh_joint": (-0.10, 0.60),
                ".*_calf_joint": (-2.4435, 0.0),
            },
            sim_target_rate_limit_range=(1.9, 2.1),
            sim_target_accel_limit_range=(80.0, 140.0),
            sim_torque_budget_range=(7.0, 10.0),
            sim_short_peak_torque_range=(10.0, 14.0),
            sim_short_peak_prob=0.05,
            sim_motor_delay_steps_range=(0, 3),
            sim_motor_strength_scale_range=(0.95, 1.05),
            sim_kp=40.0,
            sim_kp_scale_range=(0.90, 1.10),
            sim_kd_scale_range=(0.90, 1.10),
        )
        self.observations = WaveObservationsCfg()

        # Wave gait owns contact timing; inherited diagonal-trot shaping must not compete with it.
        for name in (
            "gait", "phase_trot_foot_clearance", "phase_trot_swing_contact",
            "phase_trot_contact_pattern", "phase_trot_calf_flexion",
            "phase_diagonal_support", "phase_diagonal_support_switch",
            "diagonal_support_accuracy", "fl_rr_support_accuracy",
            "swing_foot_clearance", "rear_swing_foot_clearance",
            "swing_calf_flexion", "air_time_variance", "excessive_foot_air_time",
            "moving_few_contacts", "moving_too_many_contacts",
        ):
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.lin_vel_z_l2.weight = -8.0
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -1.0e-4
        self.rewards.dof_acc_l2.weight = -3.0e-6
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.action_l2.weight = 0.0
        self.rewards.q_ref_tracking = RewTerm(
            func=wave_rew.q_ref_tracking_penalty, weight=-0.15,
            params={"asset_cfg": JOINT_CFG, "deadzone": 0.04},
        )
        self.rewards.residual_magnitude = RewTerm(
            func=wave_rew.residual_magnitude_penalty, weight=-0.05
        )
        self.rewards.residual_rate = RewTerm(func=wave_rew.residual_rate_penalty, weight=-0.03)
        self.rewards.wave_swing_contact = RewTerm(
            func=wave_rew.wave_swing_contact_penalty, weight=-0.10, params={"sensor_cfg": FOOT_CFG}
        )
        self.rewards.wave_stance_loss = RewTerm(
            func=wave_rew.wave_stance_contact_loss_penalty, weight=-0.05, params={"sensor_cfg": FOOT_CFG}
        )
        self.rewards.filter_tracking_error = RewTerm(func=wave_rew.filter_tracking_error, weight=-0.02)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.18
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.15)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.curriculum.auto_speed = CurrTerm(
            func=wave_curriculum,
            params={"num_steps_per_iter": 24, "stages": WAVE_CURRICULUM_STAGES},
        )

        self.events.add_base_mass = EventTerm(
            func=base_mdp.randomize_rigid_body_mass, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "mass_distribution_params": (0.0, 0.0), "operation": "add",
            },
        )
        self.events.rs01_actuator_gains.mode = "reset"
        self.events.rs01_actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        self.events.rs01_actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        self.events.rs01_joint_properties.mode = "reset"
        self.events.rs01_joint_properties.params["friction_distribution_params"] = (0.08, 0.08)
        self.events.rs01_joint_properties.params["armature_distribution_params"] = (0.010, 0.010)
        # The staged profiles do not include COM randomization; leaving the
        # inherited startup event enabled would contaminate deterministic Stage 1.
        self.events.base_com = None
        self.events.push_robot = EventTerm(
            func=stage_gated_push, mode="interval", interval_range_s=(12.0, 18.0),
            params={
                "velocity_range": {"x": (-0.20, 0.20), "y": (-0.15, 0.15)},
                "minimum_stage": 4,
            },
        )
        self.events.reset_base.params["pose_range"].update({"roll": (-0.01, 0.01), "pitch": (-0.01, 0.01)})


@configclass
class FanfanRlCpgResidualFlatEnvCfg_PLAY(FanfanRlCpgResidualFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.curriculum.auto_speed = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.rs01_actuator_gains = None
        self.events.rs01_joint_properties = None
        self.events.base_com = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        # 0.15 selects the full 0.038 m / 0.62 Hz gait used by the real-machine node.
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)


@configclass
class FanfanRlCpgResidualReferenceEnvCfg(FanfanRlCpgResidualFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0
        self.actions.joint_pos.action_mode = "reference_rate_limit"
        self.actions.joint_pos.enable_target_rate_limit = True
        self.actions.joint_pos.enable_torque_target_limit = False
        self.actions.joint_pos.enable_target_accel_limit = False
        self.actions.joint_pos.enable_action_delay = False
        self.actions.joint_pos.fixed_delay_steps = 0
        self.actions.joint_pos.sim_motor_delay_steps_range = (0, 2)
        self.actions.joint_pos.sim_target_rate_limit_range = (2.1, 2.1)
        self.actions.joint_pos.sim_torque_budget_range = (10.0, 10.0)
        self.actions.joint_pos.sim_short_peak_torque_range = (10.0, 10.0)
        self.actions.joint_pos.sim_short_peak_prob = 0.0
        self.actions.joint_pos.sim_motor_strength_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kp_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kd_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_target_accel_limit_range = (140.0, 140.0)
        self.actions.joint_pos.hip_err_limit_mul = 1.0
        self.actions.joint_pos.thigh_err_limit_mul = 1.0
        self.actions.joint_pos.calf_err_limit_mul = 1.0
        self.actions.joint_pos.hip_target_rate_mul = 1.0
        self.actions.joint_pos.thigh_target_rate_mul = 1.0
        self.actions.joint_pos.calf_target_rate_mul = 1.0
        self.actions.joint_pos.hip_target_accel_mul = 1.0
        self.actions.joint_pos.thigh_target_accel_mul = 1.0
        self.actions.joint_pos.calf_target_accel_mul = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }


@configclass
class FanfanRlCpgResidualReferenceRawEnvCfg(FanfanRlCpgResidualReferenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "reference_raw"
        self.actions.joint_pos.enable_deploy_target_filter = False
        self.actions.joint_pos.enable_target_rate_limit = False
        self.actions.joint_pos.enable_target_accel_limit = False
        self.actions.joint_pos.enable_torque_target_limit = False
        self.actions.joint_pos.enable_action_delay = False
        self.actions.joint_pos.fixed_delay_steps = 0
        self.actions.joint_pos.reference_cfg.warmup_sec = 1.0
        self.actions.joint_pos.reference_cfg.step_hz = 0.62
        self.actions.joint_pos.reference_cfg.stride_length = 0.038
        self.actions.joint_pos.reference_cfg.swing_height = 0.072
        self.actions.joint_pos.reference_cfg.duty_factor = 0.78
        self.actions.joint_pos.clip = None
        self.terminations.time_out = None
        self.terminations.base_contact = None
        self.terminations.low_base = None
        self.terminations.bad_orientation = None
        reward_names = set(getattr(self.rewards, "__dataclass_fields__", {}))
        reward_names.update(name for name in vars(self.rewards) if not name.startswith("_"))
        for reward_name in reward_names:
            setattr(self.rewards, reward_name, None)


@configclass
class FanfanRlCpgResidualReferenceRateEnvCfg(FanfanRlCpgResidualReferenceEnvCfg):
    pass


@configclass
class FanfanRlCpgResidualReferenceTorqueMonitorEnvCfg(FanfanRlCpgResidualReferenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "reference_torque_monitor"


@configclass
class FanfanRlCpgResidualReferenceTorqueClipEnvCfg(FanfanRlCpgResidualReferenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "reference_torque_clip"
        self.actions.joint_pos.enable_torque_target_limit = True


@configclass
class FanfanRlCpgResidualReferenceDelayEnvCfg(FanfanRlCpgResidualReferenceTorqueClipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "reference_delay"
        self.actions.joint_pos.enable_action_delay = True


@configclass
class FanfanRlCpgResidualReferenceFilteredEnvCfg(FanfanRlCpgResidualReferenceDelayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "reference_filtered"
        self.actions.joint_pos.enable_target_accel_limit = True


@configclass
class FanfanRlCpgResidualJointMappingEnvCfg(FanfanRlCpgResidualReferenceRawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "joint_mapping_debug"
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)


@configclass
class FanfanRlCpgResidualCsvPlaybackEnvCfg(FanfanRlCpgResidualReferenceRawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "csv_playback"
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)


@configclass
class FanfanRlCpgResidualSmallHighFreqEnvCfg(FanfanRlCpgResidualFlatEnvCfg):
    """Dormant residual-training entry point using the small high-frequency reference."""

    def __post_init__(self):
        super().__post_init__()
        _set_rear_stand_pose(
            self.scene.robot,
            SMALL_HIGH_FREQ_REAR_THIGH,
            SMALL_HIGH_FREQ_REAR_CALF,
        )
        self.actions.joint_pos.reference_cfg = FanfanSmallHighFreqReferenceGaitCfg(
            thigh_length=HEAVY_FANFAN_MODEL.thigh_length_m,
            calf_length=HEAVY_FANFAN_MODEL.calf_length_m,
        )


@configclass
class FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg(FanfanRlCpgResidualReferenceRawEnvCfg):
    """Deterministic small high-frequency reference task with Stage-1 debug limits."""

    def __post_init__(self):
        super().__post_init__()
        _set_rear_stand_pose(
            self.scene.robot,
            SMALL_HIGH_FREQ_REAR_THIGH,
            SMALL_HIGH_FREQ_REAR_CALF,
        )
        self.actions.joint_pos.action_mode = "reference_stage"
        self.actions.joint_pos.reference_cfg = FanfanSmallHighFreqReferenceGaitCfg(
            thigh_length=HEAVY_FANFAN_MODEL.thigh_length_m,
            calf_length=HEAVY_FANFAN_MODEL.calf_length_m,
        )
        self.actions.joint_pos.control_stage = 1
        self.actions.joint_pos.enable_vmc = False
        self.actions.joint_pos.vmc_mode = "off"
        self.actions.joint_pos.enable_deploy_target_filter = True
        self.actions.joint_pos.enable_target_rate_limit = True
        self.actions.joint_pos.enable_target_accel_limit = True
        self.actions.joint_pos.enable_torque_target_limit = True
        self.actions.joint_pos.enable_action_delay = False
        self.actions.joint_pos.fixed_delay_steps = 0
        self.actions.joint_pos.sim_target_rate_limit_range = (10.0, 10.0)
        self.actions.joint_pos.sim_target_accel_limit_range = (240.0, 240.0)
        self.actions.joint_pos.sim_torque_budget_range = (12.0, 12.0)
        self.actions.joint_pos.sim_short_peak_torque_range = (12.0, 12.0)
        self.actions.joint_pos.sim_short_peak_prob = 0.0
        self.actions.joint_pos.sim_motor_strength_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kp_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kd_scale_range = (1.0, 1.0)
        self.actions.joint_pos.hip_err_limit_mul = 1.0
        self.actions.joint_pos.thigh_err_limit_mul = 1.0
        self.actions.joint_pos.calf_err_limit_mul = 1.0
        self.actions.joint_pos.hip_target_rate_mul = 1.0
        self.actions.joint_pos.thigh_target_rate_mul = 1.0
        self.actions.joint_pos.calf_target_rate_mul = 1.0
        self.actions.joint_pos.hip_target_accel_mul = 1.0
        self.actions.joint_pos.thigh_target_accel_mul = 1.0
        self.actions.joint_pos.calf_target_accel_mul = 1.0
        self.actions.joint_pos.clip = None


@configclass
class FanfanRlCpgResidualSmallHighFreqStage0ReferenceEnvCfg(
    FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.control_stage = 0
        self.actions.joint_pos.enable_deploy_target_filter = False
        self.actions.joint_pos.enable_target_rate_limit = False
        self.actions.joint_pos.enable_target_accel_limit = False
        self.actions.joint_pos.enable_torque_target_limit = False


@configclass
class FanfanRlCpgResidualSmallHighFreqStage1ReferenceEnvCfg(
    FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg
):
    """Compatibility alias for the Stage-1 debug profile."""

    pass


@configclass
class FanfanRlCpgResidualSmallHighFreqStage1DebugReferenceEnvCfg(
    FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg
):
    """Simulation-only profile for checking whether the gait can execute."""

    pass


@configclass
class FanfanRlCpgResidualSmallHighFreqStage1SafeReferenceEnvCfg(
    FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg
):
    """Conservative 5 rad/s and 6 N.m profile for hardware proximity checks."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.sim_target_rate_limit_range = (5.0, 5.0)
        self.actions.joint_pos.sim_target_accel_limit_range = (180.0, 180.0)
        self.actions.joint_pos.sim_torque_budget_range = (6.0, 6.0)
        self.actions.joint_pos.sim_short_peak_torque_range = (6.0, 6.0)


@configclass
class FanfanRlCpgResidualRearLiftTestEnvCfg(
    FanfanRlCpgResidualSmallHighFreqStage0ReferenceEnvCfg
):
    """Reference-only isolated RR/RL lift test with configurable rear stand pose."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "rear_lift_test"
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)


@configclass
class FanfanRlCpgResidualPressSignTestEnvCfg(
    FanfanRlCpgResidualSmallHighFreqStage0ReferenceEnvCfg
):
    """Measure the contact-force response to +/-10 mm foot-z commands."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "press_sign_test"
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)


@configclass
class FanfanRlCpgResidualBodyShiftSweepEnvCfg(
    FanfanRlCpgResidualSmallHighFreqStage0ReferenceEnvCfg
):
    """Sweep body x/y shifts while holding all four feet in stance."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "body_shift_sweep"
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)


@configclass
class FanfanRlCpgResidualRearLiftFixedBaseTestEnvCfg(
    FanfanRlCpgResidualRearLiftTestEnvCfg
):
    """Fixed-Trunk rear lift test that isolates IK and joint semantics."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.spawn.usd_file_name = (
            "fanfan_mass_scaled_only_trunk_plus_800g_fixed_base.usd"
        )
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.actions.joint_pos.rear_lift_diagonal_front_preload_m = 0.0
        self.actions.joint_pos.rear_lift_same_front_preload_m = 0.0
        self.actions.joint_pos.rear_lift_other_rear_preload_m = 0.0
        self.actions.joint_pos.rear_lift_target_unload_m = 0.0
        self.actions.joint_pos.rear_lift_body_shift_x_m = 0.0
        self.actions.joint_pos.rear_lift_body_shift_y_m = 0.0
        self.actions.joint_pos.rear_lift_force_drop_threshold_n = 1.0e9


@configclass
class FanfanRlCpgResidualFastDiagonalTrotReferenceEnvCfg(
    FanfanRlCpgResidualSmallHighFreqStage0ReferenceEnvCfg
):
    """Reference-only diagonal trot that uses FR+RL and FL+RR swing pairs."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.action_mode = "fast_diagonal_trot"
        # Mid gains are for simulation morphology checks, not hardware-default safety validation.
        self.actions.joint_pos.fast_trot_swing_hip_kp = 50.0
        self.actions.joint_pos.fast_trot_swing_thigh_kp = 80.0
        self.actions.joint_pos.fast_trot_swing_calf_kp = 80.0
        self.actions.joint_pos.fast_trot_swing_kd = 4.5
        self.actions.joint_pos.fast_trot_support_hip_kp = 70.0
        self.actions.joint_pos.fast_trot_support_thigh_kp = 160.0
        self.actions.joint_pos.fast_trot_support_calf_kp = 180.0
        self.actions.joint_pos.fast_trot_support_kd = 5.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)


@configclass
class FanfanRlCpgResidualFastDiagonalTrotSafeReferenceEnvCfg(
    FanfanRlCpgResidualFastDiagonalTrotReferenceEnvCfg
):
    """FastDiagonalTrot with the deploy target filter enabled for safety-chain checks."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.enable_deploy_target_filter = True
        self.actions.joint_pos.enable_target_rate_limit = True
        self.actions.joint_pos.enable_target_accel_limit = True
        self.actions.joint_pos.enable_torque_target_limit = True
        self.actions.joint_pos.enable_action_delay = False
        self.actions.joint_pos.fixed_delay_steps = 0
        self.actions.joint_pos.sim_target_rate_limit_range = (10.0, 10.0)
        self.actions.joint_pos.sim_target_accel_limit_range = (240.0, 240.0)
        self.actions.joint_pos.sim_torque_budget_range = (8.0, 8.0)
        self.actions.joint_pos.sim_short_peak_torque_range = (12.0, 12.0)
        self.actions.joint_pos.sim_short_peak_prob = 0.0
        self.actions.joint_pos.sim_hard_torque_budget = 17.0
        self.actions.joint_pos.sim_motor_strength_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kp_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kd_scale_range = (1.0, 1.0)
        self.actions.joint_pos.hip_err_limit_mul = 1.0
        self.actions.joint_pos.thigh_err_limit_mul = 1.0
        self.actions.joint_pos.calf_err_limit_mul = 1.0
        self.actions.joint_pos.hip_target_rate_mul = 1.0
        self.actions.joint_pos.thigh_target_rate_mul = 1.0
        self.actions.joint_pos.calf_target_rate_mul = 1.0
        self.actions.joint_pos.hip_target_accel_mul = 1.0
        self.actions.joint_pos.thigh_target_accel_mul = 1.0
        self.actions.joint_pos.calf_target_accel_mul = 1.0
        # real_safe: conservative enough to test hardware proximity without hiding Kp/err-limit mismatch.
        self.actions.joint_pos.fast_trot_swing_hip_kp = 40.0
        self.actions.joint_pos.fast_trot_swing_thigh_kp = 70.0
        self.actions.joint_pos.fast_trot_swing_calf_kp = 70.0
        self.actions.joint_pos.fast_trot_swing_kd = 4.2
        self.actions.joint_pos.fast_trot_support_hip_kp = 60.0
        self.actions.joint_pos.fast_trot_support_thigh_kp = 120.0
        self.actions.joint_pos.fast_trot_support_calf_kp = 140.0
        self.actions.joint_pos.fast_trot_support_kd = 5.0


@configclass
class FanfanRlCpgResidualSmallHighFreqStage2ReferenceEnvCfg(
    FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.control_stage = 2
        self.actions.joint_pos.enable_vmc = True
        self.actions.joint_pos.vmc_mode = "light"
