from isaaclab.envs import mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .fanfan_robot_cfg import FANFAN_CFG
from .deploy_actions import DeployFilteredJointPositionActionCfg
from .auto_curriculum import auto_speed_curriculum
from . import mdp_observations as fanfan_obs
from . import mdp_rewards as fanfan_mdp
from .rs01_motor_params import (
    RS01_ACTION_SCALE_NORMAL,
    RS01_ACTION_SCALE_SAFE,
    RS01_ARMATURE_RANGE,
    RS01_CONTINUOUS_TORQUE,
    RS01_DAMPING_SCALE_RANGE,
    RS01_DEPLOY_SHORT_PEAK_TORQUE_RANGE,
    RS01_DEPLOY_TORQUE_BUDGET_RANGE,
    RS01_JOINT_FRICTION_RANGE,
    RS01_KD,
    RS01_KP,
    RS01_PEAK_TORQUE,
    RS01_RATED_VELOCITY,
    RS01_MOTOR_STRENGTH_SCALE_RANGE,
    RS01_STIFFNESS_SCALE_RANGE,
)


@configclass
class FanfanA1CleanRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FANFAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

        # Keep the A1 structure and only scale terrain difficulty to the smaller frame.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.015, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.005

        # RS01 deployment-aligned action path.  The policy still outputs the
        # same 12 actions, but training targets now pass through a limiter close
        # to the real SafeTargetLimiter before reaching the simulator PD drive.
        self.actions.joint_pos = DeployFilteredJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            use_default_offset=True,
            scale={
                "FR_hip_joint": 0.12,
                "FL_hip_joint": 0.12,
                "RR_hip_joint": 0.12,
                "RL_hip_joint": 0.12,
                "FR_thigh_joint": 0.22,
                "FL_thigh_joint": 0.22,
                "RR_thigh_joint": 0.26,
                "RL_thigh_joint": 0.26,
                "FR_calf_joint": 0.30,
                "FL_calf_joint": 0.30,
                "RR_calf_joint": 0.42,
                "RL_calf_joint": 0.42,
            },
            clip={
                "FR_hip_joint": (-0.16, 0.08),
                "FL_hip_joint": (-0.08, 0.16),
                "RR_hip_joint": (-0.16, 0.08),
                "RL_hip_joint": (-0.08, 0.16),
                "FR_thigh_joint": (-1.5708, 0.6458),
                "FL_thigh_joint": (-1.5708, 0.6458),
                "RR_thigh_joint": (-0.10, 0.60),
                "RL_thigh_joint": (-0.10, 0.60),
                ".*_calf_joint": (-2.4435, 0.0),
            },
            enable_deploy_target_filter=True,
            sim_target_rate_limit_range=(2.0, 3.0),
            sim_target_accel_limit_range=(60.0, 120.0),
            sim_torque_budget_range=RS01_DEPLOY_TORQUE_BUDGET_RANGE,
            sim_short_peak_torque_range=RS01_DEPLOY_SHORT_PEAK_TORQUE_RANGE,
            sim_short_peak_prob=0.05,
            sim_motor_delay_steps_range=(1, 3),
            sim_motor_strength_scale_range=RS01_MOTOR_STRENGTH_SCALE_RANGE,
            sim_kp=RS01_KP,
            sim_kp_scale_range=RS01_STIFFNESS_SCALE_RANGE,
            sim_kd_scale_range=RS01_DAMPING_SCALE_RANGE,
        )
        self.actions.joint_pos.scale = {
            "FR_hip_joint": 0.12,
            "FL_hip_joint": 0.12,
            "RR_hip_joint": 0.12,
            "RL_hip_joint": 0.12,
            "FR_thigh_joint": 0.22,
            "FL_thigh_joint": 0.22,
            "RR_thigh_joint": 0.26,
            "RL_thigh_joint": 0.26,
            "FR_calf_joint": 0.30,
            "FL_calf_joint": 0.30,
            "RR_calf_joint": 0.42,
            "RL_calf_joint": 0.42,
        }
        self.actions.joint_pos.clip = {
            "FR_hip_joint": (-0.16, 0.08),
            "FL_hip_joint": (-0.08, 0.16),
            "RR_hip_joint": (-0.16, 0.08),
            "RL_hip_joint": (-0.08, 0.16),
            "FR_thigh_joint": (-1.5708, 0.6458),
            "FL_thigh_joint": (-1.5708, 0.6458),
            # Conservative rear-thigh release experiment: verify whether rear
            # swing space is clip-limited.  If training collapses, revert to
            # (0.08, 0.55).
            "RR_thigh_joint": (-0.10, 0.60),
            "RL_thigh_joint": (-0.10, 0.60),
            ".*_calf_joint": (-2.4435, 0.0),
        }
        self.observations.policy.actions = ObsTerm(func=mdp.last_action)
        self.observations.policy.base_lin_vel = ObsTerm(
            func=fanfan_obs.base_lin_vel_deploy_corrupted,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "enable_randomization": True,
                "zero_prob": 0.15,
                "command_prob": 0.20,
                "noise_std": (0.08, 0.05, 0.02),
                "bias_range": (-0.05, 0.05),
                "delay_steps_range": (0, 3),
                "scale_range": (0.7, 1.2),
            },
        )
        # A simple gait clock lets the policy know which diagonal pair should
        # swing, instead of guessing phase from action history.
        self.observations.policy.gait_phase = ObsTerm(
            func=fanfan_mdp.gait_phase_obs,
            params={"gait_period": 0.55},
        )

        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 0.8)
        self.events.add_base_mass.params["asset_cfg"].body_names = "Trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "Trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # RS01 simple domain randomization.  These ranges are conservative
        # engineering priors; identify them with real step/velocity tests later.
        self.events.rs01_actuator_gains = EventTerm(
            func=base_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": RS01_STIFFNESS_SCALE_RANGE,
                "damping_distribution_params": RS01_DAMPING_SCALE_RANGE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.rs01_joint_properties = EventTerm(
            func=base_mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": RS01_JOINT_FRICTION_RANGE,
                "armature_distribution_params": RS01_ARMATURE_RANGE,
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        # Motor strength, target delay, rate/accel, and torque-budget
        # randomization live in DeployFilteredJointPositionActionCfg above.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com.params["asset_cfg"].body_names = "Trunk"
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
        }

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.08
        self.rewards.feet_air_time.params["threshold"] = 0.10
        self.rewards.gait = RewTerm(
            func=fanfan_mdp.GaitReward,
            # Old GaitReward only rewards diagonal contact-time sync and can
            # encourage symmetric dragging.  The training line now uses the
            # phase-based trot rewards below.
            weight=0.0,
            params={
                "std": 0.16,
                "max_err": 0.26,
                "velocity_threshold": 0.2,
                "synced_feet_pair_names": (("FR_foot", "RL_foot"), ("FL_foot", "RR_foot")),
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
            },
        )
        self.rewards.phase_trot_foot_clearance = RewTerm(
            func=fanfan_mdp.phase_trot_foot_clearance_reward,
            weight=-2.0,
            params={
                # Keep this order aligned with the reward's phase offsets:
                # FR/RL swing together, FL/RR swing half a cycle later.
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                ),
                "gait_period": 0.55,
                "swing_ratio": 0.45,
                "base_clearance": 0.025,
                "lift_height": 0.055,
                "stance_contact_penalty": 0.025,
                "contact_threshold": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.phase_trot_swing_contact = RewTerm(
            func=fanfan_mdp.phase_trot_swing_contact_penalty,
            weight=-0.8,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                ),
                "gait_period": 0.55,
                "swing_ratio": 0.45,
                "contact_threshold": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.phase_trot_contact_pattern = RewTerm(
            func=fanfan_mdp.phase_trot_contact_pattern_penalty,
            weight=-0.8,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                ),
                "gait_period": 0.55,
                "swing_ratio": 0.45,
                "contact_threshold": 1.0,
                "stance_miss_cost": 0.5,
                "swing_contact_cost": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.phase_trot_calf_flexion = RewTerm(
            func=fanfan_mdp.phase_trot_calf_flexion_penalty,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"],
                ),
                "target_calf_pos": (-0.95, -0.95, -0.72, -0.72),
                "gait_period": 0.55,
                "swing_ratio": 0.45,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.air_time_variance = RewTerm(
            func=fanfan_mdp.air_time_variance_penalty,
            weight=-0.3,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        self.rewards.excessive_foot_air_time = RewTerm(
            func=fanfan_mdp.excessive_foot_air_time_penalty,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "max_air_time": 0.24,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.swing_foot_clearance = RewTerm(
            func=fanfan_mdp.swing_foot_clearance_penalty,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_clearance": 0.065,
                "contact_threshold": 1.0,
                "min_air_time": 0.035,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.rear_swing_foot_clearance = RewTerm(
            func=fanfan_mdp.rear_swing_foot_clearance_penalty,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RR_foot", "RL_foot"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["RR_foot", "RL_foot"]),
                "target_clearance": 0.070,
                "contact_threshold": 1.0,
                "min_air_time": 0.025,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.swing_calf_flexion = RewTerm(
            func=fanfan_mdp.swing_calf_flexion_penalty,
            weight=-0.6,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"],
                ),
                # Stage-1B keeps the rear calf swing target reachable from the
                # straighter rear default pose, while front legs still fold more.
                "target_calf_pos": (-0.95, -0.95, -0.72, -0.72),
                "contact_threshold": 1.0,
                "min_air_time": 0.04,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.contact_foot_drag = RewTerm(
            func=fanfan_mdp.contact_foot_drag_penalty,
            weight=-0.4,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "contact_threshold": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.long_contact = RewTerm(
            func=fanfan_mdp.long_contact_penalty,
            weight=-1.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "max_contact_time": 0.32,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.rear_foot_drag = RewTerm(
            func=fanfan_mdp.contact_foot_drag_penalty,
            weight=-1.2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RR_foot", "RL_foot"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["RR_foot", "RL_foot"]),
                "contact_threshold": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.rear_long_contact = RewTerm(
            func=fanfan_mdp.long_contact_penalty,
            weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RR_foot", "RL_foot"]),
                "max_contact_time": 0.20,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.joint_target_error = RewTerm(
            func=fanfan_mdp.joint_target_tracking_error_penalty,
            weight=-0.8,
            params={
                "threshold": 0.26,
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.estimated_pd_torque_limit = RewTerm(
            func=fanfan_mdp.estimated_pd_torque_limit_penalty,
            weight=0.0,
            params={
                "kp": RS01_KP,
                "kd": RS01_KD,
                "torque_limit": RS01_PEAK_TORQUE,
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.applied_torque_limit = RewTerm(
            func=fanfan_mdp.applied_torque_limit_penalty,
            weight=-0.01,
            params={
                "torque_limit": RS01_PEAK_TORQUE,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.continuous_torque = RewTerm(
            func=fanfan_mdp.continuous_torque_penalty,
            weight=-0.02,
            params={
                "torque_reference": RS01_CONTINUOUS_TORQUE,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.low_speed_high_torque = RewTerm(
            func=fanfan_mdp.low_speed_high_torque_penalty,
            weight=-0.01,
            params={
                "torque_reference": RS01_CONTINUOUS_TORQUE,
                "velocity_reference": RS01_RATED_VELOCITY,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.power = RewTerm(
            func=fanfan_mdp.power_penalty,
            weight=-0.01,
            params={
                "torque_reference": RS01_CONTINUOUS_TORQUE,
                "velocity_reference": RS01_RATED_VELOCITY,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.rear_calf_fold = RewTerm(
            func=fanfan_mdp.rear_calf_fold_penalty,
            weight=-2.0,
            params={
                "threshold": 0.18,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["RR_calf_joint", "RL_calf_joint"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RR_foot", "RL_foot"]),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.rear_thigh_low = RewTerm(
            func=fanfan_mdp.rear_thigh_low_penalty,
            weight=-2.0,
            params={
                "threshold": 0.14,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh_joint", "RL_thigh_joint"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RR_foot", "RL_foot"]),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.rear_leg_length = RewTerm(
            func=fanfan_mdp.rear_leg_length_penalty,
            weight=-1.5,
            params={
                "thigh_threshold": 0.10,
                "calf_threshold": 0.12,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["RR_thigh_joint", "RR_calf_joint", "RL_thigh_joint", "RL_calf_joint"],
                ),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RR_foot", "RL_foot"]),
                "contact_threshold": 1.0,
            },
        )
        self.rewards.front_calf_fold = RewTerm(
            func=fanfan_mdp.joint_fold_below_default_penalty,
            weight=-5.0,
            params={
                "threshold": 0.12,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["FR_calf_joint", "FL_calf_joint"]),
            },
        )
        self.rewards.front_thigh_low = RewTerm(
            func=fanfan_mdp.joint_fold_below_default_penalty,
            weight=-3.0,
            params={
                "threshold": 0.10,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["FR_thigh_joint", "FL_thigh_joint"]),
            },
        )
        self.rewards.thigh_front_rear_balance = RewTerm(
            func=fanfan_mdp.front_rear_posture_balance_penalty,
            weight=-2.0,
            params={
                "front_asset_cfg": SceneEntityCfg("robot", joint_names=["FR_thigh_joint", "FL_thigh_joint"]),
                "rear_asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh_joint", "RL_thigh_joint"]),
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.calf_front_rear_balance = RewTerm(
            func=fanfan_mdp.front_rear_posture_balance_penalty,
            weight=-2.5,
            params={
                "front_asset_cfg": SceneEntityCfg("robot", joint_names=["FR_calf_joint", "FL_calf_joint"]),
                "rear_asset_cfg": SceneEntityCfg("robot", joint_names=["RR_calf_joint", "RL_calf_joint"]),
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.base_height = RewTerm(
            func=base_mdp.base_height_l2,
            weight=-60.0,
            params={
                "target_height": 0.293,
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-18.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
                "threshold": 1.0,
            },
        )
        self.rewards.joint_deviation = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.02)
        self.rewards.dof_torques_l2.weight = -2.0e-4
        # Stage-1C prioritizes conservative, real-motor-executable motion:
        # keep speed tracking useful, but make raw action size/rate expensive
        # enough that exported policies do not saturate on hardware.
        self.rewards.action_l2 = RewTerm(func=base_mdp.action_l2, weight=-0.008)
        self.rewards.track_lin_vel_xy_exp.weight = 0.85
        self.rewards.track_ang_vel_z_exp.weight = 0.50
        self.rewards.lin_vel_z_l2.weight = -12.0
        self.rewards.ang_vel_xy_l2.weight = -1.5
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.dof_acc_l2.weight = -6.0e-6

        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"
        self.terminations.low_base = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.16, "asset_cfg": SceneEntityCfg("robot")},
        )
        # Initial range is Stage 1.  auto_speed_curriculum expands it during
        # training while preserving standing environments in every stage.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.50
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.08)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.curriculum.auto_speed = CurrTerm(
            func=auto_speed_curriculum,
            params={
                "enabled": True,
                "command_name": "base_velocity",
                "num_steps_per_iter": 24,
                "print_on_stage_change": True,
            },
        )


@configclass
class FanfanA1CleanRoughEnvCfg_PLAY(FanfanA1CleanRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.observations.policy.base_lin_vel.params["enable_randomization"] = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.rs01_actuator_gains = None
        self.events.rs01_joint_properties = None
        self.curriculum.auto_speed = None
