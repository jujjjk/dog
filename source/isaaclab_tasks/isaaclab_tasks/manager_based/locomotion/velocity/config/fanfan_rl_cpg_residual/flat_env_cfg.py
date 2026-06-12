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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.flat_env_cfg import (
    FanfanA1CleanFlatEnvCfg,
)

from . import mdp_observations as wave_obs
from . import mdp_rewards as wave_rew
from .curriculum import WAVE_CURRICULUM_STAGES, stage_gated_push, wave_curriculum
from .joint_semantics import SIM_JOINT_NAMES
from .residual_action import WaveResidualJointPositionActionCfg


JOINT_NAMES = list(SIM_JOINT_NAMES)
FOOT_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
    preserve_order=True,
)
JOINT_CFG = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


@configclass
class WavePolicyCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=wave_obs.ordered_joint_pos_rel,
        params={"asset_cfg": JOINT_CFG},
        noise=Unoise(n_min=-0.005, n_max=0.005),
    )
    joint_vel = ObsTerm(
        func=wave_obs.ordered_joint_vel,
        params={"asset_cfg": JOINT_CFG},
        noise=Unoise(n_min=-0.15, n_max=0.15),
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
            sim_motor_delay_steps_range=(0, 2),
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
        self.commands.base_velocity.rel_standing_envs = 0.35
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.05)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.curriculum.auto_speed = CurrTerm(
            func=wave_curriculum,
            params={"num_steps_per_iter": 24, "stages": WAVE_CURRICULUM_STAGES},
        )

        self.events.add_base_mass = EventTerm(
            func=base_mdp.randomize_rigid_body_mass, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "mass_distribution_params": (-0.20, 0.20), "operation": "add",
            },
        )
        self.events.push_robot = EventTerm(
            func=stage_gated_push, mode="interval", interval_range_s=(12.0, 18.0),
            params={
                "velocity_range": {"x": (-0.20, 0.20), "y": (-0.15, 0.15)},
                "minimum_stage": 4,
            },
        )
        self.events.reset_base.params["pose_range"].update({"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05)})


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
        self.actions.joint_pos.action_mode = "reference_only"
        self.actions.joint_pos.sim_motor_delay_steps_range = (0, 0)
        self.actions.joint_pos.sim_target_rate_limit_range = (2.1, 2.1)
        self.actions.joint_pos.sim_torque_budget_range = (10.0, 10.0)
        self.actions.joint_pos.sim_short_peak_torque_range = (10.0, 10.0)
        self.actions.joint_pos.sim_short_peak_prob = 0.0
        self.actions.joint_pos.sim_motor_strength_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kp_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_kd_scale_range = (1.0, 1.0)
        self.actions.joint_pos.sim_target_accel_limit_range = (140.0, 140.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
