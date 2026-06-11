from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import mdp_rewards as fanfan_mdp
from .rough_env_cfg import FanfanA1CleanRoughEnvCfg
from .rs01_motor_params import RS01_CONTINUOUS_TORQUE, RS01_KD, RS01_KP, RS01_PEAK_TORQUE


# Zero-weight diagnostics for checking whether foot body height/contact data
# matches the visual foot trajectory.  If TensorBoard only shows weighted zeros,
# temporarily set one debug term weight to 1e-6 or scale the returned value.
ENABLE_FOOT_DEBUG_METRICS = True


HEAVY_ROBOT_AUTO_SPEED_STAGES = (
    {
        "stage": 1,
        "start_iter": 0,
        "end_iter": 10_000,
        "lin_vel_x": (0.05, 0.10),
        "rel_standing_envs": 0.25,
    },
    {
        "stage": 2,
        "start_iter": 10_000,
        "end_iter": 30_000,
        "lin_vel_x": (0.05, 0.18),
        "rel_standing_envs": 0.15,
    },
    {
        "stage": 3,
        "start_iter": 30_000,
        "end_iter": 60_000,
        "lin_vel_x": (0.05, 0.27),
        "rel_standing_envs": 0.08,
    },
    {
        "stage": 4,
        "start_iter": 60_000,
        "end_iter": None,
        "lin_vel_x": (0.05, 0.35),
        "rel_standing_envs": 0.05,
    },
)


@configclass
class FanfanA1CleanFlatEnvCfg(FanfanA1CleanRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Rebalance actuator availability for the corrected 7.24 kg URDF.
        self.actions.joint_pos.sim_torque_budget_range = (7.0, 12.0)
        self.actions.joint_pos.sim_short_peak_torque_range = (12.0, 17.0)
        self.actions.joint_pos.sim_short_peak_prob = 0.05
        self.actions.joint_pos.sim_motor_strength_scale_range = (0.85, 1.05)

        # 低能耗和平滑项使用保守初值；先让策略学稳定对称步态，再逐步加大速度/随机化。
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.feet_air_time.weight = 0.08
        self.rewards.feet_air_time.params["threshold"] = 0.08
        # Old GaitReward only rewards diagonal contact-time sync and can
        # encourage symmetric dragging.  The training line now uses the
        # phase-based trot rewards inherited from Rough.
        self.rewards.gait.weight = 0.0
        self.rewards.gait.params["std"] = 0.14
        self.rewards.gait.params["max_err"] = 0.24
        self.rewards.phase_trot_foot_clearance.weight = -2.0
        self.rewards.phase_trot_swing_contact.weight = -0.8
        self.rewards.phase_trot_contact_pattern.weight = -0.9
        self.rewards.phase_trot_calf_flexion.weight = -0.5
        self.rewards.air_time_variance.weight = -0.2
        self.rewards.excessive_foot_air_time.weight = -4.0
        self.rewards.excessive_foot_air_time.params["max_air_time"] = 0.16
        self.rewards.swing_foot_clearance.weight = -4.0
        self.rewards.swing_foot_clearance.params["target_clearance"] = 0.065
        self.rewards.rear_swing_foot_clearance.weight = -4.0
        self.rewards.rear_swing_foot_clearance.params["target_clearance"] = 0.070
        self.rewards.swing_calf_flexion.weight = -0.6
        self.rewards.joint_target_error.weight = -1.0
        self.rewards.joint_target_error.params["threshold"] = 0.26
        self.rewards.estimated_pd_torque_limit.weight = 0.0
        self.rewards.estimated_pd_torque_limit.params["kp"] = RS01_KP
        self.rewards.estimated_pd_torque_limit.params["kd"] = RS01_KD
        self.rewards.estimated_pd_torque_limit.params["torque_limit"] = RS01_PEAK_TORQUE
        self.rewards.applied_torque_limit.weight = -0.01
        self.rewards.applied_torque_limit.params["torque_limit"] = RS01_PEAK_TORQUE
        # RS01 continuous rating is 6 N*m.  The simulator may peak at 17 N*m,
        # but this term discourages using that peak as normal operating torque.
        self.rewards.continuous_torque.weight = -0.012
        self.rewards.continuous_torque.params["torque_reference"] = RS01_CONTINUOUS_TORQUE
        self.rewards.low_speed_high_torque.weight = -0.007
        self.rewards.low_speed_high_torque.params["torque_reference"] = RS01_CONTINUOUS_TORQUE
        self.rewards.power.weight = -0.005
        self.rewards.rear_foot_drag.weight = -1.5
        self.rewards.rear_long_contact.weight = -3.5
        self.rewards.rear_long_contact.params["max_contact_time"] = 0.20
        self.rewards.rear_calf_fold.weight = -2.0
        self.rewards.rear_calf_fold.params["threshold"] = 0.18
        self.rewards.rear_thigh_low.weight = -2.0
        self.rewards.rear_thigh_low.params["threshold"] = 0.14
        self.rewards.rear_leg_length.weight = -1.5
        self.rewards.front_calf_fold.weight = -6.0
        self.rewards.front_calf_fold.params["threshold"] = 0.12
        self.rewards.front_thigh_low.weight = -4.0
        self.rewards.front_thigh_low.params["threshold"] = 0.10
        self.rewards.thigh_front_rear_balance.weight = -2.0
        self.rewards.calf_front_rear_balance.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -10.0
        self.rewards.ang_vel_xy_l2.weight = -1.5
        self.rewards.dof_torques_l2.weight = -1.0e-4
        # Stage-1C: make raw action size/rate visibly expensive so the ONNX
        # does not enter hardware with large saturated actions at cmd=0.
        self.rewards.action_l2 = RewTerm(func=base_mdp.action_l2, weight=-0.004)
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.dof_acc_l2.weight = -3.0e-6
        self.rewards.base_height = RewTerm(
            func=base_mdp.base_height_l2,
            weight=-60.0,
            params={
                "target_height": 0.293,
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            },
        )
        self.rewards.joint_deviation = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.02)
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        self.rewards.moving_few_contacts = RewTerm(
            func=fanfan_mdp.moving_few_contacts_penalty,
            weight=-0.8,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "min_contacts": 2.0,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        self.rewards.moving_too_many_contacts = RewTerm(
            func=fanfan_mdp.moving_too_many_contacts_penalty,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "max_contacts": 3.0,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_threshold": 0.03,
            },
        )
        # EXPERIMENT B: zero-weight diagnostic terms for foot-link height and
        # contact state.  If TensorBoard only shows weighted rewards as zeros,
        # set one diagnostic weight to 1e-6 temporarily for manual inspection.
        if ENABLE_FOOT_DEBUG_METRICS:
            for foot_name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
                foot_key = foot_name.replace("_foot", "").lower()
                for metric in ("height", "contact", "air_time", "contact_time"):
                    setattr(
                        self.rewards,
                        f"debug_{foot_key}_{metric}",
                        RewTerm(
                            func=fanfan_mdp.debug_foot_height_contact_metric,
                            weight=0.0,
                            params={
                                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_name),
                                "asset_cfg": SceneEntityCfg("robot", body_names=foot_name),
                                "metric": metric,
                                "contact_threshold": 1.0,
                            },
                        ),
                    )
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-24.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
                "threshold": 1.0,
            },
        )
        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.terminations.low_base = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.16, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 0.9, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Initial command distribution is Stage 1.  The auto_speed curriculum
        # expands this during training, while keeping standing envs alive.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.10)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.curriculum.auto_speed.params["stages"] = HEAVY_ROBOT_AUTO_SPEED_STAGES


class FanfanA1CleanFlatEnvCfg_PLAY(FanfanA1CleanFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.rs01_actuator_gains = None
        self.events.rs01_joint_properties = None
        self.curriculum.auto_speed = None
        # Use a deterministic walking command for checkpoint evaluation.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)
