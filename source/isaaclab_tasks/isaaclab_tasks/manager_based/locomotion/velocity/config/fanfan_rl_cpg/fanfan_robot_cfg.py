import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .rs01_motor_params import (
    RS01_ARMATURE,
    RS01_FRICTION,
    RS01_KD,
    RS01_KP,
    RS01_PEAK_TORQUE,
    RS01_VELOCITY_LIMIT,
)


FANFAN_JOINT_NAMES = [
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
]

# Default stand pose converted from the real RS01 motion_mode_run values in
# real motor order FR, FL, RL, RR into the policy/simulation joint convention
# used here: FR, FL, RR, RL.  Keep this aligned with deployment
# stand_pose_source=policy_default and semantic_mapper signs.
FANFAN_TEXT_STAND_JOINT_POS = {
    "FR_hip_joint": -0.1571,
    "FR_thigh_joint": 0.3491,
    "FR_calf_joint": -0.7854,
    "FL_hip_joint": 0.1571,
    "FL_thigh_joint": 0.3491,
    "FL_calf_joint": -0.7854,
    "RR_hip_joint": -0.1571,
    "RR_thigh_joint": 0.2269,
    "RR_calf_joint": -0.3491,
    "RL_hip_joint": 0.1571,
    "RL_thigh_joint": 0.2269,
    "RL_calf_joint": -0.3491,
}


def _resolve_fanfan_urdf_path() -> str:
    env_path = os.environ.get("FANFAN_URDF_PATH")
    if env_path:
        return env_path

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "fanfan" / "urdf" / "fanfan.urdf"
        if candidate.exists():
            return str(candidate)

    return str(current_file.parents[10] / "fanfan" / "urdf" / "fanfan.urdf")


def _resolve_fanfan_usd_dir() -> str:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "fanfan" / "USD"
        if candidate.exists():
            return str(candidate)

    return str(current_file.parents[10] / "fanfan" / "USD")


def _make_fanfan_leg_actuator_cfg():
    # This training pass intentionally disables the learned actuator network.
    # Use the RS01 manual peak torque as the simulator hard limit, while rewards
    # separately discourage long-term use above the 6 N*m continuous rating.
    return ImplicitActuatorCfg(
        joint_names_expr=FANFAN_JOINT_NAMES,
        effort_limit_sim=RS01_PEAK_TORQUE,
        velocity_limit_sim=RS01_VELOCITY_LIMIT,
        stiffness=RS01_KP,
        damping=RS01_KD,
        # Engineering initial values.  They are not in the RS01 manual and need
        # later identification from real joint step/velocity response.
        armature=RS01_ARMATURE,
        friction=RS01_FRICTION,
    )


FANFAN_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_resolve_fanfan_urdf_path(),
        usd_dir=_resolve_fanfan_usd_dir(),
        usd_file_name="fanfan_no_merge.usd",
        fix_base=False,
        # Fixed-joint links such as *_foot must stay as real runtime articulation bodies.
        merge_fixed_joints=False,
        # Instanceable USDs can hide fixed-joint helper links from runtime sensors/views.
        make_instanceable=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=RS01_KP,
                damping=RS01_KD,
            ),
            target_type="position",
            drive_type="force",
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Match the real stand script from /home/jetson/text.  This makes
        # mdp.joint_pos_rel zero at the same mechanical posture used on RS01.
        pos=(0.0, 0.0, 0.293),
        joint_pos=FANFAN_TEXT_STAND_JOINT_POS,
    ),
    actuators={"legs": _make_fanfan_leg_actuator_cfg()},
)
