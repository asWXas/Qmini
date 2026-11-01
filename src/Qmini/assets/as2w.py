"""Unitree AS2W wheeled quadruped using the upstream URDF.

Limits come from the URDF. PD gains and the crouched initial pose are
simulation starting values, not validated hardware parameters.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from . import QMINI_ASSETS_DIR

AS2W_ASSET_DIR = QMINI_ASSETS_DIR / "as2w_description"
# Explicit action ordering; simulator joint ordering may differ.
LEG_PREFIXES = ("FL", "FR", "RL", "RR")
LEG_JOINT_NAMES = [f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in ("hip", "thigh", "calf")]
WHEEL_JOINT_NAMES = [f"{leg}_foot" for leg in LEG_PREFIXES]
JOINT_NAMES = [f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in ("hip", "thigh", "calf", "foot")]
WHEEL_BODY_NAMES = WHEEL_JOINT_NAMES.copy()
RIGID_BODY_NAMES = ["base_link", *JOINT_NAMES, "JT128"]
FIXED_JOINT_NAMES = ["JT128"]

AS2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(AS2W_ASSET_DIR / "as2w.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        collision_from_visuals=False,
        force_usd_conversion=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*_hip": 0.0, ".*_thigh": 0.8, ".*_calf": -1.6, ".*_foot": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips_and_thighs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip", ".*_thigh"],
            actuator_effort_limit=60.0,
            actuator_velocity_limit=25.0,
            joint_effort_limit=60.0,
            joint_velocity_limit=25.0,
            stiffness=60.0,
            damping=2.0,
        ),
        "calves": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf"],
            actuator_effort_limit=90.0,
            actuator_velocity_limit=16.67,
            joint_effort_limit=90.0,
            joint_velocity_limit=16.67,
            stiffness=60.0,
            damping=2.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_foot"],
            actuator_effort_limit=15.0,
            actuator_velocity_limit=98.47,
            joint_effort_limit=15.0,
            joint_velocity_limit=98.47,
            stiffness=0.0,
            damping=0.5,
        ),
    },
)
