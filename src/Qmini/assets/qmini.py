"""Qmini Q1 biped from the project-owned URDF, for PhysX and Newton.

Effort and velocity limits match q1.urdf. PD gains are initial simulation
tuning values, not motor specifications or validated hardware gains.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from . import QMINI_ASSETS_DIR

# Explicit task/action ordering; the simulator's internal joint order may differ.
JOINT_NAMES = [
    "hip_yaw_l",
    "hip_roll_l",
    "hip_pitch_l",
    "knee_pitch_l",
    "ankle_pitch_l",
    "hip_yaw_r",
    "hip_roll_r",
    "hip_pitch_r",
    "knee_pitch_r",
    "ankle_pitch_r",
]
FIXED_JOINT_NAMES = ["imu_in_torso_joint"]
RIGID_BODY_NAMES = ["base_link", "imu_in_torso", *JOINT_NAMES]
FEET_BODY_NAMES = ["ankle_pitch_l", "ankle_pitch_r"]

QMINI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(QMINI_ASSETS_DIR / "Qmini" / "urdf" / "q1.urdf"),
        fix_base=False,
        activate_contact_sensors=True,
        merge_fixed_joints=False,
        collision_from_visuals=False,
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
        # Zero-pose collision meshes extend 0.42874 m below base_link.
        pos=(0.0, 0.0, 0.45),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_roll": IdealPDActuatorCfg(
            joint_names_expr=["hip_roll_[lr]"],
            actuator_effort_limit=60.0,
            actuator_velocity_limit=10.0,
            joint_effort_limit=60.0,
            joint_velocity_limit=10.0,
            stiffness=80.0,
            damping=2.0,
        ),
        "hip_yaw": IdealPDActuatorCfg(
            joint_names_expr=["hip_yaw_[lr]"],
            actuator_effort_limit=20.0,
            actuator_velocity_limit=30.0,
            joint_effort_limit=20.0,
            joint_velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
        ),
        "leg_pitch": IdealPDActuatorCfg(
            joint_names_expr=["hip_pitch_[lr]", "knee_pitch_[lr]", "ankle_pitch_[lr]"],
            actuator_effort_limit=20.0,
            actuator_velocity_limit=30.0,
            joint_effort_limit=20.0,
            joint_velocity_limit=30.0,
            stiffness={"hip_pitch_[lr]": 40.0, "knee_pitch_[lr]": 40.0, "ankle_pitch_[lr]": 20.0},
            damping={"hip_pitch_[lr]": 1.0, "knee_pitch_[lr]": 1.0, "ankle_pitch_[lr]": 0.8},
        ),
    },
)
