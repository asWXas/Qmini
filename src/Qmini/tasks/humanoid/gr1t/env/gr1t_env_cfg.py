# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import Qmini.tasks.humanoid.gr1t.mdp as mdp

##
# Pre-defined configs
##
from ..mdp.terrains import ROUGH_TERRAINS_CFG  # isort: skip

from Qmini.assets.gr1t import ARM_JOINT_NAMES, GR1T2_CFG, JOINT_NAMES, LEG_JOINT_NAMES1, WAIST_JOINT_NAMES

# ==============================================================================
# 通用传感器子配置（复用避免重复定义）
# ==============================================================================

FOOT_RAYCASTER_CFG = RayCasterCfg(
    prim_path=MISSING,  # 实例化时动态替换为具体 foot prim # type: ignore
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)


# ==============================================================================
# 场景主配置
# ==============================================================================


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # --------------------------------------------------------------------------
    # 地形与环境 (Ground & Lighting)
    # --------------------------------------------------------------------------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    # --------------------------------------------------------------------------
    # 机器人资产 (Robot)
    # --------------------------------------------------------------------------
    robot: ArticulationCfg = GR1T2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # --------------------------------------------------------------------------
    # 传感器组 (Sensors)
    # --------------------------------------------------------------------------
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.2, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_left: RayCasterCfg = FOOT_RAYCASTER_CFG.replace( # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*/l_foot_roll"
    )
    height_scanner_right: RayCasterCfg = FOOT_RAYCASTER_CFG.replace( # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*/r_foot_roll"
    )

    # 接触力检测
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*",
        history_length=3,
        track_air_time=True,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.GroupedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        rel_standing_envs=0.05,
        # 移动环境内的分组比例；mix 仍包含横向速度。
        moving_proportions={"x": 0.3, "y": 0.0, "yaw": 0.2, "mix": 0.5},
        heading_command=True,
        rel_heading_envs=1.0,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.GroupedVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={
            # =========================
            # Legs
            # =========================
            "[lr]_hip_roll": 0.15,
            "[lr]_hip_yaw": 0.20,
            "[lr]_hip_pitch": 0.35,
            "[lr]_knee_pitch": 0.45,
            "[lr]_ankle_pitch": 0.25,
            "[lr]_ankle_roll": 0.25,
            # =========================
            # Waist
            # =========================
            "joint_waist_yaw": 0.20,
            "joint_waist_pitch": 0.15,
            "joint_waist_roll": 0.15,
            # =========================
            # Arms
            # =========================
            "[lr]_shoulder_pitch": 0.35,
            "[lr]_shoulder_roll": 0.30,
            "[lr]_shoulder_yaw": 0.30,
            "[lr]_elbow_pitch": 0.40,
            "[lr]_wrist_yaw": 0.20,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the actor."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        torso_ang_vel = ObsTerm(
            func=mdp.body_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )

        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        torso_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=2.0)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-10.0, 10.0),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.25,
            clip=(-10.0, 10.0),
        )
        
        actions = ObsTerm(func=mdp.last_action)


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        torso_ang_vel = ObsTerm(
            func=mdp.body_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
            scale=0.25,
        )

        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        torso_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
        )

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=2.0)


        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            scale=0.25,
        )
        actions = ObsTerm(func=mdp.last_action,history_length=3)

        joint_pos_err = ObsTerm(
            func=mdp.joint_pos_err,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            history_length=5
        )

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.5, 1.5),
        )


        feet_scan = ObsTerm(
            func=mdp.feet_scan,
            params={"sensor_names": ["height_scanner_left", "height_scanner_right"]},
            clip=(-1.5, 1.5),
        )


        feet_air_time = ObsTerm(
            func=mdp.air_time,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
                )
            },
        )

        feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
                ),
            },
        )

        feet_static_friction = ObsTerm(
            func=mdp.feet_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
                ),
                "component": 0,
            },
            scale=1.0,
        )

        feet_dynamic_friction = ObsTerm(
            func=mdp.feet_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
                ),
                "component": 1,
            },
            scale=1.0,
        )

        base_mass_ratio = ObsTerm(
            func=mdp.base_mass_ratio,
            params={"event_name": "base_mass"},
            scale=1.0,
        )
        base_com_offset = ObsTerm(
            func=mdp.base_com_offset_observation,
            params={"event_name": "base_com"},
            scale=(10.0, 10.0, 10.0),
        )

        base_external_wrench = ObsTerm(
            func=mdp.base_wrench_observation,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
            scale=(0.01, 0.01, 0.01, 0.1, 0.1, 0.1),
        )



        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()



@configclass
class EventCfg:

    # ---------- startup ----------

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (0.8, 1.25),
            "operation": "scale",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "com_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.015, 0.015),
            },
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
            "friction_distribution_params": (0.0, 0.08),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # ---------- reset ----------

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.4, 0.4),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )

    external_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (-15.0, 15.0),
            "torque_range": (-2.0, 2.0),
        },
    )

    # ---------- interval ----------

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-0.25, 0.25),
            },
        },
    )
    external_wrench = None


@configclass
class RewardsCfg:
    """Reward terms ordered as task, coordination, and constraint groups."""

    # Task：跟踪指令、保持直立和离地高度；零指令附近保持站姿。
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "command_name": "base_velocity",
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "command_name": "base_velocity",
        },
    )

    base_upright = RewTerm(
        func=mdp.base_mothion_up_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
        },
    )

    torso_upright = RewTerm(
        func=mdp.base_mothion_up_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["link_waist_roll"],
            ),
        },
    )

    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-0.0,
        params={
            "target_height": 0.94,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "scan_half_extent": (0.2, 0.2),  # 基座 yaw 坐标系：x/y 各 ±0.2 m
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.0,
        params={
            "command_name": "base_velocity",
            "legs_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES1,
                preserve_order=True,
            ),
            "waist_cfg": SceneEntityCfg(
                "robot",
                joint_names=WAIST_JOINT_NAMES,
                preserve_order=True,
            ),
            "arms_cfg": SceneEntityCfg(
                "robot",
                joint_names=ARM_JOINT_NAMES,
                preserve_order=True,
            ),
            "legs_weight": 1.0,
            "waist_weight": 0.5,
            "arms_weight": 0.2,
            "stand_speed": 0.05,
            "move_speed": 0.20,
            "posture_softness": 0.05,
            "gait_reward_name": "gait",  # gait 完全启用后解除默认站姿约束
        },
    )

    # Coordination：基座/躯干运动稳定性、步态和全身协调。
    base_motion_penalty = RewTerm(
        func=mdp.base_motion_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "lin_vel_b": 0.8,
            "ang_vel_b": 0.2,
        },
    )

    torso_motion_penalty = RewTerm(
        func=mdp.body_motion_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["link_waist_roll"],
            ),
            "lin_vel_b": 0.8,
            "ang_vel_b": 0.2,
        },
    )

    base_lateral_sway = RewTerm(
        func=mdp.base_lateral_sway_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "lin_vel_weight": 1.0,
            "roll_rate_weight": 0.5,
            "roll_angle_weight": 0.5,
            "max_speed": 0.10,
        },
    )

    gait = RewTerm(
        func=mdp.gait,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "period": 0.4,  # 完整左右脚周期（秒），相位观测自动使用同一参数
            "duty_factor": 0.6,
            "target_height": 0.06,  # 摆腿峰值：脚底高于该脚 scan 最高地形 6 cm
            "height_std": 0.03,
            "sole_offset": 0.054909,
            "command_threshold": 0.1,
            "command_transition_width": 0.05,
            "phase_transition": 0.1,
            "contact_threshold": 1.0,
            "sensor_names": ["height_scanner_left", "height_scanner_right"],
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
        },
    )




    foothold_flatness = RewTerm(
        func=mdp.foothold_flatness,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
            ),
            "sensor_names": ["height_scanner_left", "height_scanner_right"],
            "command_name": "base_velocity",
            "height_std": 0.02,
            "min_air_time": 0.08,
            "contact_threshold": 1.0,
            "command_threshold": 0.1,
            "command_transition_width": 0.05,
        },
    )


    # Style costs share gait timing; each term has a separate log and weight.
    leg_style = RewTerm(
        func=mdp.leg_style_penalty,
        weight=0.0,
        params={
            "knee_cfg": SceneEntityCfg("robot", joint_names=["l_knee_pitch", "r_knee_pitch"], preserve_order=True),
            "hip_cfg": SceneEntityCfg(
                "robot", joint_names=["l_hip_roll", "l_hip_yaw", "r_hip_roll", "r_hip_yaw"], preserve_order=True
            ),
            "pitch_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_hip_pitch", "l_knee_pitch", "l_ankle_pitch",
                    "r_hip_pitch", "r_knee_pitch", "r_ankle_pitch",
                ],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
            ),
            "gait_reward_name": "gait",
            "stance_knee_limit": 0.12,
            "fast_stance_knee_limit": 0.45,
            "swing_knee_min": 0.35,
            "fast_swing_knee_min": 0.55,
            "swing_weight": 0.25,
            "hip_weight": 0.1,
            # 摆腿中段：髋 + 膝 + 踝俯仰角之和接近 0，允许 0.15 rad 自由调整。
            "pitch_weight": 0.5,
            "pitch_tolerance": 0.15,
        },
    )

    arm_body_coordination = RewTerm(
        func=mdp.arm_body_coordination_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["l_hip_pitch", "r_hip_pitch", "l_shoulder_pitch", "r_shoulder_pitch", "joint_waist_yaw"],
                preserve_order=True,
            ),
            "gait_reward_name": "gait",
            "min_arm_scale": 0.4,
            "max_arm_scale": 1.0,
            "position_tolerance": 0.05,
            "velocity_weight": 0.15,
            "waist_weight": 0.5,
            "waist_yaw_gain": 0.2,
            "max_waist_yaw": 0.15,
        },
    )

    upper_body_style = RewTerm(
        func=mdp.upper_body_style_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_shoulder_roll", "l_shoulder_yaw", "l_elbow_pitch", "l_wrist_yaw",
                    "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch", "r_wrist_yaw",
                    "joint_waist_pitch", "joint_waist_roll",
                ],
                preserve_order=True,
            ),
            "gait_reward_name": "gait",
            "min_elbow_offset": -0.1,
            "max_elbow_offset": -0.6,
            "arm_tolerance": 0.1,
            "elbow_tolerance": 0.15,
            "waist_tolerance": 0.08,
        },
    )

    cycle_symmetry = RewTerm(
        func=mdp.cycle_symmetry_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "sensor_names": ["height_scanner_left", "height_scanner_right"],
            "gait_reward_name": "gait",
            "posture_tolerance": 0.05,
            "amplitude_tolerance": 0.05,
            "command_tolerance": 0.15,
            "terrain_tolerance": 0.05,
        },
    )

    # Constraint：接触、动作平滑、能耗、关节限位和终止惩罚。
    flight_phase = RewTerm(
        func=mdp.flight_phase_penalty,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot_roll", "r_foot_roll"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "speed_threshold": 0.1,
        },
    )




    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble_penalty,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot_roll", "r_foot_roll"],
                preserve_order=True,
            ),
            "threshold": 4.0,
        },
    )


    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0,
    )

    action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2,
        weight=0.0,
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_mechanical_power = RewTerm(
        func=mdp.joint_mechanical_power_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=r"^(?!l_foot_roll$|r_foot_roll$).*$",
            ),
        },
    )
    
    terminated = RewTerm(
        func=mdp.is_terminated,
        weight=0.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # type: ignore


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material # type: ignore
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
