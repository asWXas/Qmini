# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the velocity-tracking locomotion environments on rough terrain."""

import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
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
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab_tasks.core.velocity import mdp as isaaclab_velocity_mdp
from isaaclab_tasks.utils import PresetCfg

from Qmini.assets.as2w import AS2W_CFG, LEG_JOINT_NAMES, WHEEL_JOINT_NAMES

from .. import mdp
from ..mdp.terrains import ROUGH_TERRAINS_CFG

WHEEL_SCANNER_NAMES = ["fl_wheel_scanner", "fr_wheel_scanner", "rl_wheel_scanner", "rr_wheel_scanner"]
# 两项步态奖励共用时序；actor/critic 相位观测直接读取奖励参数与启用条件。
GAIT_PERIOD = 0.5
GAIT_FREQUENCY = None  # Hz；设定后覆盖 GAIT_PERIOD。
GAIT_DUTY_FACTOR = 0.5  # 每条腿支撑相占比。
GAIT_PHASE_OFFSETS = (0.0, 0.5, 0.5, 0.0)  # FL、FR、RL、RR；叠加在观测的全局相位上。
# 步态切换与对角轮距约束共用：局部累计二阶高度差在此区间内平滑过渡。
TERRAIN_STEP_THRESHOLD = 0.03
TERRAIN_STEP_FULL_HEIGHT = 0.06

##
# Physics backend presets
##


@configclass
class RoughPhysicsCfg(PresetCfg):
    """Shared backend presets for locomotion velocity environments."""

    isaacsim_physx = PhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**15,
        gpu_collision_stack_size=2**28,  # 256 MiB：为 4096 环境的地形碰撞预留容量。
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default = isaacsim_physx


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # 本地纯色材质，避免 play 时下载远程 MDL 和 8K 地面纹理。
        visual_material=PreviewSurfaceCfg(
            diffuse_color=(0.45, 0.45, 0.45),
            roughness=0.8,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING # type: ignore
    # sensors: the concrete implementation is selected automatically from the active physics backend;
    # backend-specific fields such as ``global_world_only`` are ignored by the backends that do not use them
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/(?!.*base_link/).*base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.4, 1.0]), # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    # 轮边 3×3 扫描，从轮心上方 1 m 向下，才能看到高于轮心的台阶顶。
    # world 对齐使射线起点偏移不随轮子自转。
    fl_wheel_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/(?!.*FL_foot/).*FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]), # type: ignore
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=False,
    )
    fr_wheel_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/(?!.*FR_foot/).*FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]), # type: ignore
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=False,
    )
    rl_wheel_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/(?!.*RL_foot/).*RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]), # type: ignore
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=False,
    )
    rr_wheel_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/(?!.*RR_foot/).*RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]), # type: ignore
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)
    # 单轮逐接触点数据；用于静态地形上的切向滑动检测（PhysX）。
    fl_wheel_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        track_contact_points=True,
        max_contact_data_count_per_prim=64,
    )
    fr_wheel_contact = fl_wheel_contact.replace(prim_path="{ENV_REGEX_NS}/Robot/FR_foot") # type: ignore
    rl_wheel_contact = fl_wheel_contact.replace(prim_path="{ENV_REGEX_NS}/Robot/RL_foot") # type: ignore
    rr_wheel_contact = fl_wheel_contact.replace(prim_path="{ENV_REGEX_NS}/Robot/RR_foot") # type: ignore
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.GroupedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.05,
        # 先覆盖沿轮子滚动方向的越障，仍保留横移和转向训练。
        moving_proportions={"x": 0.3, "y": 0.2, "yaw": 0.2, "mix": 0.3},
        heading_command=False,
        rel_heading_envs=1.0,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.GroupedVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = isaaclab_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={
            ".*_hip": 0.125,
            ".*_thigh": 0.25,
            ".*_calf": 0.25,
        },
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    wheel_vel = isaaclab_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=10.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=isaaclab_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaaclab_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=isaaclab_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=isaaclab_mdp.generated_commands, 
            params={
                "command_name": "base_velocity"
            },
            scale=(2.0, 2.0, 2.0),
        )
        # 连续旋转轮子的累计角度没有地形信息，避免其无界增长进入策略。
        joint_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=isaaclab_mdp.last_action)
        height_scan = ObsTerm(
            func=isaaclab_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 特权观测（无噪声）。"""

        # =========================================================
        # 1. 机身状态与速度命令
        # =========================================================
        base_lin_vel = ObsTerm(
            func=isaaclab_mdp.base_lin_vel,
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=isaaclab_mdp.base_ang_vel,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=isaaclab_mdp.projected_gravity,
            scale=1.0,
        )
        
        velocity_commands = ObsTerm(
            func=isaaclab_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 2.0),
        )
        # 当前训练 actor 也读取 critic 组；纯 vx 平地/静止为零，仅 gait 模式显示相位。
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
        )

        # =========================================================
        # 2. 关节状态与动作历史
        # =========================================================
        joint_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=isaaclab_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            scale=0.05,
        )
        actions = ObsTerm(
            func=isaaclab_mdp.last_action,
            history_length=2
        )

        # =========================================================
        # 3. 执行误差（目标 - 实际）
        # =========================================================
        leg_position_error = ObsTerm(
            func=mdp.joint_position_execution_error,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            scale=1.0,
            history_length=5,
        )
        wheel_velocity_error = ObsTerm(
            func=mdp.joint_velocity_execution_error,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=WHEEL_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            scale=0.05,
            history_length=5,
        )

        # =========================================================
        # 4. 地形扫描
        # =========================================================
        height_scan = ObsTerm(
            func=isaaclab_mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            scale=1.0,
            clip=(-1.5, 1.5),
        )
        wheel_height_scan = ObsTerm(
            func=mdp.wheel_height_scan,
            params={
                "sensor_names": [
                    "fl_wheel_scanner",
                    "fr_wheel_scanner",
                    "rl_wheel_scanner",
                    "rr_wheel_scanner",
                ],
                "wheel_radius": 0.086,
            },
            scale=1.0,
            clip=(-1.5, 1.5),
        )

        # =========================================================
        # 5. 轮子接触状态与离地时间
        # =========================================================
        wheel_air_time = ObsTerm(
            func=mdp.feet_air_time,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                )
            },
            scale=1.0,
        )
        wheel_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                ),
                "threshold": 1.0,
            },
            scale=1.0,
        )
        # =========================================================
        # 6. 轮子摩擦系数
        # =========================================================
        wheel_dynamic_friction = ObsTerm(
            func=mdp.feet_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                ),
                "component": 1,
            },
            scale=1.0,
        )
        wheel_static_friction = ObsTerm(
            func=mdp.feet_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                ),
                "component": 0,
            },
            scale=1.0,
        )

        # =========================================================
        # 7. 动力学扰动（机身坐标系）
        # =========================================================
        base_mass_ratio = ObsTerm(
            func=mdp.base_mass_ratio,
            params={"event_name": "add_base_mass"},
            scale=1.0,
        )
        base_com_offset = ObsTerm(
            func=mdp.base_com_offset_observation,
            params={"event_name": "base_com"},
            scale=(10.0, 10.0, 10.0),
        )
        base_external_wrench = ObsTerm(
            func=mdp.base_wrench_observation,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["base_link"],
                )
            },
            scale=(0.01, 0.01, 0.01, 0.1, 0.1, 0.1),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # multiplicative +-25% log-uniform: scale-invariant across robot sizes with geometric mean 1.0
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    base_com = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    joint_pd_gains = EventTerm(
        func=isaaclab_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=isaaclab_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-2.0, 2.0),
        },
    )

    reset_base = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=isaaclab_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=isaaclab_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    base_external_force_torque = None
    push_robot = None



@configclass
class RewardsCfg:
    """按用途排列奖励项；名称、权重及多 critic 分组保持兼容。"""

    # ============================================================
    # 1. 速度跟踪
    # ============================================================
    track_lin_vel_xy_exp = RewTerm(
        func=isaaclab_mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=isaaclab_mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    # XY 指令与实际机体系速度的平方误差；停滞、反向及超速均惩罚，大误差不饱和。
    track_lin_vel_xy_error = RewTerm(
        func=mdp.lin_vel_xy_tracking_error_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # yaw 角速度平方误差；未转动、反向转动及转速过大均惩罚。
    track_ang_vel_z_error = RewTerm(
        func=mdp.ang_vel_z_tracking_error_l2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ============================================================
    # 2. 机体稳定与关节姿态
    # ============================================================
    lin_vel_z_l2 = RewTerm(
        func=isaaclab_mdp.lin_vel_z_l2,
        weight=-2.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=isaaclab_mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.38,  # 默认关节姿态对应的轮底到机身高度约 0.381 m。
            "height_std": 0.10,  # 偏低 10 cm 的代价为 1，避免低姿态滚动只受很弱惩罚。
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    gravity_alignment = RewTerm(
        func=mdp.gravity_z_alignment,
        weight=0.2,
        params={
            "std": 0.15,
        },
    )

    # 对角腿姿态镜像：FL ↔ RR、FR ↔ RL；hip 反号，thigh/calf 同号。
    diagonal_joint_symmetry = RewTerm(
        func=mdp.diagonal_joint_symmetry_l2,
        weight=-1.0,
        params={
            "left_cfg": SceneEntityCfg(
                "robot",
                joint_names=[f"{leg}_{joint}" for leg in ("FL", "FR") for joint in ("hip", "thigh", "calf")],
                preserve_order=True,
            ),
            "right_cfg": SceneEntityCfg(
                "robot",
                joint_names=[f"{leg}_{joint}" for leg in ("RR", "RL") for joint in ("hip", "thigh", "calf")],
                preserve_order=True,
            ),
        },
    )

    joint_pose_deviation = RewTerm(
        func=mdp.stand_still_joint_deviation,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "command_threshold": 0.1,  # 姿态系数过渡中心。
            "command_transition_width": 0.15,  # 总宽度：vx、vy、yaw 最大绝对值在 0.05–0.15 平滑过渡。
            "stand_still_scale": 1.5,
            "moving_hip_scale": 1.25,
            "moving_thigh_scale": 1.0,
            "moving_calf_scale": 1.0,
        },
    )




    # ============================================================
    # 3. 步态与轮子约束
    # ============================================================
    # 诊断门控、四轮摆动/支撑、关节执行误差与机身高度；始终返回 0。
    # weight 必须非零，否则 RewardManager 会跳过调用；指标见 Test/*。
    # test = RewTerm(
    #     func=mdp.GaitScanTest,
    #     weight=1.0,
    #     params={
    #         "height_sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "leg_joint_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
    #         "print_interval": 0,  # 只写训练日志。
    #     },
    # )

    # 静止时四轮支撑。
    standing_wheel_contact = RewTerm(
        func=mdp.standing_wheel_contact_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,  # vx、vy、yaw 均在死区内时要求四轮支撑。
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 10.0,  # 向上的法向力必须大于 1 N。
        },
    )

    # vy/yaw 任一超出死区：任何地形启用高度步态，允许同时有 vx。
    explicit_diagonal_gait = RewTerm(
        func=mdp.diagonal_gait_contact_penalty,
        weight=-5.0,
        params={
            "enabled": True,
            "command_name": "base_velocity",
            "period": GAIT_PERIOD,  # 完整周期，秒；与观测共享。
            "frequency": GAIT_FREQUENCY,
            "duty_factor": GAIT_DUTY_FACTOR,
            "phase_offsets": GAIT_PHASE_OFFSETS,  # FL+RR 与 FR+RL 对角交替。
            "command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
            ),
            "wheel_sensor_names": WHEEL_SCANNER_NAMES,
            "swing_height": 0.03,  # 摆动中点轮底高于轮边扫描最高地面，m。
            "wheel_radius": 0.086,
            "height_tracking_scale": 1.0,
            "height_std": 0.02,  # 高度误差尺度，独立于抬腿高度。
        },
    )

    # 仅纯 vx：无突变四轮滚动，有突变过渡到 gait；与上项互斥。
    pure_x_terrain_gait = RewTerm(
        func=mdp.pure_x_terrain_gait_penalty,
        weight=-2.0,
        params={
            "enabled": True,
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "step_threshold": TERRAIN_STEP_THRESHOLD,  # 不超过此突变强度时四轮滚动。
            "step_full_height": TERRAIN_STEP_FULL_HEIGHT,  # 达到此突变强度时完全使用步态。
            "pure_x_tolerance": None,  # 跟随 command_threshold，与横移/转向奖励互补。
            "command_name": "base_velocity",
            "period": GAIT_PERIOD,  # 完整周期，秒；与观测共享。
            "frequency": GAIT_FREQUENCY,
            "duty_factor": GAIT_DUTY_FACTOR,
            "phase_offsets": GAIT_PHASE_OFFSETS,  # FL+RR 与 FR+RL 对角交替。
            "command_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
            ),
            "wheel_sensor_names": WHEEL_SCANNER_NAMES,
            "swing_height": 0.03,  # 摆动中点高于轮边扫描最高地面，m。
            "wheel_radius": 0.086,
            "height_tracking_scale": 1.0,
            "height_std": 0.05,
        },
    )

    # 世界坐标三维对角轮距差；无突变完整约束，有突变平滑关闭。
    diagonal_distance_difference = RewTerm(
        func=mdp.diagonal_wheel_distance_difference,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "tolerance": 0.03,
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "step_threshold": TERRAIN_STEP_THRESHOLD,  # 不超过此突变强度时完整启用。
            "step_full_height": TERRAIN_STEP_FULL_HEIGHT,  # 达到此突变强度时完全关闭。
        },
    )

    # 轮子与地面的切向滑动。
    wheel_slip = RewTerm(
        func=mdp.WheelContactSlip,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "sensor_names": [
                "fl_wheel_contact",
                "fr_wheel_contact",
                "rl_wheel_contact",
                "rr_wheel_contact",
            ],
            "slip_threshold": 0.05,
            "contact_threshold": 2.0,      # 法向接触力阈值：2 N。
        },
    )

    # ============================================================
    # 4. 动作平滑与能耗
    # ============================================================
    action_rate_l2 = RewTerm(
        func=isaaclab_mdp.action_rate_l2,
        weight=-0.01,
    )

    action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2,
        weight=-0.001,
    )

    joint_torques_l2 = RewTerm(
        func=isaaclab_mdp.joint_torques_l2,
        weight=-1.0e-5,
    )

    leg_joint_acc_l2 = RewTerm(
        func=isaaclab_mdp.joint_acc_l2,
        weight=-1.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            ),
        },
    )

    wheel_joint_acc_l2 = RewTerm(
        func=isaaclab_mdp.joint_acc_l2,
        weight=-1.0e-9,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHEEL_JOINT_NAMES,
            ),
        },
    )

    # ============================================================
    # 5. 接触与关节安全
    # ============================================================
    # 轮子接触合力过大或水平碰撞。
    wheel_stumble = RewTerm(
        func=mdp.wheel_lateral_contact_force,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "force_ratio": 4,
            "force_limit": 180.0,
        },
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            # 弧度范围 (下限, 上限)；None 沿用各关节原有软限位。
            "joint_ranges": {
                ".*_hip": (-0.25,0.25),
                ".*_thigh": None,
                ".*_calf": None,
            },
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_torque_limits = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES,
            ),
        },
    )

    undesired_contacts = RewTerm(
        func=isaaclab_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="(?!(?:FL|FR|RL|RR)_foot$).*",
            ),
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(
        func=isaaclab_mdp.is_terminated,
        weight=-100.0,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaaclab_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=isaaclab_velocity_mdp.terrain_levels_vel) # type: ignore

    command_levels_lin_vel = CurrTerm(
        func=mdp.command_levels_lin_vel, # type: ignore
        params={"reward_term_name": "track_lin_vel_xy_exp", "range_multiplier": (0.2, 1.0)},
    )
    command_levels_ang_vel = CurrTerm(
        func=mdp.command_levels_ang_vel, # type: ignore
        params={"reward_term_name": "track_ang_vel_z_exp", "range_multiplier": (0.2, 1.0)},
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg()) # type: ignore
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
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
        # we tbash src/Qmini/tasks/quadruped/as2w_lp_cc/train.shick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        for sensor_name in ("fl_wheel_scanner", "fr_wheel_scanner", "rl_wheel_scanner", "rr_wheel_scanner"):
            sensor = getattr(self.scene, sensor_name, None)
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt
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

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
        # 播放直接使用配置的完整速度范围，不在首次 reset 时缩小至 10%。
        self.curriculum.command_levels_lin_vel = None # type: ignore
        self.curriculum.command_levels_ang_vel = None # type: ignore
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.episode_length_s = 150
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # remove random pushing events
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AS2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """AS2W rough-terrain training with explicit PD actuators."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AS2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        self.sim.use_newton_actuators = False


@configclass
class AS2WFlatEnvCfg(AS2WRoughEnvCfg):
    """Flat terrain with the same observation layout as rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None # type: ignore
        for name in ("fl_wheel_contact", "fr_wheel_contact", "rl_wheel_contact", "rr_wheel_contact"):
            getattr(self.scene, name).filter_prim_paths_expr = ["/World/ground/terrain/GroundPlane/CollisionPlane"]
