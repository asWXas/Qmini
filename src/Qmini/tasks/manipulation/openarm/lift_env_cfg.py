# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm grasp/transport task with drop recovery, progress rewards and quiet arrival."""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import ObjectUniformPoseCommandCfg

from . import mdp

# Both the discrete reward and success termination use these object-pose tolerances.
OBJECT_POSITION_TOLERANCE = 0.025  # meters
OBJECT_ORIENTATION_TOLERANCE = 0.15  # radians (about 8.6 degrees)


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Tabletop scene; the robot and rigid object are supplied by the concrete config."""

    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Object position and orientation goals in the robot root frame."""

    object_pose = ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        position_only=False,
        make_quat_unique=True,
        # Sample B at reset and keep it fixed until this episode ends.
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.4),
            pos_y=(-0.2, 0.2),
            pos_z=(0.1, 0.4),
            roll=(-math.pi, math.pi),
            pitch=(-math.pi, math.pi),
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Joint-position control for the arm and binary position control for the gripper."""

    # Robot-specific joints and position targets are set by the derived environment config.
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Actor observations and a separate critic group with privileged object state."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy measurements with additive uniform noise; commands remain exact."""

        # Joint-position noise uses radians for the arm and meters for the fingers.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_error = ObsTerm(
            func=mdp.joint_position_execution_error,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, noise=Unoise(n_min=-0.005, n_max=0.005))
        object_orientation = ObsTerm(
            func=mdp.object_orientation_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
        )
        # Full object target in robot-root axes: (x, y, z, qx, qy, qz, qw).
        target_object_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Policy observations plus privileged object state from the simulator."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_pos_error = ObsTerm(
            func=mdp.joint_position_execution_error,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        object_orientation = ObsTerm(
            func=mdp.object_orientation_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
        )
        object_lin_vel = ObsTerm(
            func=mdp.object_linear_velocity_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
        )
        object_ang_vel = ObsTerm(
            func=mdp.object_angular_velocity_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
        )
        contact_state = ObsTerm(func=mdp.object_and_gripper_contacts)
        contact_forces = ObsTerm(func=mdp.object_and_gripper_contact_forces, scale=0.05, clip=(-5.0, 5.0))
        end_effector_pose = ObsTerm(func=mdp.end_effector_pose_in_robot_root_frame)
        end_effector_velocity = ObsTerm(
            func=mdp.end_effector_velocity_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["openarm_ee_tcp"])},
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Sample mass/material at startup and object position at each episode reset.

    Size variation is configured by the scene spawner in joint_pos_env_cfg.py.
    """

    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    # Small offsets around the nominal raised pose (radians); leave fingers open.
    # Keep this after reset_all so it is not overwritten by the default reset.
    reset_arm_joints = EventTerm(
        func=mdp.reset_arm_joints_near_default,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint[1-7]"]),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Stage-specific differences and a success bonus; no continuous positive income.

    The object_pose_success termination owns the shared stage state and runs
    before these rewards. Uncontrolled losses repay positive net transport
    progress; supported set-downs preserve accounting across regrasp attempts.
    """

    reaching_object = RewTerm(func=mdp.staged_manipulation_reward, params={"component": "reaching_object"}, weight=1.0)

    approach_open_gripper = RewTerm(
        func=mdp.staged_manipulation_reward,
        params={"component": "approach_open_gripper"},
        weight=0.25,
    )

    # Bridge alignment -> physical closure -> contact without paying for hovering.
    closing_object = RewTerm(func=mdp.staged_manipulation_reward, params={"component": "closing_object"}, weight=2.0)

    grasping_object = RewTerm(
        func=mdp.staged_manipulation_reward,
        params={"component": "grasping_object"},
        weight=2.0,
    )

    grasp_confirmed = RewTerm(func=mdp.staged_manipulation_reward, params={"component": "grasp_confirmed"}, weight=5.0)

    lifting_object = RewTerm(
        func=mdp.staged_manipulation_reward,
        params={"component": "lifting_object"},
        weight=4.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.staged_manipulation_reward,
        params={"component": "object_goal_tracking"},
        weight=4.0,  # +0.04 per cm closer while held; -0.04 per cm farther.
    )

    object_orientation_tracking = RewTerm(
        func=mdp.staged_manipulation_reward,
        params={"component": "object_orientation_tracking"},
        weight=2.0,  # +0.2 per 0.1 rad improvement, held or during supported tabletop adjustment.
    )

    object_pose_success = RewTerm(
        func=mdp.staged_manipulation_reward,
        params={"component": "object_pose_success"},
        weight=100.0,  # Actual one-time +100, compensated for RewardManager's dt scaling.
    )

    # Supported, quiet, deliberate set-downs are permitted; only uncontrolled
    # losses receive this penalty. Neither set-down nor regrasp pays a bonus.
    grasp_lost = RewTerm(func=mdp.staged_manipulation_reward, params={"component": "grasp_lost"}, weight=-2.0)

    elapsed_time = RewTerm(func=mdp.elapsed_time_cost, weight=-0.1)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_limits = RewTerm(func=mdp.joint_pos_limits, params={"asset_cfg": SceneEntityCfg("robot")}, weight=-1.0)

    object_motion = RewTerm(func=mdp.object_motion_l2, weight=-0.2)
    excessive_grip_force = RewTerm(func=mdp.excessive_grip_force, params={"force_limit": 40.0}, weight=-0.05)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_pose_success = DoneTerm(
        func=mdp.grasp_transport_success,
        params={
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
            "command_name": "object_pose",
            "grasp_time": 0.1,
            "lift_height": 0.08,
            "position_tolerance": OBJECT_POSITION_TOLERANCE,
            "orientation_tolerance": OBJECT_ORIENTATION_TOLERANCE,
            "linear_speed_tolerance": 0.05,
            "angular_speed_tolerance": 0.5,
            "settle_time": 0.2,
            "force_limit": 40.0,
            "support_time": 0.1,
            "table_linear_speed_limit": 0.15,
            "table_angular_speed_limit": 1.5,
        },
    )

    # A recoverable drop returns to grasping. Only a fall below the tabletop
    # workspace ends the episode; do not reset merely on loss of finger contact.
    object_out_of_workspace = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Grasp, carry or set down/reorient/regrasp, recover drops, and reach the object goal."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Use 100 Hz physics and 50 Hz control, with time for gentle transport and holding."""
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
