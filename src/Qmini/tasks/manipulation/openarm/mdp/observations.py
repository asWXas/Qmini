# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the rigid lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_unique, subtract_frame_transforms

from .contacts import finger_object_contact, finger_object_forces
from .state import tcp_pose_w

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def joint_position_execution_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Last applied joint position target minus the current measured joint position.

    Observations follow physics, so the actuator command buffer holds the target
    from the transition just executed, including action scaling, offsets and the
    gripper's binary position mapping. Revolute errors are in radians; prismatic
    errors are in meters. Reset rows have no previous action and return zero.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    target = robot.actuators.target_command.position.torch[:, asset_cfg.joint_ids]
    position = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error = target - position
    return error.masked_fill((env.episode_length_buf == 0).unsqueeze(-1), 0.0)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene["robot" if robot_cfg is None else robot_cfg.name]
    object: RigidObject = env.scene["object" if object_cfg is None else object_cfg.name]
    object_pos_w = object.data.root_pos_w.torch[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, object_pos_w)
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Object orientation relative to the robot root, as a canonical (x, y, z, w) quaternion."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_unique(quat_mul(quat_inv(robot.data.root_quat_w.torch), obj.data.root_quat_w.torch))


def object_linear_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Object COM world linear velocity expressed in robot-root axes, in m/s.

    The OpenArm base is fixed; this rotates the velocity without subtracting base motion.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w.torch, obj.data.root_lin_vel_w.torch)


def object_angular_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Object world angular velocity expressed in the fixed robot-root axes, in rad/s."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w.torch, obj.data.root_ang_vel_w.torch)


def object_and_gripper_contacts(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[object touching, left/right touching anything, left/right touching object].

    Robot contact points detect opposing finger forces even if they cancel out;
    net force additionally detects support by static surfaces such as the table.
    """
    points = env.scene["object_contact"].data.contact_pos_w.torch
    object_touching = torch.isfinite(points).all(dim=-1).reshape(env.num_envs, -1).any(dim=-1)
    object_touching |= env.scene["object_contact"].data.net_forces_w.torch.norm(dim=-1).amax(dim=-1) > 0.1
    finger_object = finger_object_contact(env)
    finger_any = (
        torch.stack(
            [
                env.scene[name].data.net_forces_w.torch[:, 0].norm(dim=-1) > 0.1
                for name in ("left_finger_contact", "right_finger_contact")
            ],
            dim=-1,
        )
        | finger_object
    )
    return torch.cat((object_touching[:, None], finger_any, finger_object), dim=-1).float()


def object_and_gripper_contact_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Object net, finger net, and finger-object forces in base axes (15 values)."""
    obj = env.scene["object_contact"].data.net_forces_w.torch[:, :1]
    finger_net = torch.stack(
        [env.scene[name].data.net_forces_w.torch[:, 0] for name in ("left_finger_contact", "right_finger_contact")],
        dim=1,
    )
    forces = torch.cat((obj, finger_net, finger_object_forces(env)), dim=1)
    rotation = env.scene["robot"].data.root_quat_w.torch[:, None].expand(-1, 5, -1)
    return quat_apply_inverse(rotation.reshape(-1, 4), forces.reshape(-1, 3)).reshape(env.num_envs, 15)


def end_effector_pose_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """TCP position [m] and canonical xyzw quaternion in the robot base frame."""
    robot = env.scene["robot"]
    tcp_position, tcp_orientation = tcp_pose_w(env)
    position, orientation = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        tcp_position,
        tcp_orientation,
    )
    return torch.cat((position, quat_unique(orientation)), dim=-1)


def end_effector_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """TCP-origin linear velocity [m/s] and angular velocity [rad/s] in the fixed base frame."""
    robot = env.scene[asset_cfg.name]
    rotation = robot.data.root_quat_w.torch
    linear = robot.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids[0]]
    angular = robot.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids[0]]
    return torch.cat((quat_apply_inverse(rotation, linear), quat_apply_inverse(rotation, angular)), dim=-1)
