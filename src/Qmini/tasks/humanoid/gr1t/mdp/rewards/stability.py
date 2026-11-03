# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordination rewards: base and torso damping and lateral sway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

from .common import command_smooth_scale

__all__ = ["base_motion_penalty", "body_motion_penalty", "base_lateral_sway_penalty"]


def base_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_b: float = 0.8,
    ang_vel_b: float = 0.2,
) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return lin_vel_b * torch.square(asset.data.root_lin_vel_b.torch[:, 2]) + ang_vel_b * torch.sum(
        torch.abs(asset.data.root_ang_vel_b.torch[:, :2]), dim=1
    )


def body_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_b: float = 0.8,
    ang_vel_b: float = 0.2,
) -> torch.Tensor:
    """惩罚指定刚体自身坐标系下的垂直线速度 (z) 和水平角速度 (roll/pitch)。支持单/多 body。"""
    asset = env.scene[asset_cfg.name]

    # [N, B, 3] & [N, B, 4]
    body_lin_vel_w = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids]
    body_ang_vel_w = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]

    # World frame -> 各自 Body frame (原生支持多维广播)
    body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)

    # z 轴线速度平方 (L2) + roll/pitch 角速度绝对值之和 (L1)
    lin_penalty = torch.square(body_lin_vel_b[..., 2])
    ang_penalty = torch.linalg.norm(body_ang_vel_b[..., :2], ord=1, dim=-1)

    # 沿 body 维度求和，返回 [num_envs]
    return torch.sum(lin_vel_b * lin_penalty + ang_vel_b * ang_penalty, dim=1)


def base_lateral_sway_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    lin_vel_weight: float = 1.0,
    roll_rate_weight: float = 0.5,
    roll_angle_weight: float = 0.5,
    max_speed: float = 0.10,
) -> torch.Tensor:
    """抑制基座左右摇摆 (横向速度 + roll 角速度 + 侧倾角)。

    人形机器人在行走时常见 "鸭子步" (左右大幅晃动), 该惩罚可让质心轨迹更接近
    人类的窄支撑、直线行走风格, 同时在站立时通过平滑门控自动关闭。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    commanded_y = env.command_manager.get_command(command_name)[:, 1]
    lin_vel_y_error = asset.data.root_lin_vel_b.torch[:, 1] - commanded_y
    roll_rate = asset.data.root_ang_vel_b.torch[:, 0]
    roll_angle = asset.data.projected_gravity_b.torch[:, 1]

    penalty = (
        lin_vel_weight * lin_vel_y_error.square()
        + roll_rate_weight * roll_rate.square()
        + roll_angle_weight * roll_angle.square()
    )
    return penalty * command_smooth_scale(env, command_name, max_speed=max_speed)
