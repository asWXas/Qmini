# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared command gates and body-frame helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["command_smooth_scale"]


def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """把 value 从 [lower, upper] 平滑映射到 [0, 1] (三次 smoothstep)。"""
    x = torch.clamp((value - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _command_speed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """指令水平速度大小 |vx, vy|, shape [num_envs]。"""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.norm(command[:, :2], dim=1)


def _command_norm1(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """指令 L1 范数 |vx| + |vy| + |wz|, shape [num_envs]。"""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.norm(command[:, :3], ord=1, dim=1)


def _projected_gravity_body(asset: Articulation, body_ids) -> torch.Tensor:
    """指定刚体坐标系下的重力方向, shape [num_envs, num_bodies, 3]。"""
    quat_w = asset.data.body_quat_w.torch[:, body_ids]
    gravity_w = asset.data.GRAVITY_VEC_W.torch.unsqueeze(1).expand(-1, quat_w.shape[1], -1)
    return quat_apply_inverse(quat_w, gravity_w)


def command_smooth_scale(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    max_speed: float = 0.15,
) -> torch.Tensor:
    """指令平滑开关: 移动速度在 0 ~ max_speed 之间由 0 平滑过渡到 1。"""
    return _smoothstep(_command_speed(env, command_name), 0.0, max_speed)
