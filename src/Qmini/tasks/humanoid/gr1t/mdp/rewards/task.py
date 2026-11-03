# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task rewards: upright posture, standing, and local terrain-relative height."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster

from ..gait import _activation as _gait_activation
from ..gait import _reward_cfg
from .common import _command_norm1, _projected_gravity_body, _smoothstep

__all__ = ["base_mothion_up_exp", "stand_still", "base_height_penalty"]


def base_mothion_up_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """躯干直立奖励 (指数核) + 倒立惩罚。

    - 直立: 投影重力 xy 分量 ≈ 0 -> 奖励接近 1
    - 倒立: 投影重力 z > 0 -> 额外惩罚

    未指定 body_names 时使用 root(基座)的投影重力;
    指定 body_names 时使用对应刚体坐标系下的投影重力 (例如躯干 link_waist_roll)。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.body_names is not None:
        # 使用目标刚体自身坐标系下的投影重力 (修复原先 torso 与 base 完全等效的问题)
        gravity_b = _projected_gravity_body(asset, asset_cfg.body_ids).mean(dim=1)
    else:
        gravity_b = asset.data.projected_gravity_b.torch

    # 直立奖励: gx, gy 接近 0
    tilt_error = torch.sum(gravity_b[:, :2].square(), dim=1)
    upright_reward = torch.exp(-tilt_error / (std**2))

    # 倒立惩罚: gz > 0
    inverted_penalty = 2.0 * torch.clamp(gravity_b[:, 2], min=0.0)

    return upright_reward - inverted_penalty


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    legs_cfg: SceneEntityCfg,
    waist_cfg: SceneEntityCfg,
    arms_cfg: SceneEntityCfg,
    legs_weight: float = 0.8,
    waist_weight: float = 0.6,
    arms_weight: float = 0.8,
    stand_speed: float = 0.05,
    move_speed: float = 0.20,
    posture_softness: float = 0.05,
    gait_reward_name: str | None = None,
) -> torch.Tensor:
    """默认站姿附近的平滑姿态惩罚，随指令大小平滑淡出。

    使用 |vx| + |vy| + |wz|：<= stand_speed 完全启用，>= move_speed
    完全关闭，中间采用端点斜率为零的 smoothstep。此处平滑针对指令幅值，
    不对时间上的指令跳变进行滤波。
    指定 gait_reward_name 后，再乘 (1 - gait 启用系数)：gait 完全启用时
    停止拉回默认站姿，避免起步时与摆腿目标竞争。未启用 gait 时保留原淡出。

    每个关节使用 sqrt(error^2 + posture_softness^2) - posture_softness，
    默认姿态附近为柔和的二次惩罚，较大偏差接近 L1；允许小幅平衡调整。
    posture_softness 的单位为 rad。本函数返回非负代价，应配置负奖励权重。
    """
    if not all(math.isfinite(value) for value in (stand_speed, move_speed, posture_softness)):
        raise ValueError("Standstill thresholds and posture_softness must be finite")
    if not 0.0 <= stand_speed < move_speed:
        raise ValueError("Standstill thresholds must satisfy 0 <= stand_speed < move_speed")
    if posture_softness <= 0.0:
        raise ValueError("posture_softness must be positive")
    asset: Articulation = env.scene[legs_cfg.name]

    # 指令 L1 范数: |vx| + |vy| + |wz|
    command_norm = _command_norm1(env, command_name)

    # 站立 -> 行走 平滑过渡 (<= stand_speed: 1, >= move_speed: 0)
    blend = 1.0 - _smoothstep(command_norm, stand_speed, move_speed)
    if gait_reward_name is not None:
        gait_cfg = _reward_cfg(env, gait_reward_name)
        if gait_cfg is not None and gait_cfg.weight != 0.0:
            params = gait_cfg.params
            blend = blend * (
                1.0
                - _gait_activation(
                    env,
                    params.get("command_name", command_name),
                    params.get("command_threshold", 0.1),
                    params.get("command_transition_width", 0.05),
                )
            )

    # Joint positions
    pos = asset.data.joint_pos.torch
    default_pos = asset.data.default_joint_pos.torch

    # Smooth absolute error: gentle near the default pose, approximately L1 farther away.
    deviation = torch.sqrt((pos - default_pos).square() + posture_softness**2) - posture_softness
    legs_penalty = deviation[:, legs_cfg.joint_ids].sum(dim=1)
    waist_penalty = deviation[:, waist_cfg.joint_ids].sum(dim=1)
    arms_penalty = deviation[:, arms_cfg.joint_ids].sum(dim=1)

    # Group weighting
    posture_penalty = (legs_weight * legs_penalty) + (waist_weight * waist_penalty) + (arms_weight * arms_penalty)

    return blend * posture_penalty


def base_height_penalty(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    scan_half_extent: tuple[float, float] = (0.2, 0.2),
) -> torch.Tensor:
    """基座相对局部平均地形高度的平方误差。

    以基座为中心，按 yaw 对齐的水平坐标选取 |x|、|y| 不超过
    scan_half_extent 的扫描命中点，仅对有限点的世界系 Z 求平均。
    区域内无有效点时跳过该环境的高度惩罚；不使用区域外地形替代。
    未配置传感器时，target_height 仍表示世界系目标高度。
    """
    if len(scan_half_extent) != 2 or any(not math.isfinite(v) or v <= 0.0 for v in scan_half_extent):
        raise ValueError("scan_half_extent must contain two finite positive half-widths")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w.torch
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        hits = sensor.data.ray_hits_w.torch
        finite = torch.isfinite(hits).all(dim=-1)
        safe_hits = torch.where(finite[..., None], hits, root_pos[:, None, :])
        heading = yaw_quat(asset.data.root_quat_w.torch)[:, None, :].expand(-1, hits.shape[1], -1)
        local = quat_apply_inverse(heading, safe_hits - root_pos[:, None, :])
        # Include grid points on the boundary despite float32 transform rounding.
        extent = local.new_tensor(scan_half_extent)
        selected = finite & (local[..., :2].abs() <= extent + 1e-5).all(dim=-1)
        count = selected.sum(dim=-1)
        mean_z = torch.where(selected, safe_hits[..., 2], 0.0).sum(dim=-1) / count.clamp(min=1)
        penalty = (root_pos[:, 2] - mean_z - target_height).square()
        return torch.where(count > 0, penalty, 0.0)
    return (root_pos[:, 2] - target_height).square()
