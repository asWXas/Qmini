# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility exports for older clearance and phase configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

from .common import command_smooth_scale

__all__ = ["foot_clearance_scan_reward", "gait_phase"]


def foot_clearance_scan_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
    left_scanner_name: str = "height_scanner_left",
    right_scanner_name: str = "height_scanner_right",
) -> torch.Tensor:
    """Reward moderate clearance of moving swing feet relative to local terrain."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_scanner = env.scene.sensors[left_scanner_name]
    right_scanner = env.scene.sensors[right_scanner_name]

    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    left_hits = left_scanner.data.ray_hits_w.torch[..., 2]
    right_hits = right_scanner.data.ray_hits_w.torch[..., 2]
    left_valid = torch.isfinite(left_hits)
    right_valid = torch.isfinite(right_hits)
    left_ground_z = torch.where(left_valid, left_hits, -torch.inf).max(dim=1).values
    right_ground_z = torch.where(right_valid, right_hits, -torch.inf).max(dim=1).values
    ground_z = torch.stack([left_ground_z, right_ground_z], dim=1)
    valid_ground = torch.isfinite(ground_z)
    ground_z = torch.where(valid_ground, ground_z, foot_z)

    clearance = foot_z - ground_z
    clearance_reward = torch.exp(-torch.square(clearance - target_height) / (std**2))
    foot_vel_xy = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=2)
    velocity_gate = torch.tanh(tanh_mult * foot_speed)
    swing = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] <= 0.0
    reward = clearance_reward * velocity_gate * swing * valid_ground
    return reward.mean(dim=1) * command_smooth_scale(env, command_name)


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float = 0.5,
) -> torch.Tensor:
    """周期步态相位观测。返回 [num_envs, 2] 的 [sin(phase), cos(phase)]。"""
    # 每个 env 当前已运行时间及对应的周期相位 [0, 2π)
    time = env.episode_length_buf * env.step_dt
    phase = 2.0 * torch.pi * torch.remainder(time, period) / period

    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
