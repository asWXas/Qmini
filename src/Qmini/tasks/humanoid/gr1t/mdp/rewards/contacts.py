# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foot contact costs and touchdown terrain quality."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

from .common import _command_norm1, _command_speed, _smoothstep

__all__ = ["feet_slide", "feet_stumble_penalty", "flight_phase_penalty", "foothold_flatness"]


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding (linear velocity) when in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]

    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_stumble_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 4.0,
) -> torch.Tensor:
    """惩罚脚部水平冲击力 (绊脚 / 蹭地 / 拖脚)。

    当脚底水平力显著大于垂直力时判定为绊脚, 促使机器人真正抬脚跨步而非贴地拖行。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]

    force_xy = torch.linalg.norm(forces[..., :2], dim=-1)
    force_z = torch.abs(forces[..., 2])

    return torch.sum(torch.relu(force_xy - threshold * force_z), dim=1)


def flight_phase_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.1,
    running_speed_range: tuple[float, float] | None = None,
    air_time_tolerance: float = 0.02,
) -> torch.Tensor:
    """Discourage walking flight, with contact-noise grace and optional running relief."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]

    both_in_air = torch.all(contact_time <= 0.0, dim=1).float()
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    flight_gate = _smoothstep(air_time.min(dim=1).values, air_time_tolerance, air_time_tolerance + 0.02)
    speed = _command_speed(env, command_name)
    gate = _smoothstep(speed, speed_threshold, speed_threshold + 0.2)
    if running_speed_range is not None:
        gate = gate * (1.0 - _smoothstep(speed, *running_speed_range))
    return both_in_air * flight_gate * gate


def foothold_flatness(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    sensor_names: list[str],
    command_name: str = "base_velocity",
    height_std: float = 0.02,
    min_air_time: float = 0.08,
    contact_threshold: float = 1.0,
    command_threshold: float = 0.1,
    command_transition_width: float = 0.05,
) -> torch.Tensor:
    """Reward flat local terrain only on touchdown after a meaningful swing.

    Foot order is left, right. Each landing scores exp(-(height_range/height_std)^2),
    using max(Z)-min(Z) of all nine local scan hits. This prefers level support:
    slopes and step edges also lower the score. Absolute terrain altitude does not.
    All nine rays must be finite, so incomplete patches cannot appear safely flat.

    Sustained contact, insufficient airtime, and commands in the standing
    deadband earn nothing. Scores are summed over landing feet (0..2),
    independently of the gait clock, with a smooth command activation.
    """
    if len(sensor_names) != 2 or len(set(sensor_names)) != 2:
        raise ValueError("Expected distinct left and right foothold scanners")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (height_std, min_air_time, contact_threshold, command_transition_width)
    ):
        raise ValueError("Foothold scales, minimum air time, and contact threshold must be finite and positive")
    if not math.isfinite(command_threshold) or command_threshold < 0.0:
        raise ValueError("command_threshold must be finite and nonnegative")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    # The sensor frontend can expose touchdown flags as float 0/1 arrays.
    # Convert to a boolean mask before combining with the support conditions.
    first_contact = first_contact > 0
    last_air_time = contact_sensor.data.last_air_time
    if last_air_time is None:
        raise RuntimeError("Foothold quality requires track_air_time=True")
    air_time = last_air_time.torch[:, sensor_cfg.body_ids]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    if first_contact.shape[1] != 2:
        raise ValueError("Expected two foot contacts ordered left, right")
    supported = torch.isfinite(forces).all(dim=-1) & (forces[..., 2] > contact_threshold)
    landing = first_contact & supported & torch.isfinite(air_time) & (air_time >= min_air_time)

    qualities = []
    for name in sensor_names:
        hits = env.scene.sensors[name].data.ray_hits_w.torch
        if hits.shape[1] != 9:
            raise ValueError(f"Foothold scanner {name} must have 9 rays")
        finite = torch.isfinite(hits).all(dim=-1)
        heights = torch.where(finite, hits[..., 2], 0.0)
        height_range = heights.amax(dim=-1) - heights.amin(dim=-1)
        quality = torch.exp(-(height_range / height_std).square())
        qualities.append(torch.where(finite.all(dim=-1), quality, 0.0))
    quality = torch.stack(qualities, dim=-1)
    gate = _smoothstep(
        _command_norm1(env, command_name), command_threshold, command_threshold + command_transition_width
    )
    return (quality * landing).sum(dim=-1) * gate
