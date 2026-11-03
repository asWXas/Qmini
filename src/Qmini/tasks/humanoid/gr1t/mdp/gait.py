"""Timed biped gait with terrain-relative sole-height guidance."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["gait", "biped_gait_clock"]


def _phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("Gait period must be finite and positive")
    # Each environment resets its own clock; the clock continues while standing.
    return torch.remainder(env.episode_length_buf.to(torch.float32) * (env.step_dt / period), 1.0)


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def _swing_profile(env, period: float, duty_factor: float, phase_transition: float):
    """Shared left/right swing progress and phase weights for foot and joint style."""
    if not math.isfinite(duty_factor) or not 0.5 <= duty_factor < 1.0:
        raise ValueError("Walking duty_factor must be in [0.5, 1)")
    if not math.isfinite(phase_transition) or not 0.0 < phase_transition <= 0.5:
        raise ValueError("phase_transition must be in (0, 0.5]")
    phase = _phase(env, period)
    foot_phase = torch.remainder(phase[:, None] + phase.new_tensor((0.0, 0.5)), 1.0)
    swing_time = ((foot_phase - duty_factor) / (1.0 - duty_factor)).clamp(0.0, 1.0)
    swing_weight = _smoothstep(swing_time / phase_transition) * _smoothstep((1.0 - swing_time) / phase_transition)
    return swing_time, swing_weight


def _activation(env, command_name: str, command_threshold: float, command_transition_width: float):
    if not math.isfinite(command_threshold) or command_threshold < 0.0:
        raise ValueError("Gait command_threshold must be finite and nonnegative")
    if not math.isfinite(command_transition_width) or command_transition_width <= 0.0:
        raise ValueError("Gait command_transition_width must be finite and positive")
    command = env.command_manager.get_command(command_name)
    magnitude = command[:, :3].abs().sum(dim=-1)
    return _smoothstep((magnitude - command_threshold) / command_transition_width)


def _reward_cfg(env, reward_name: str):
    """Find a term in either flat or grouped reward configurations."""
    rewards = env.cfg.rewards
    cfg = getattr(rewards, reward_name, None)
    if cfg is not None:
        return cfg
    for group in vars(rewards).values():
        cfg = getattr(group, reward_name, None)
        if cfg is not None:
            return cfg
    return None


def biped_gait_clock(env: ManagerBasedRLEnv, reward_name: str = "gait") -> torch.Tensor:
    """[sin, cos] of the reward's clock, faded to zero when gait is disabled.

    Read the reward configuration so editing its period also updates actor and
    critic observations. Left/right reflection advances this clock by half a cycle.
    """
    cfg = _reward_cfg(env, reward_name)
    if cfg is None or cfg.weight == 0.0:
        return torch.zeros((env.num_envs, 2), device=env.device)
    params = cfg.params
    phase = _phase(env, params.get("period", 0.8))
    gate = _activation(
        env,
        params.get("command_name", "base_velocity"),
        params.get("command_threshold", 0.1),
        params.get("command_transition_width", 0.05),
    )
    angle = 2.0 * math.pi * phase
    return torch.stack((angle.sin(), angle.cos()), dim=-1) * gate[:, None]


def gait(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sensor_names: list[str],
    command_name: str = "base_velocity",
    period: float = 0.8,
    duty_factor: float = 0.6,
    target_height: float = 0.06,
    height_std: float = 0.03,
    sole_offset: float = 0.054909,
    command_threshold: float = 0.1,
    command_transition_width: float = 0.05,
    phase_transition: float = 0.1,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward alternating stance and smooth swing-height tracking, in [0, 1].

    All body/sensor selections must be left, right. ``period`` is one complete
    left/right cycle in seconds. Each foot spends ``duty_factor`` of it in stance;
    their clocks are half a cycle apart. During swing, normalized time s runs
    from 0 to 1 and target sole Z is max(valid local scan Z) + height*sin(pi*s)^2.
    The sole reference is the foot-link point (0, 0, -sole_offset), rotated with
    the foot; 0.054909 m is the GR1T2 foot mesh's lowest local Z magnitude.

    Stance rewards upward support; swing rewards no contact AND height tracking.
    Smooth phase weights allow lift-off/touchdown without abrupt clock switches.
    Missing scans give no swing-height reward, while valid stance support remains
    rewarded. No median/mean terrain approximation is used for the swing target.
    Activation uses |vx|+|vy|+|wz|, matching stand_still: zero at <=0.1 by default,
    smoothly reaching full strength at 0.15. This is not a temporal command filter.
    """
    if len(sensor_names) != 2 or len(set(sensor_names)) != 2:
        raise ValueError("Expected distinct left and right foot scanners")
    if any(not math.isfinite(value) or value <= 0.0 for value in (target_height, height_std, contact_threshold)):
        raise ValueError("Height, height_std, and contact_threshold must be finite and positive")
    if not math.isfinite(sole_offset) or sole_offset < 0.0:
        raise ValueError("sole_offset must be finite and nonnegative")

    swing_time, swing_weight = _swing_profile(env, period, duty_factor, phase_transition)
    gate = _activation(env, command_name, command_threshold, command_transition_width)
    target_clearance = target_height * torch.sin(math.pi * swing_time).square()

    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    orientations = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    if positions.shape[1] != 2 or forces.shape[1] != 2:
        raise ValueError("Expected two feet and contacts ordered left, right")
    sole_local = torch.zeros_like(positions)
    sole_local[..., 2] = -sole_offset
    sole_z = positions[..., 2] + quat_apply(orientations, sole_local)[..., 2]

    ground, valid_scans = [], []
    for name in sensor_names:
        hits = env.scene.sensors[name].data.ray_hits_w.torch
        valid = torch.isfinite(hits).all(dim=-1)
        highest = hits[..., 2].masked_fill(~valid, -torch.inf).amax(dim=-1)
        valid_scan = valid.any(dim=-1)
        ground.append(torch.where(valid_scan, highest, 0.0))
        valid_scans.append(valid_scan)
    ground_z = torch.stack(ground, dim=-1)
    valid = torch.stack(valid_scans, dim=-1)
    height_reward = torch.exp(-((sole_z - ground_z - target_clearance) / height_std).square())
    height_reward = torch.where(valid, torch.nan_to_num(height_reward, nan=0.0), 0.0)

    finite_forces = torch.isfinite(forces).all(dim=-1)
    supported = finite_forces & (forces[..., 2] > contact_threshold)
    airborne = finite_forces & (torch.linalg.vector_norm(forces, dim=-1) <= contact_threshold)
    per_foot_reward = (1.0 - swing_weight) * supported + swing_weight * airborne * height_reward
    return per_foot_reward.mean(dim=-1) * gate
