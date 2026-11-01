"""Command-gated diagonal contact schedule computed inside the reward."""

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class DiagonalGaitCfg:
    enabled: bool = True
    period: float = 0.8  # seconds per full cycle
    frequency: float | None = None  # Hz; when set, overrides period
    duty_factor: float = 0.5  # fraction of cycle in stance
    command_threshold: float = 0.1
    command_name: str = "base_velocity"
    phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)


def diagonal_gait_targets(env, gait_cfg: DiagonalGaitCfg):
    """Return phase, activation and FL/FR/RL/RR stance targets.

    The episode clock resets with the environment and continues while disabled.
    Only lateral velocity and yaw commands enable the schedule.
    """
    cfg = gait_cfg
    frequency = cfg.frequency
    if frequency is None:
        if not math.isfinite(cfg.period) or cfg.period <= 0:
            raise ValueError("gait period must be finite and positive")
        frequency = 1.0 / cfg.period
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("gait frequency must be finite and positive")
    if not 0 < cfg.duty_factor < 1:
        raise ValueError("gait duty_factor must lie between 0 and 1")
    if not math.isfinite(cfg.command_threshold) or cfg.command_threshold < 0:
        raise ValueError("gait command_threshold must be finite and nonnegative")
    command = env.command_manager.get_command(cfg.command_name)
    active = (torch.linalg.vector_norm(command[:, 1:3], dim=-1) > cfg.command_threshold) & cfg.enabled
    phase = torch.remainder(env.episode_length_buf * (env.step_dt * frequency), 1.0)
    if len(cfg.phase_offsets) != 4 or any(not math.isfinite(x) or not 0 <= x < 1 for x in cfg.phase_offsets):
        raise ValueError("gait phase_offsets must contain four finite values in [0, 1)")
    offsets = phase.new_tensor(cfg.phase_offsets)
    stance = torch.remainder(phase[:, None] + offsets, 1.0) < cfg.duty_factor
    return phase, active, stance


def diagonal_gait_contact_penalty(
    env, sensor_cfg: SceneEntityCfg, contact_threshold: float = 1.0,
    enabled: bool = True, command_name: str = "base_velocity",
    period: float = 0.8, frequency: float | None = None,
    duty_factor: float = 0.5, phase_offsets: tuple = (0.0, 0.5, 0.5, 0.0),
    command_threshold: float = 0.1,
):
    """Fraction of wheel contacts mismatching the schedule; disabled penalty is zero."""
    cfg = DiagonalGaitCfg(
        enabled=enabled, command_name=command_name, period=period, frequency=frequency,
        duty_factor=duty_factor, phase_offsets=phase_offsets, command_threshold=command_threshold,
    )
    _, active, stance = diagonal_gait_targets(env, cfg)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    if forces.shape[1] != 4:
        raise ValueError("Expected four wheel contacts ordered FL, FR, RL, RR")
    contact = torch.linalg.vector_norm(forces, dim=-1) > contact_threshold
    return (contact != stance).float().mean(dim=-1) * active
