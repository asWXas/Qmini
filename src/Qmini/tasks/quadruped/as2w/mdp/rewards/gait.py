"""Command-gated wheel gait and terrain-dependent longitudinal rolling costs."""

import math

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from ..terrain_geometry import GridScanEdgeDetector as _GridScanEdgeDetector
from ..terrain_geometry import terrain_step_gate as _terrain_step_gate


@configclass
class DiagonalGaitCfg:
    enabled: bool = True
    period: float = 0.8  # seconds per full cycle
    frequency: float | None = None  # Hz; when set, overrides period
    duty_factor: float = 0.5  # fraction of cycle in stance
    command_threshold: float = 0.1
    command_name: str = "base_velocity"
    phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)  # FL+RR / FR+RL


def _gait_phase(env, period: float, frequency: float | None):
    """Shared episode clock in [0, 1), reset by the environment episode buffer."""
    if frequency is None:
        if not math.isfinite(period) or period <= 0:
            raise ValueError("gait period must be finite and positive")
        frequency = 1.0 / period
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("gait frequency must be finite and positive")
    return torch.remainder(env.episode_length_buf.to(torch.float32) * (env.step_dt * frequency), 1.0)


def _pure_x_command_mask(command, command_threshold: float, pure_x_tolerance: float | None):
    """Longitudinal command gate shared by the reward and its phase observation."""
    if pure_x_tolerance is None:
        pure_x_tolerance = command_threshold
    if not math.isfinite(pure_x_tolerance) or not 0 <= pure_x_tolerance <= command_threshold:
        raise ValueError("pure_x_tolerance must lie between zero and command_threshold")
    return (command[:, 0].abs() > command_threshold) & (command[:, 1:3].abs() <= pure_x_tolerance).all(-1)


def diagonal_gait_targets(env, gait_cfg: DiagonalGaitCfg):
    """Return phase, activation and FL/FR/RL/RR stance targets.

    The episode clock resets with the environment and continues while disabled.
    Only lateral velocity and yaw commands enable the schedule.
    """
    cfg = gait_cfg
    phase = _gait_phase(env, cfg.period, cfg.frequency)
    if not 0 < cfg.duty_factor < 1:
        raise ValueError("gait duty_factor must lie between 0 and 1")
    if not math.isfinite(cfg.command_threshold) or cfg.command_threshold < 0:
        raise ValueError("gait command_threshold must be finite and nonnegative")
    command = env.command_manager.get_command(cfg.command_name)
    active = (command[:, 1:3].abs() > cfg.command_threshold).any(dim=-1) & cfg.enabled
    if len(cfg.phase_offsets) != 4 or any(not math.isfinite(x) or not 0 <= x < 1 for x in cfg.phase_offsets):
        raise ValueError("gait phase_offsets must contain four finite values in [0, 1)")
    offsets = phase.new_tensor(cfg.phase_offsets)
    stance = torch.remainder(phase[:, None] + offsets, 1.0) < cfg.duty_factor
    return phase, active, stance


def standing_wheel_contact_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    contact_threshold: float = 1.0,
):
    """Require all four wheels to support the robot inside the command deadband.

    Each unsupported wheel contributes 0.25. This term is independent of gait
    phase and terrain scans, and returns zero whenever a motion command is active.
    """
    if not math.isfinite(command_threshold) or command_threshold < 0:
        raise ValueError("command_threshold must be finite and nonnegative")
    if not math.isfinite(contact_threshold) or contact_threshold <= 0:
        raise ValueError("contact_threshold must be finite and positive")
    command = env.command_manager.get_command(command_name)
    standing = (command[:, :3].abs() <= command_threshold).all(dim=-1)
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    if forces.shape[1] != 4:
        raise ValueError("Expected four wheel contacts ordered FL, FR, RL, RR")
    supported = torch.isfinite(forces).all(dim=-1) & (forces[..., 2] > contact_threshold)
    return torch.where(standing, (~supported).float().mean(dim=-1), 0.0)


def diagonal_gait_contact_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    enabled: bool = True,
    command_name: str = "base_velocity",
    period: float = 0.8,
    frequency: float | None = None,
    duty_factor: float = 0.5,
    phase_offsets: tuple = (0.0, 0.5, 0.5, 0.0),
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg | None = None,
    wheel_sensor_names: list[str] | None = None,
    swing_height: float = 0.10,
    wheel_radius: float = 0.086,
    height_tracking_scale: float = 1.0,
    height_std: float = 0.05,
):
    """Contact mismatch plus phase-dependent wheel clearance error; zero when gated off.

    Lateral or yaw commands beyond the deadband activate gait on any terrain,
    with or without vx. Pure-X motion is handled by its own terrain-gated term.
    Wheel order is FL/FR/RL/RR. Swing clearance above the highest valid local
    scan hit follows height*sin(pi*s)^2, where s runs from zero to one during
    swing. This prevents mixed upper/lower tread samples from lowering the
    target into a step when turning. Supported stance wheels need no height cost.
    Height cost is u^2/(1+u^2), where u=error/height_std. The long tail
    preserves reward differences while approaching a high step from below.
    Support requires upward normal force; touching a riser is not support.
    The result lies in [0, 1 + height_tracking_scale].
    """
    cfg = DiagonalGaitCfg(
        enabled=enabled,
        command_name=command_name,
        period=period,
        frequency=frequency,
        duty_factor=duty_factor,
        phase_offsets=phase_offsets,
        command_threshold=command_threshold,
    )
    phase, active, stance = diagonal_gait_targets(env, cfg)
    penalty, _ = _gait_tracking_penalties(
        env,
        phase,
        stance,
        sensor_cfg,
        contact_threshold,
        asset_cfg,
        wheel_sensor_names,
        swing_height,
        wheel_radius,
        height_tracking_scale,
        duty_factor,
        phase_offsets,
        height_std,
    )
    return torch.where(active, penalty, 0.0)


def _gait_tracking_penalties(
    env,
    phase,
    stance,
    sensor_cfg,
    contact_threshold,
    asset_cfg,
    wheel_sensor_names,
    swing_height,
    wheel_radius,
    height_tracking_scale,
    duty_factor,
    phase_offsets,
    height_std,
    diagnostics: dict | None = None,
):
    """Return gait and rolling costs from one sensor read, in FL/FR/RL/RR order."""
    if not math.isfinite(contact_threshold) or contact_threshold <= 0:
        raise ValueError("contact_threshold must be finite and positive")
    if not math.isfinite(height_tracking_scale) or height_tracking_scale < 0:
        raise ValueError("height_tracking_scale must be finite and nonnegative")
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    if forces.shape[1] != 4:
        raise ValueError("Expected four wheel contacts ordered FL, FR, RL, RR")
    finite_forces = torch.isfinite(forces).all(dim=-1)
    supported = finite_forces & (forces[..., 2] > contact_threshold)
    touching = ~finite_forces | (torch.linalg.vector_norm(forces, dim=-1) > contact_threshold)
    penalty = torch.where(stance, ~supported, touching).float().mean(dim=-1)
    rolling_penalty = (~supported).float().mean(dim=-1)
    if diagnostics is not None:
        diagnostics.update(stance=stance, supported=supported, touching=touching)
    if wheel_sensor_names is not None:
        if asset_cfg is None or len(wheel_sensor_names) != 4 or len(set(wheel_sensor_names)) != 4:
            raise ValueError("Height tracking requires four wheel scanners and ordered wheel bodies")
        if not math.isfinite(swing_height) or swing_height <= 0:
            raise ValueError("swing_height must be finite and positive")
        if not math.isfinite(wheel_radius) or wheel_radius < 0:
            raise ValueError("wheel_radius must be finite and nonnegative")
        if not math.isfinite(height_std) or height_std <= 0:
            raise ValueError("height_std must be finite and positive")
        wheel_z = env.scene[asset_cfg.name].data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
        if wheel_z.shape[1] != 4:
            raise ValueError("Expected four wheel bodies ordered FL, FR, RL, RR")
        ground, upper, valid_scans = [], [], []
        for name in wheel_sensor_names:
            hits = env.scene.sensors[name].data.ray_hits_w.torch
            heights = hits[..., 2]
            finite = torch.isfinite(hits).all(dim=-1)
            valid = finite.any(dim=-1)
            ground.append(heights.masked_fill(~finite, 0.0).sum(-1) / finite.sum(-1).clamp(min=1))
            upper.append(torch.where(valid, heights.masked_fill(~finite, -torch.inf).amax(-1), 0.0))
            valid_scans.append(valid)
        ground = torch.stack(ground, dim=1)
        # Both maneuver and longitudinal gait must clear the same local tread.
        # The mean is only used for unsupported stance/rolling height guidance.
        swing_ground = torch.stack(upper, dim=1)
        bottom_z = wheel_z - wheel_radius
        wheel_phase = torch.remainder(phase[:, None] + phase.new_tensor(phase_offsets), 1.0)
        swing_phase = ((wheel_phase - duty_factor) / (1.0 - duty_factor)).clamp(0.0, 1.0)
        target = torch.where(stance, ground, swing_ground + swing_height * torch.sin(math.pi * swing_phase).square())
        # Exponential errors round to 1 for large stair clearances in float32,
        # so lifting a blocked wheel can otherwise yield no reward improvement.
        height_error_sq = ((bottom_z - target) / height_std).square()
        rolling_error_sq = ((bottom_z - ground) / height_std).square()
        height_error = height_error_sq / (1.0 + height_error_sq)
        rolling_error = rolling_error_sq / (1.0 + rolling_error_sq)
        # A supported wheel already follows the ground. Do not pull it down to
        # a scan mean that may lie below the tread supporting it on a step.
        height_error = torch.where(stance & supported, 0.0, height_error)
        rolling_error = torch.where(supported, 0.0, rolling_error)
        valid = torch.stack(valid_scans, dim=1)
        if diagnostics is not None:
            diagnostics.update(bottom_z=bottom_z, target_z=target, swing_ground=swing_ground, valid_height=valid)
        # Missing scans contribute no height term; contact costs still apply.
        # Keep a fixed four-wheel denominator to avoid amplifying the remaining rays.
        height_penalty = torch.nan_to_num(height_error, nan=1.0).masked_fill(~valid, 0.0).mean(-1)
        penalty = penalty + height_tracking_scale * height_penalty
        rolling_penalty = rolling_penalty + height_tracking_scale * torch.nan_to_num(
            rolling_error, nan=1.0
        ).masked_fill(~valid, 0.0).mean(-1)
    return penalty, rolling_penalty


class pure_x_terrain_gait_penalty(ManagerTermBase):
    """Pure-X motion: four-wheel rolling on smooth scans, timed gait near sharp edges.

    Requires vx outside the deadband and both vy/yaw inside it, so this term
    never overlaps maneuver gait when their command thresholds match.
    Valid second differences in the main scan blend rolling and gait costs
    between step_threshold and step_full_height. No valid triples means unknown
    terrain and no mode penalty. Swing targets clear the highest wheel-scan hit.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["height_sensor_cfg"].name]
        self._terrain_edges = _GridScanEdgeDetector(sensor.cfg.pattern_cfg, env.device)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        height_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        wheel_sensor_names: list[str],
        enabled: bool = True,
        command_name: str = "base_velocity",
        command_threshold: float = 0.1,
        pure_x_tolerance: float | None = None,
        step_threshold: float = 0.03,
        step_full_height: float = 0.05,
        period: float = 0.4,
        frequency: float | None = None,
        duty_factor: float = 0.5,
        phase_offsets: tuple = (0.0, 0.5, 0.5, 0.0),
        contact_threshold: float = 1.0,
        swing_height: float = 0.03,
        wheel_radius: float = 0.086,
        height_tracking_scale: float = 1.0,
        height_std: float = 0.05,
    ):
        # Match the maneuver deadband by default so the two terms neither overlap
        # nor leave a gap for longitudinal commands with small yaw/vy components.
        command = env.command_manager.get_command(command_name)
        pure_x = _pure_x_command_mask(command, command_threshold, pure_x_tolerance)
        cfg = DiagonalGaitCfg(
            enabled=enabled,
            command_name=command_name,
            period=period,
            frequency=frequency,
            duty_factor=duty_factor,
            phase_offsets=phase_offsets,
            command_threshold=command_threshold,
        )
        phase, _, gait_stance = diagonal_gait_targets(env, cfg)
        hits = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w.torch
        edge, valid = self._terrain_edges(hits)
        blend = _terrain_step_gate(edge, step_threshold, step_full_height)
        gait_penalty, rolling_penalty = _gait_tracking_penalties(
            env,
            phase,
            gait_stance,
            sensor_cfg,
            contact_threshold,
            asset_cfg,
            wheel_sensor_names,
            swing_height,
            wheel_radius,
            height_tracking_scale,
            duty_factor,
            phase_offsets,
            height_std,
        )
        penalty = torch.lerp(rolling_penalty, gait_penalty, blend)
        return torch.where(pure_x & valid & enabled, penalty, 0.0)


__all__ = [
    "DiagonalGaitCfg",
    "diagonal_gait_targets",
    "standing_wheel_contact_penalty",
    "diagonal_gait_contact_penalty",
    "pure_x_terrain_gait_penalty",
]
