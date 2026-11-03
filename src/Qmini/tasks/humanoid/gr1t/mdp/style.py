"""GR1T style costs: leg shape, cross-body swing, posture, and cycle symmetry.

All terms return nonnegative costs for negative reward weights. The existing gait
owns timing and foot clearance. Style leaves tolerance for balance, terrain, and
turning instead of specifying a second joint-space walking trajectory.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .gait import _activation, _reward_cfg, _smoothstep, _swing_profile

__all__ = ["leg_style_penalty", "arm_body_coordination_penalty", "upper_body_style_penalty", "cycle_symmetry_penalty"]


def _context(env, gait_reward_name):
    cfg = _reward_cfg(env, gait_reward_name)
    if cfg is None:
        raise ValueError(f"Style requires the configured gait term {gait_reward_name!r}")
    params = cfg.params
    command_name = params.get("command_name", "base_velocity")
    command = env.command_manager.get_command(command_name)
    gate = _activation(
        env, command_name, params.get("command_threshold", 0.1), params.get("command_transition_width", 0.05)
    )
    if cfg.weight == 0.0:
        gate = torch.zeros_like(gate)
    return command, gate, params


def _straight_gate(command):
    """Sagittal style fades smoothly during lateral travel and turning."""
    forward = _smoothstep((command[:, 0].abs() - 0.1) / 0.2)
    return forward * torch.exp(-(command[:, 1] / 0.4).square() - (command[:, 2] / 0.8).square())


def _soft_error(error, tolerance):
    """Zero within tolerance; continuous first derivative at its boundary."""
    return torch.relu(error.abs() - tolerance).square()


def _relative_joints(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )


def _nonnegative(*values):
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Style weights, limits, and tolerances must be finite and nonnegative")


def leg_style_penalty(
    env,
    knee_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    gait_reward_name: str = "gait",
    stance_knee_limit: float = 0.12,
    fast_stance_knee_limit: float = 0.45,
    swing_knee_min: float = 0.35,
    fast_swing_knee_min: float = 0.55,
    swing_weight: float = 0.25,
    hip_weight: float = 0.1,
    pitch_cfg: SceneEntityCfg | None = None,
    pitch_weight: float = 0.5,
    pitch_tolerance: float = 0.15,
):
    """Support-leg extension and mid-swing flexion, without joint-activity ratios.

    Knees: [L, R]; hips: [L roll, L yaw, R roll, R yaw]. Actual contact controls
    support shaping, released during the planned swing even before contact ends.
    Swing shaping uses the SAME clock/duty factor as gait and vanishes near
    lift-off/touchdown. Hip roll/yaw tolerances widen for vy/wz.

    Optional pitches: [L hip, L knee, L ankle, R hip, R knee, R ankle].
    GR1T2 has aligned +Y pitch axes and zero joint-origin rotations, so their
    sum is the foot pitch relative to the hip's parent frame. During straight
    travel, softly keep this sum near zero in airborne mid-swing. This couples
    ankle compensation to hip/knee motion without prescribing their amplitudes.
    It is a sagittal posture approximation, not world/terrain foot alignment;
    stance, lift-off, touchdown, lateral travel, and turns remain free to adapt.
    """
    _nonnegative(
        stance_knee_limit, fast_stance_knee_limit, swing_knee_min, fast_swing_knee_min,
        swing_weight, hip_weight, pitch_weight, pitch_tolerance,
    )
    command, gate, params = _context(env, gait_reward_name)
    swing_time, swing_weight_phase = _swing_profile(
        env, params.get("period", 0.8), params.get("duty_factor", 0.6), params.get("phase_transition", 0.1)
    )
    mid_swing = torch.sin(math.pi * swing_time).square()
    knee = env.scene[knee_cfg.name].data.joint_pos.torch[:, knee_cfg.joint_ids]
    hips = _relative_joints(env, hip_cfg).reshape(-1, 2, 2)
    sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    support = _smoothstep((contact_time - 0.04) / 0.08)
    if knee.shape[-1] != 2 or contact_time.shape[-1] != 2:
        raise ValueError("Leg style requires left/right knee and contact selections")
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    fast = _smoothstep((speed - 1.0) / 1.5)
    limit = stance_knee_limit + fast * (fast_stance_knee_limit - stance_knee_limit)
    # Allow cushioning while both feet support the body.
    limit = limit[:, None] + support.flip(-1) * torch.relu(0.30 - limit[:, None])
    extension = support * (1.0 - swing_weight_phase) * (torch.relu(knee - limit).square() + torch.relu(-knee).square())
    minimum = swing_knee_min + fast * (fast_swing_knee_min - swing_knee_min)
    flexion = mid_swing * (contact_time <= 0.0) * torch.relu(minimum[:, None] - knee).square()
    hip_tolerance = torch.stack(
        (0.12 + 0.35 * _smoothstep(command[:, 1].abs()), 0.15 + 0.30 * _smoothstep(command[:, 2].abs())), dim=-1
    )
    hip_cost = _soft_error(hips, hip_tolerance[:, None]).mean(dim=(1, 2))
    cost = extension.mean(-1) + swing_weight * flexion.mean(-1) + hip_weight * hip_cost
    if pitch_cfg is not None and pitch_weight > 0.0:
        pitch = env.scene[pitch_cfg.name].data.joint_pos.torch[:, pitch_cfg.joint_ids]
        if pitch.shape[-1] != 6:
            raise ValueError("Leg pitch coordination requires hip/knee/ankle pitches ordered left, right")
        foot_pitch = pitch.reshape(-1, 2, 3).sum(-1)
        pitch_cost = _soft_error(foot_pitch, pitch_tolerance)
        swing_gate = mid_swing * swing_weight_phase * (contact_time <= 0.0)
        cost = cost + pitch_weight * (pitch_cost * swing_gate).mean(-1) * _straight_gate(command)
    return cost * gate


def arm_body_coordination_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    gait_reward_name: str = "gait",
    min_arm_scale: float = 0.4,
    max_arm_scale: float = 1.0,
    position_tolerance: float = 0.05,
    velocity_weight: float = 0.15,
    waist_weight: float = 0.5,
    waist_yaw_gain: float = 0.2,
    max_waist_yaw: float = 0.15,
):
    """Opposite arm follows hip position AND velocity; waist follows the same swing.

    Joint order: [L hip pitch, R hip pitch, L shoulder pitch, R shoulder pitch,
    waist yaw]. GR1T2 pitch axes have matching signs. Remove common hip flexion
    so crouching does not push both arms forward. No minimum arm amplitude is
    imposed: gait generates leg motion and this term couples the arms to it.
    """
    _nonnegative(min_arm_scale, max_arm_scale, position_tolerance, velocity_weight, waist_weight, waist_yaw_gain)
    if not math.isfinite(max_waist_yaw) or max_waist_yaw <= 0.0:
        raise ValueError("max_waist_yaw must be finite and positive")
    command, gate, params = _context(env, gait_reward_name)
    q = _relative_joints(env, asset_cfg)
    if q.shape[-1] != 5:
        raise ValueError("Arm/body coordination requires two hips, two shoulders, and waist yaw")
    dq = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    hips = q[:, :2] - q[:, :2].mean(-1, keepdim=True)
    hip_rates = dq[:, :2] - dq[:, :2].mean(-1, keepdim=True)
    scale = min_arm_scale + _smoothstep((command[:, 0].abs() - 0.1) / 1.9) * (max_arm_scale - min_arm_scale)
    target = scale[:, None] * hips.flip(-1)
    target_rate = scale[:, None] * hip_rates.flip(-1)
    position_cost = _soft_error(q[:, 2:4] - target, position_tolerance).mean(-1)
    # Convert rad/s mismatch to a phase-comparable angular deviation.
    time_scale = params.get("period", 0.8) / (2.0 * math.pi)
    rate_cost = _soft_error((dq[:, 2:4] - target_rate) * time_scale, position_tolerance).mean(-1)
    yaw_target = max_waist_yaw * torch.tanh(-waist_yaw_gain * (hips[:, 0] - hips[:, 1]) / max_waist_yaw)
    yaw_cost = _soft_error(q[:, 4] - yaw_target, position_tolerance)
    return (position_cost + velocity_weight * rate_cost + waist_weight * yaw_cost) * gate * _straight_gate(command)


def upper_body_style_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    gait_reward_name: str = "gait",
    min_elbow_offset: float = -0.1,
    max_elbow_offset: float = -0.6,
    arm_tolerance: float = 0.1,
    elbow_tolerance: float = 0.15,
    waist_tolerance: float = 0.08,
):
    """Relaxed arms and torso, leaving shoulder pitch and waist yaw to coordination.

    Joint order: [L shoulder roll/yaw, L elbow pitch, L wrist yaw, R equivalents,
    waist pitch/roll]. Soft tolerances permit small balance corrections. Signed
    elbow targets follow the negative-flexion convention of GR1T2_new.urdf.
    """
    _nonnegative(arm_tolerance, elbow_tolerance, waist_tolerance)
    if not math.isfinite(min_elbow_offset) or not math.isfinite(max_elbow_offset):
        raise ValueError("Elbow targets must be finite")
    command, gate, _ = _context(env, gait_reward_name)
    q = _relative_joints(env, asset_cfg)
    if q.shape[-1] != 10:
        raise ValueError("Upper-body style requires eight arm joints and waist pitch/roll")
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    elbow = min_elbow_offset + _smoothstep((speed - 0.1) / 1.9) * (max_elbow_offset - min_elbow_offset)
    target = torch.zeros_like(q)
    target[:, [2, 6]] = elbow[:, None]
    tolerance = torch.full_like(q, arm_tolerance)
    tolerance[:, [2, 6]] = elbow_tolerance
    tolerance[:, 8:] = waist_tolerance + 0.1 * _smoothstep(command[:, 1].abs())[:, None]
    errors = _soft_error(q - target, tolerance)
    return (errors[:, :8].mean(-1) + errors[:, 8:].mean(-1)) * gate


class cycle_symmetry_penalty(ManagerTermBase):
    """Compare mirrored mean posture and centered RMS amplitude over a COMPLETE cycle.

    Never compare left/right knee angles at the same instant: stance and swing
    legitimately differ. One rolling window follows the configured gait period.
    Warm-up, reset, command changes, turns, and asymmetric terrain suppress this
    style cost. Joint activity is driven by gait, not by an artificial minimum.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        selection = cfg.params["asset_cfg"].joint_ids
        ids = list(range(len(self.asset.joint_names)))[selection] if isinstance(selection, slice) else selection
        names = [self.asset.joint_names[i] for i in ids]
        partners = [("r_" if n.startswith("l_") else "l_") + n[2:] if n.startswith(("l_", "r_")) else n for n in names]
        self._mirror = torch.tensor([names.index(n) for n in partners], device=env.device)
        self._signs = torch.tensor([-1.0 if n.endswith(("_roll", "_yaw")) else 1.0 for n in names], device=env.device)
        self._width = len(names)
        self._history = None
        self._length = 0
        self._last_step = None
        self._result = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        if self._history is not None:
            self._history[:, env_ids] = 0.0
            self._sum[env_ids] = 0.0
            self._sum_squares[env_ids] = 0.0
            self._count[env_ids] = 0
        self._result[env_ids] = 0.0

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        sensor_names: list[str],
        gait_reward_name: str = "gait",
        posture_tolerance: float = 0.05,
        amplitude_tolerance: float = 0.05,
        command_tolerance: float = 0.15,
        terrain_tolerance: float = 0.05,
    ):
        command, gate, params = _context(env, gait_reward_name)
        period = params.get("period", 0.8)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("Cycle symmetry requires a finite positive gait period")
        _nonnegative(posture_tolerance, amplitude_tolerance)
        if any(not math.isfinite(value) or value <= 0.0 for value in (command_tolerance, terrain_tolerance)):
            raise ValueError("Command and terrain tolerances must be finite and positive")
        if len(sensor_names) != 2 or len(set(sensor_names)) != 2:
            raise ValueError("Cycle symmetry requires distinct left and right terrain scanners")
        length = max(2, math.ceil(period / env.step_dt))
        if length != self._length:
            self._length = length
            self._history = torch.zeros((length, env.num_envs, self._width + 3), device=env.device)
            self._sum = torch.zeros_like(self._history[0])
            self._sum_squares = torch.zeros_like(self._sum)
            self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._head = 0
            self._last_step = None
        # Reward-manager calls once per simulation step; debug reads must not advance history.
        if self._last_step == env.common_step_counter:
            return self._result.clone()
        self._last_step = env.common_step_counter
        q = _relative_joints(env, asset_cfg)
        sample = torch.cat((q, command[:, :3]), dim=-1)
        previous = self._history[self._head]
        self._sum += sample - previous
        self._sum_squares += sample.square() - previous.square()
        previous.copy_(sample)
        self._head = (self._head + 1) % length
        self._count = torch.where(gate > 0.0, (self._count + 1).clamp(max=length), 0)

        mean = self._sum / length
        mean_square = (self._sum_squares / length).clamp_min(0.0)
        q_mean = mean[:, : self._width]
        # Center the motion before measuring amplitude; a posture offset should
        # be charged by mean_error only, not charged again as unequal swing.
        rms = (mean_square[:, : self._width] - q_mean.square()).clamp_min(0.0).sqrt()
        mean_error = q_mean - q_mean[:, self._mirror] * self._signs
        amplitude_error = rms - rms[:, self._mirror]
        cost = _soft_error(mean_error, posture_tolerance).mean(-1)
        cost += _soft_error(amplitude_error, amplitude_tolerance).mean(-1)
        command_variance = (mean_square[:, -3:] - mean[:, -3:].square()).clamp_min(0.0).sum(-1)
        command_change = (command[:, :3] - mean[:, -3:]).square().sum(-1)
        stable = torch.exp(-(command_variance + command_change) / command_tolerance**2)

        ground, valid = [], []
        for name in sensor_names:
            hits = env.scene.sensors[name].data.ray_hits_w.torch
            finite = torch.isfinite(hits).all(-1)
            z = hits[..., 2].masked_fill(~finite, -torch.inf).amax(-1)
            ground.append(torch.where(finite.all(-1), z, 0.0))
            valid.append(finite.all(-1))
        terrain_gate = torch.exp(-((ground[0] - ground[1]) / terrain_tolerance).square()) * valid[0] * valid[1]
        self._result = cost * gate * _straight_gate(command) * stable * terrain_gate * (self._count >= length)
        return self._result.clone()
