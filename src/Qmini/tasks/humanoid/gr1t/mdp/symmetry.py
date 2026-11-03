"""Left-right reflection for GR1T2 PPO observations and actions.

Actor and critic use the same physical reflection rules, including privileged
observations. Scalar invariants are explicitly listed; unknown terms fail loudly.
Term-major histories and explicitly configured time-major histories preserve
chronological order while reflecting each frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from Qmini.assets.gr1t import JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_JOINT_INDEX = [
    JOINT_NAMES.index(("r_" if name.startswith("l_") else "l_") + name[2:])
    if name.startswith(("l_", "r_"))
    else JOINT_NAMES.index(name)
    for name in JOINT_NAMES
]
# Reflection across the body's sagittal plane (y -> -y), verified against
# GR1T2_new.urdf joint axes and mirrored shoulder mounting rotations.
# Rotation axes are axial vectors: pitch (+Y) keeps its sign; roll (+X)
# and yaw (+Z) reverse. Central waist joints do not swap, but still reflect.
_JOINT_AXIS_SIGNS = {
    "hip_roll": -1,
    "hip_yaw": -1,
    "hip_pitch": 1,
    "knee_pitch": 1,
    "ankle_pitch": 1,
    "ankle_roll": -1,
    "shoulder_pitch": 1,
    "shoulder_roll": -1,
    "shoulder_yaw": -1,
    "elbow_pitch": 1,
    "wrist_yaw": -1,
    "joint_waist_yaw": -1,
    "joint_waist_pitch": 1,
    "joint_waist_roll": -1,
}
_JOINT_SIGN = [_JOINT_AXIS_SIGNS[name[2:] if name.startswith(("l_", "r_")) else name] for name in JOINT_NAMES]

# Direction-free scalars stay unchanged; per-foot scalars must still swap sides.
_INVARIANT_WIDTHS = {"base_mass_ratio": 1}
_JOINT_TERMS = {"joint_pos", "joint_pos_err", "joint_vel", "actions"}

# Polar vectors: (x, -y, z). Axial vectors: (-x, y, -z).
_VECTOR_SIGNS = {
    "gait_clock": (-1, -1),  # Left/right exchange advances the clock by half a cycle.
    "base_com_offset": (1, -1, 1),
    "base_external_wrench": (1, -1, 1, -1, 1, -1),
    "base_lin_vel": (1, -1, 1),
    "base_ang_vel": (-1, 1, -1),
    "torso_ang_vel": (-1, 1, -1),
    "base_projected_gravity": (1, -1, 1),
    "torso_projected_gravity": (1, -1, 1),
    "velocity_commands": (1, -1, -1),
}
_FEET_TERMS = {
    "air_time",
    "feet_air_time",
    "feet_height",
    "contact_forces",
    "feet_contact_state",
    "feet_static_friction",
    "feet_dynamic_friction",
}
_PHASE_TERMS = {"gait_phase", "gait_phase_h", "gait_phase_l"}


def _mirror_joints(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} GR1T2 joints, got {values.shape[-1]}")
    indices = torch.tensor(_JOINT_INDEX, device=values.device)
    signs = values.new_tensor(_JOINT_SIGN)
    return values.index_select(-1, indices) * signs


def _mirror_height_scan(
    env: ManagerBasedRLEnv, values: torch.Tensor, sensor_name: str = "height_scanner"
) -> torch.Tensor:
    pattern = getattr(env.unwrapped.cfg.scene, sensor_name).pattern_cfg
    nx = math.floor(pattern.size[0] / pattern.resolution + 0.5) + 1
    ny = math.floor(pattern.size[1] / pattern.resolution + 0.5) + 1
    if values.shape[-1] != nx * ny:
        raise ValueError(f"Height scan has {values.shape[-1]} rays; expected {nx * ny}")
    if pattern.ordering == "xy":
        return values.reshape(-1, ny, nx).flip(1).reshape_as(values)
    if pattern.ordering == "yx":
        return values.reshape(-1, nx, ny).flip(2).reshape_as(values)
    raise ValueError(f"Unsupported height scan ordering: {pattern.ordering}")


def _mirror_term(env: ManagerBasedRLEnv, name: str, values: torch.Tensor) -> torch.Tensor:
    if name in _INVARIANT_WIDTHS:
        if values.shape[-1] != _INVARIANT_WIDTHS[name]:
            raise ValueError(f"Unexpected invariant observation width: {name}")
        return values.clone()
    if name in _VECTOR_SIGNS:
        signs = _VECTOR_SIGNS[name]
        if values.shape[-1] != len(signs):
            raise ValueError(f"Observation {name} must have {len(signs)} values per frame")
        return values * values.new_tensor(signs)
    if name in _JOINT_TERMS:
        return _mirror_joints(values)
    if name in _FEET_TERMS:
        if values.shape[-1] != 2:
            raise ValueError(f"Observation {name} must contain left and right feet")
        return values.flip(-1)
    if name == "height_scan":
        return _mirror_height_scan(env, values)
    if name == "feet_scan":
        if values.shape[-1] != 18:
            raise ValueError("Foot scans must contain 9 left and 9 right rays")
        left, right = values.split(9, dim=-1)
        return torch.cat(
            (
                _mirror_height_scan(env, right, "height_scanner_right"),
                _mirror_height_scan(env, left, "height_scanner_left"),
            ),
            dim=-1,
        )
    if name in _PHASE_TERMS:
        if values.shape[-1] != 2:
            raise ValueError(f"Observation {name} must contain sine and cosine")
        return -values  # Exchange left/right phase by half a cycle.
    raise ValueError(f"No left-right reflection defined for observation {name}")


def _frame_width(env: ManagerBasedRLEnv, name: str) -> int:
    """Single-frame width, independent of each term's history length."""
    if name in _INVARIANT_WIDTHS:
        return _INVARIANT_WIDTHS[name]
    if name in _VECTOR_SIGNS:
        return len(_VECTOR_SIGNS[name])
    if name in _JOINT_TERMS:
        return len(JOINT_NAMES)
    if name in _FEET_TERMS | _PHASE_TERMS:
        return 2
    if name == "feet_scan":
        return 18
    if name == "height_scan":
        pattern = env.unwrapped.cfg.scene.height_scanner.pattern_cfg
        return math.prod(math.floor(size / pattern.resolution + 0.5) + 1 for size in pattern.size)
    raise ValueError(f"No left-right reflection defined for observation {name}")


def _mirror_group(env: ManagerBasedRLEnv, group_name: str, values: torch.Tensor) -> torch.Tensor:
    manager = env.unwrapped.observation_manager
    names = manager.active_terms[group_name]
    shapes = manager.group_obs_term_dim[group_name]
    group_cfg = getattr(env.unwrapped.cfg.observations, group_name)
    time_major = (
        getattr(group_cfg, "history_order", "term") == "time"
        and (group_cfg.history_length or 0) > 0
        and group_cfg.flatten_history_dim
        and group_cfg.concatenate_terms
        and group_cfg.concatenate_dim == -1
    )
    if time_major:
        frame_width = sum(_frame_width(env, name) for name in names)
        if values.shape[-1] != frame_width * group_cfg.history_length:
            raise ValueError(f"Unexpected time-major observation width: {group_name}")
        return _mirror_frames(env, names, values.reshape(-1, frame_width)).reshape_as(values)
    mirrored = values.clone()
    offset = 0
    for name, shape in zip(names, shapes, strict=True):
        width = math.prod(shape)
        base_width = _frame_width(env, name)
        if width % base_width:
            raise ValueError(f"Observation {group_name}/{name} has incompatible width {width}")
        # Flattened term history is [frame0 joints..., frame1 joints..., ...].
        # Reflect each full frame; never reverse time or mix historical frames.
        term = values[:, offset : offset + width].reshape(-1, base_width)
        mirrored[:, offset : offset + width] = _mirror_term(env, name, term).reshape(-1, width)
        offset += width
    if offset != values.shape[-1]:
        raise ValueError(f"Observation {group_name} has {values.shape[-1]} values, expected {offset}")
    return mirrored


def _mirror_frames(env: ManagerBasedRLEnv, names: list[str], values: torch.Tensor) -> torch.Tensor:
    mirrored = values.clone()
    offset = 0
    for name in names:
        width = _frame_width(env, name)
        mirrored[:, offset : offset + width] = _mirror_term(env, name, values[:, offset : offset + width])
        offset += width
    if offset != values.shape[-1]:
        raise ValueError(f"Single observation frame has {values.shape[-1]} values, expected {offset}")
    return mirrored


def _mirror_policy_frame(env: ManagerBasedRLEnv, values: torch.Tensor) -> torch.Tensor:
    """Mirror auxiliary next-state targets, regardless of policy history settings."""
    return _mirror_frames(env, env.unwrapped.observation_manager.active_terms["policy"], values)


def _mirror_auxiliary(env: ManagerBasedRLEnv, name: str, values: torch.Tensor) -> torch.Tensor:
    if name in {"dreamwaq_next", "him_next"}:
        return _mirror_policy_frame(env, values)
    if name in {"dreamwaq_velocity", "him_velocity"}:
        return _mirror_term(env, "base_lin_vel", values)
    if name in {"dreamwaq_valid", "him_valid"}:
        return values.clone()
    return _mirror_group(env, name, values)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return original and reflected samples in that order for RSL-RL PPO."""
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        batch_size = obs.batch_size[0]
        for group_name in obs.keys():
            obs_aug[group_name][batch_size:] = _mirror_auxiliary(env, group_name, obs[group_name])

    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, _mirror_joints(actions)), dim=0)
    return obs_aug, actions_aug
