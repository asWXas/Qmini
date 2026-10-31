"""OpenArm reset events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_arm_joints_near_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Perturb the default arm pose, with zero velocity and matching servo targets.

    Use a small, collision-checked range for the configured scene. Joint-limit
    clamping alone is not a collision check; validate wider ranges with
    scripts/check_openarm_reset.py before using them for training.
    """
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_ids = asset_cfg.joint_ids
    rows = env_ids if isinstance(env_ids, slice) or isinstance(joint_ids, slice) else env_ids[:, None]
    position = robot.data.default_joint_pos.torch[rows, joint_ids].clone()
    position += torch.empty_like(position).uniform_(*position_range)
    limits = robot.data.soft_joint_pos_limits.torch[rows, joint_ids]
    position.clamp_(limits[..., 0], limits[..., 1])
    velocity = torch.zeros_like(position)
    robot.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target_index(target=position, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_velocity_target_index(target=velocity, joint_ids=joint_ids, env_ids=env_ids)
