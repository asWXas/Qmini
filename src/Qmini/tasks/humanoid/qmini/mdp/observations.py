"""Proprioceptive and privileged observations for Qmini."""

from isaaclab.managers import SceneEntityCfg


def joint_position_execution_error(env, asset_cfg: SceneEntityCfg):
    """Previous position command minus current position, in the selected joint order.

    Observations follow physics, so the actuator target is the command from the
    transition just executed (including action scale, offset, and clipping).
    A newly reset episode has no previous command: return zero for that row.
    """
    asset = env.scene[asset_cfg.name]
    target = asset.actuators.target_command.position.torch[:, asset_cfg.joint_ids]
    position = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error = target - position
    return error.masked_fill((env.episode_length_buf == 0).unsqueeze(-1), 0.0)
