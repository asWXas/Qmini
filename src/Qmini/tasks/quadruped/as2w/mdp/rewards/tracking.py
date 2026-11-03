"""Velocity tracking costs that retain sensitivity to large errors."""

from isaaclab.managers import SceneEntityCfg

__all__ = ["lin_vel_xy_tracking_error_l2", "ang_vel_z_tracking_error_l2"]


def lin_vel_xy_tracking_error_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    """Squared XY velocity error in the body frame, for a negative reward weight.

    Uses the same command and measured velocity as track_lin_vel_xy_exp.
    No exponential or clipping: stopping, reversing, and overshooting remain
    distinguishable even at large errors. Zero commands penalize XY drift;
    vertical velocity and yaw are left to their existing reward terms.
    """
    command = env.command_manager.get_command(command_name)[:, :2]
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, :2]
    return (command - velocity).square().sum(dim=-1)


def ang_vel_z_tracking_error_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    """Squared body-frame yaw-rate error, for a negative reward weight.

    Matches track_ang_vel_z_exp's velocity convention. Zero commands penalize
    unwanted turning; opposite-direction and excessive yaw rates increase the
    cost without saturation. Roll and pitch rates do not contribute.
    """
    command = env.command_manager.get_command(command_name)[:, 2]
    velocity = env.scene[asset_cfg.name].data.root_ang_vel_b.torch[:, 2]
    return (command - velocity).square()
