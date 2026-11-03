"""One velocity command per curriculum task/episode; no mid-episode reclassification."""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class LPVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._gamepad = None
        if cfg.xbox:
            from .gamepad import XboxGamepad

            self._gamepad = XboxGamepad(cfg.gamepad_path, cfg.gamepad_deadzone)

    def _update_gamepad(self):
        ranges = self.cfg.ranges
        bounds = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
        for i, (value, limits) in enumerate(zip(self._gamepad.read(), bounds, strict=True)):
            self.vel_command_b[:, i] = value * (limits[1] if value >= 0 else -limits[0])
        self.is_heading_env[:] = False
        self.is_standing_env[:] = torch.linalg.vector_norm(self.vel_command_b, dim=1) < 1e-6

    def _resample_command(self, env_ids):
        if self.cfg.xbox:
            self.vel_command_b[env_ids] = 0.0
            self.is_heading_env[env_ids] = False
            self.is_standing_env[env_ids] = True
            return
        controller = self._env.lp_curriculum
        self.vel_command_b[env_ids] = controller.commands[env_ids]
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False

    def compute(self, dt):
        # Resample exclusively through reset(); the task ID stays valid for the
        # entire measured episode, even if its maximum length changes in config.
        self._update_metrics()
        if self.cfg.xbox:
            self._update_gamepad()


@configclass
class LPVelocityCommandCfg(UniformVelocityCommandCfg):
    xbox: bool = False
    gamepad_path: str = "/dev/input/js0"
    gamepad_deadzone: float = 0.1
    class_type = LPVelocityCommand
    heading_command = False
    rel_heading_envs = 0.0
    rel_standing_envs = 0.0
    resampling_time_range = (1.e9, 1.e9)
