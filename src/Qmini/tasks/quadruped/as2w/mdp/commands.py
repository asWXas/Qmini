"""站立，以及 X / Y / yaw / mix 四类速度命令。"""

import math

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

__all__ = ["GroupedVelocityCommand", "GroupedVelocityCommandCfg"]


class GroupedVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        if not 0 <= cfg.rel_standing_envs <= 1:
            raise ValueError("rel_standing_envs must be in [0, 1]")
        if set(cfg.moving_proportions) != {"x", "y", "yaw", "mix"}:
            raise ValueError("moving_proportions must contain x, y, yaw and mix")
        weights = [cfg.moving_proportions[key] for key in ("x", "y", "yaw", "mix")]
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Movement weights must be finite, non-negative and have a positive sum")
        super().__init__(cfg, env)
        self._moving_weights = torch.tensor(weights, device=self.device, dtype=torch.float)
        self.command_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 0=standing, 1=x, 2=y, 3=yaw, 4=mix.
        self._gamepad = None
        if cfg.xbox:
            from .gamepad import XboxGamepad

            self._gamepad = XboxGamepad(cfg.gamepad_path, cfg.gamepad_deadzone)

    def _resample_command(self, env_ids):
        if self.cfg.xbox:
            self.vel_command_b[env_ids] = 0.0
            self.is_standing_env[env_ids] = True
            self.is_heading_env[env_ids] = False
            self.command_type[env_ids] = 0
            return
        super()._resample_command(env_ids)
        count = self.vel_command_b[env_ids].shape[0]
        if count == 0:
            return
        kind = torch.multinomial(self._moving_weights, count, replacement=True)
        axis = torch.arange(3, device=self.device)
        mask = (kind[:, None] == axis[None, :]) | (kind[:, None] == 3)
        standing = self.is_standing_env[env_ids]
        self.vel_command_b[env_ids] = self.vel_command_b[env_ids] * mask * (~standing[:, None])
        self.command_type[env_ids] = torch.where(standing, 0, kind + 1)
        self.is_heading_env[env_ids] &= (~standing) & (kind >= 2)

    def _update_command(self):
        if self.cfg.xbox:
            values = self._gamepad.read()
            ranges = self.cfg.ranges
            for i, (value, bounds) in enumerate(zip(values, (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z))):
                self.vel_command_b[:, i] = value * (bounds[1] if value >= 0 else -bounds[0])
            self.is_standing_env[:] = torch.linalg.vector_norm(self.vel_command_b, dim=1) < 1e-6
            self.is_heading_env[:] = False
            self.command_type[:] = torch.where(self.is_standing_env, 0, 4)
            return
        super()._update_command()
        # Heading 只生成 yaw/mix 的角速度，保持纯 X/Y 分类。
        self.vel_command_b[:, 2] = torch.where(self.command_type >= 3, self.vel_command_b[:, 2], 0.0)


@configclass
class GroupedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type = GroupedVelocityCommand
    heading_command = False
    rel_heading_envs = 0.0
    xbox: bool = False
    gamepad_path: str = "/dev/input/js0"
    gamepad_deadzone: float = 0.1
    moving_proportions: dict[str, float] = {"x": 0.25, "y": 0.25, "yaw": 0.25, "mix": 0.25}
