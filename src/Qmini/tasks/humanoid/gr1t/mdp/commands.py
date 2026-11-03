"""Grouped velocity commands and legacy four-mode sampling for GR1T."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


__all__ = [
    "GroupedVelocityCommand",
    "GroupedVelocityCommandCfg",
    "FourModeVelocityCommand",
    "FourModeVelocityCommandCfg",
]


class FourModeVelocityCommand(UniformVelocityCommand):
    """Sample four motion modes plus an explicit zero command."""

    cfg: FourModeVelocityCommandCfg

    def __init__(self, cfg: FourModeVelocityCommandCfg, env: ManagerBasedEnv):
        probabilities = (
            cfg.prob_x_only,
            cfg.prob_y_only,
            cfg.prob_yaw_only,
            cfg.prob_mixed,
            cfg.prob_zero,
        )
        if any(not math.isfinite(probability) or probability < 0.0 for probability in probabilities):
            raise ValueError("Four velocity-mode probabilities must be finite and non-negative.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Four velocity-mode probabilities must sum to 1.0.")
        if cfg.heading_command or cfg.rel_standing_envs != 0.0:
            raise ValueError("Four-mode sampling requires heading_command=False and rel_standing_envs=0.0.")

        super().__init__(cfg, env)
        self._mode_probabilities = torch.tensor(probabilities, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        super()._resample_command(env_ids)
        # Isaac Lab 3.0 会传入 slice（全部环境）或设备上的索引张量，两者都要支持。
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return
        self.is_standing_env[env_ids] = False

        # Parent samples each component from its configured range. Mask the
        # unused components after drawing one mode independently per env.
        modes = torch.multinomial(self._mode_probabilities, num_envs, replacement=True)
        commands = self.vel_command_b[env_ids]
        commands[modes == 0, 1:] = 0.0  # x only
        commands[modes == 1, 0] = 0.0  # y only
        commands[modes == 1, 2] = 0.0
        commands[modes == 2, :2] = 0.0  # yaw only
        commands[modes == 4, :] = 0.0  # explicit zero command
        # Mode 3 keeps all three sampled components.
        self.vel_command_b[env_ids] = commands


@configclass
class FourModeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Probabilities for x-only, y-only, yaw-only, and mixed modes."""

    class_type: type = FourModeVelocityCommand

    prob_x_only: float = 0.25
    prob_y_only: float = 0.25
    prob_yaw_only: float = 0.25
    prob_mixed: float = 0.25
    prob_zero: float = 0.0


class GroupedVelocityCommand(UniformVelocityCommand):
    """Standing plus X/Y/yaw/mix sampling; heading control applies to yaw/mix only."""

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
        # 0=standing, 1=x, 2=y, 3=yaw, 4=mix.
        self.command_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
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
            for i, (value, bounds) in enumerate(
                zip(values, (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z), strict=True)
            ):
                self.vel_command_b[:, i] = value * (bounds[1] if value >= 0 else -bounds[0])
            self.is_standing_env[:] = torch.linalg.vector_norm(self.vel_command_b, dim=1) < 1e-6
            self.is_heading_env[:] = False
            self.command_type[:] = torch.where(self.is_standing_env, 0, 4)
            return
        super()._update_command()
        # Preserve pure X/Y commands even when heading control is enabled.
        self.vel_command_b[:, 2] = torch.where(self.command_type >= 3, self.vel_command_b[:, 2], 0.0)


@configclass
class GroupedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Movement proportions are conditional on not sampling a standing command."""

    class_type = GroupedVelocityCommand
    heading_command = False
    rel_heading_envs = 0.0
    xbox: bool = False
    gamepad_path: str = "/dev/input/js0"
    gamepad_deadzone: float = 0.1
    moving_proportions: dict[str, float] = {"x": 0.25, "y": 0.25, "yaw": 0.25, "mix": 0.25}
