# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action smoothness and mechanical power costs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["action_smoothness_l2", "joint_mechanical_power_l1"]


class action_smoothness_l2(ManagerTermBase):
    """Raw-action second difference squared sum; no penalty for the first two steps.

    Uses a_t - 2*a_{t-1} + a_{t-2}, without dt normalization, matching
    action_rate_l2's discrete-action convention. Configure a negative weight.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._older_action = torch.zeros_like(env.action_manager.action)
        self._value = torch.zeros_like(env.action_manager.action[:, 0])
        self._last_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._older_action[ids] = 0.0
        self._value[ids] = 0.0

    def __call__(self, env):
        if self._last_step != env.common_step_counter:
            action = env.action_manager.action
            previous = env.action_manager.prev_action
            difference = action - 2.0 * previous + self._older_action
            self._value = difference.square().sum(dim=1)
            self._value.masked_fill_(env.episode_length_buf < 3, 0.0)
            self._older_action.copy_(previous)
            self._last_step = env.common_step_counter
        return self._value.clone()


def joint_mechanical_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum absolute mechanical joint power in watts for the selected motors.

    GR1T2 uses explicit DCMotor actuators, so ``applied_torque`` is the torque
    actually sent to the simulator. RewardManager multiplies this rate by dt.
    Absolute power also charges for braking, rather than cancelling it against
    positive work at another joint.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)
