"""当前使用的动作二阶平滑惩罚。"""

import torch
from isaaclab.managers import ManagerTermBase


__all__ = ["action_smoothness_l2"]


class action_smoothness_l2(ManagerTermBase):
    """原始动作二阶差分平方和；每回合前两步不惩罚。"""

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
