# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# Use the project-specific ID for training; retain the old ID for existing commands.
for task_id in ("Qmini-Lift-Cube-OpenArm", "IsaacContrib-Lift-Cube-OpenArm"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "default_agent": "rsl_rl",
        },
        disable_env_checker=True,
    )
