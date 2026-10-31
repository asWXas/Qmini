"""Multi-Critic PPO variant of the GR1T flat-terrain locomotion task."""

import gymnasium as gym

from . import agents

_ENV_ENTRY_POINT = "Qmini.algo.multi_critic.manager.rl_env:MultiCriticManagerBasedRLEnv"


gym.register(
    id="GR1T-Flat-MultiCritic",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:GR1TMultiCriticFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="GR1T-Flat-MultiCritic-Play",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:GR1TMultiCriticFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


__all__ = []
