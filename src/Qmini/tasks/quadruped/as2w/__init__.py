"""AS2W velocity-tracking tasks."""

import gymnasium as gym

for terrain in ("Flat", "Rough"):
    gym.register(
        id=f"Qmini-Velocity-{terrain}-AS2W",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env.as2w_env_cfg:AS2W{terrain}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent.rsl_rl_ppo_cfg:AS2W{terrain}PPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{__name__}.agent.rsl_rl_distillation_cfg:AS2W{terrain}DistillationRunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )

for terrain in ("Flat", "Rough"):
    gym.register(
        id=f"Qmini-Velocity-{terrain}-AS2W-MultiCritic",
        entry_point="Qmini.algo.multi_critic.manager.rl_env:MultiCriticManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env.multi_critic_env_cfg:AS2W{terrain}MultiCriticEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agent.rsl_rl_multi_critic_cfg:AS2W{terrain}MultiCriticRunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )
