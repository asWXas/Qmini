"""Independent AS2W LP-ACRL tasks; original AS2W registrations are untouched."""

import gymnasium as gym

for terrain in ("Flat", "Rough"):
    gym.register(
        id=f"Qmini-Velocity-{terrain}-AS2W-LP-ACRL",
        entry_point=f"{__name__}.env.lp_env:LPACRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env.as2w_env_cfg:AS2W{terrain}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent.rsl_rl_ppo_cfg:AS2W{terrain}PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )

for terrain in ("Flat", "Rough"):
    gym.register(
        id=f"Qmini-Velocity-{terrain}-AS2W-LP-ACRL-MultiCritic",
        entry_point=f"{__name__}.env.lp_env:LPACRLMultiCriticEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env.multi_critic_env_cfg:AS2W{terrain}MultiCriticEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agent.rsl_rl_multi_critic_cfg:AS2W{terrain}MultiCriticRunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )
