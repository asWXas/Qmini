import gymnasium as gym

from . import agents

gym.register(
    id="Qmini-Flat-HIMLoco",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_himloco_cfg:HIMLocoRunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-HIMLoco-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_himloco_cfg:HIMLocoRunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-DreamWaQ",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dream_waq_cfg:DreamWaQRunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-DreamWaQ-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dream_waq_cfg:DreamWaQRunnerCfg",
    },
)

# =============================================================================
# Train
# =============================================================================

gym.register(
    id="Qmini-Flat-TCN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatTCNEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_tcn_cfg:TCNRunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-Play-TCN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatTCNEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_tcn_cfg:TCNRunnerCfg",
    },
)


# uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Qmini-Flat --num_envs 1
# =============================================================================
# Play / Evaluation
# =============================================================================

gym.register(
    id="Qmini-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
