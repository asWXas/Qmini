import gymnasium as gym

from . import agents

# =============================================================================
# UMV - Flat terrain
# =============================================================================
#
# uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task UMV-Flat --num_envs 1
#
# =============================================================================


# ---------------------------------- Train ------------------------------------

gym.register(
    id="UMV-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UMVFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# ------------------------------ Play / Evaluation ----------------------------

gym.register(
    id="UMV-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UMVFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
