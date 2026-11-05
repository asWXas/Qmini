"""Load a frozen AS2W checkpoint for collection."""

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from Qmini.tasks.quadruped.as2w.agent.rsl_rl_ppo_cfg import AS2WRoughGRUPPORunnerCfg, AS2WRoughPPORunnerCfg


def make_agent_config(architecture):
    if architecture not in ("gru", "mlp"):
        raise ValueError(f"Unsupported policy architecture: {architecture}")
    return AS2WRoughGRUPPORunnerCfg() if architecture == "gru" else AS2WRoughPPORunnerCfg()


def load_policy(raw_env, checkpoint, architecture):
    cfg = make_agent_config(architecture)
    cfg.device = raw_env.unwrapped.device
    env = RslRlVecEnvWrapper(raw_env, clip_actions=cfg.clip_actions)
    runner = OnPolicyRunner(env, to_dict(cfg), log_dir=None, device=cfg.device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
        map_location=cfg.device,
    )
    return env, runner.get_inference_policy(device=cfg.device), cfg
