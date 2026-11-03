"""GR1T PPO settings shared with the multi-critic specialization."""

from copy import deepcopy

from isaaclab.utils.configclass import configclass

from Qmini.algo.multi_critic.cfg import RslRlMultiCriticPpoAlgorithmCfg
from Qmini.tasks.humanoid.gr1t.agents.rsl_rl_ppo_cfg import PPORunnerCfg as GR1TPPORunnerCfg


def _multi_critic_algorithm() -> RslRlMultiCriticPpoAlgorithmCfg:
    # Keep PPO hyperparameters and symmetry synchronized, retaining the multi-head algorithm.
    params = deepcopy(vars(GR1TPPORunnerCfg().algorithm))
    params.pop("class_name")
    return RslRlMultiCriticPpoAlgorithmCfg(
        **params,
        critic_groups={
            "task": {"weight": 0.50},
            "coordination": {"weight": 0.25},
            "constraint": {"weight": 0.25},
        },
        normalize_critic_weights=False,
        rnd_group=None,
    )


@configclass
class PPORunnerCfg(GR1TPPORunnerCfg):
    """Share GR1T networks and PPO tuning; use task/coordination/constraint value heads."""

    class_name = "OnPolicyRunner"
    experiment_name = "gr1t_flat_multi_critic"
    obs_groups = {"actor": ["critic"], "critic": ["critic"]}
    algorithm = _multi_critic_algorithm()


__all__ = ["PPORunnerCfg"]
