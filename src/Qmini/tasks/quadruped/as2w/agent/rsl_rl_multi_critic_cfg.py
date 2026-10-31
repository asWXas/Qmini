"""AS2W PPO with one shared critic network and three value outputs."""

from isaaclab.utils import configclass

from Qmini.algo.multi_critic.cfg import RslRlMultiCriticPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import AS2WFlatPPORunnerCfg, AS2WRoughPPORunnerCfg


def multi_critic_algorithm(base):
    params = base.to_dict()
    params.pop("class_name")
    return RslRlMultiCriticPpoAlgorithmCfg(
        **params,
        critic_groups={
            "task": {"weight": 1.0},
            "style": {"weight": 1.0},
            "safety": {"weight": 1.0},
        },
        normalize_critic_weights=False,
    )


@configclass
class AS2WRoughMultiCriticRunnerCfg(AS2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__() # type: ignore
        self.experiment_name = "as2w_rough_multi_critic"
        self.algorithm = multi_critic_algorithm(self.algorithm)


@configclass
class AS2WFlatMultiCriticRunnerCfg(AS2WFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "as2w_flat_multi_critic"
        self.algorithm = multi_critic_algorithm(self.algorithm)
