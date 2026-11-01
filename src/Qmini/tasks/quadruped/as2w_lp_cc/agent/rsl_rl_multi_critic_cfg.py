"""AS2W PPO with one shared critic network and three value outputs."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import AS2WFlatPPORunnerCfg, AS2WRoughPPORunnerCfg


@configclass
class RslRlMultiCriticPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # Keep configuration import free of runtime environment imports. The shared
    # multi_critic package eagerly imports USD; Kit must select its USD first.
    class_name: str = "Qmini.tasks.quadruped.as2w_lp_cc.agent.lp_ppo:LPACRLMultiCriticPPO"
    critic_groups: dict = {}
    normalize_critic_weights: bool = False
    rnd_group: str | None = None


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
        self.experiment_name = "as2w_lp_acrl_rough_multi_critic"
        self.algorithm = multi_critic_algorithm(self.algorithm)
        self.algorithm.class_name = "Qmini.tasks.quadruped.as2w_lp_cc.agent.lp_ppo:LPACRLMultiCriticPPO"


@configclass
class AS2WFlatMultiCriticRunnerCfg(AS2WFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "as2w_lp_acrl_flat_multi_critic"
        self.algorithm = multi_critic_algorithm(self.algorithm)
        self.algorithm.class_name = "Qmini.tasks.quadruped.as2w_lp_cc.agent.lp_ppo:LPACRLMultiCriticPPO"
