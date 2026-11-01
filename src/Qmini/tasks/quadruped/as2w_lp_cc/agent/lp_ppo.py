"""PPO adapters that put the curriculum into the same checkpoint as the policy."""

import warnings

from rsl_rl.algorithms import PPO
from Qmini.algo.multi_critic.algo.multi_critic_ppo import MultiCriticPPO


class _CurriculumCheckpoint:
    def save(self):
        state = super().save()
        state["lp_acrl"] = self.lp_env.lp_curriculum.sampler.state_dict()
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        resume = super().load(loaded_dict, load_cfg, strict)
        if "lp_acrl" in loaded_dict:
            self.lp_env.lp_curriculum.sampler.load_state_dict(loaded_dict["lp_acrl"])
            # Simulator state is not checkpointed; start fresh episodes using the
            # restored distribution. No unfinished trajectory enters LP estimates.
            self.lp_env.reset()
        else:
            warnings.warn("Checkpoint has no LP-ACRL state; curriculum starts uniformly.", stacklevel=2)
        return resume


class LPACRLPPO(_CurriculumCheckpoint, PPO):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg.lp_env = env.unwrapped
        return alg


class LPACRLMultiCriticPPO(_CurriculumCheckpoint, MultiCriticPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The repository's multi-critic updater expects this attribute even
        # when RND is disabled; current RSL-RL no longer creates it in that case.
        if self.rnd is None:
            self.rnd_optimizer = None

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = MultiCriticPPO.construct_algorithm(obs, env, cfg, device)
        alg.lp_env = env.unwrapped
        return alg
