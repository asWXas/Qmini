"""DreamWaQ training configuration for GR1T."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from Qmini.tasks.humanoid.gr1t.mdp import symmetry

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class DreamWaQActorCfg(RslRlMLPModelCfg):
    class_name: str = "Qmini.tasks.humanoid.gr1t.algo.dream_waq.model:DreamWaQActor"
    latent_dim: int = 16
    encoder_hidden_dims: list[int] = [128, 64]
    decoder_hidden_dims: list[int] = [64, 128]


@configclass
class DreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "Qmini.tasks.humanoid.gr1t.algo.dream_waq.ppo:DreamWaQPPO"
    velocity_loss_coef: float = 1.0
    reconstruction_loss_coef: float = 1.0
    kl_loss_coef: float = 0.01


@configclass
class DreamWaQRunnerCfg(PPORunnerCfg):
    experiment_name = "flat_dream_waq"
    actor = DreamWaQActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetry.compute_symmetric_states,
        ),
    )
