"""HIMLoco training configuration for GR1T."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from Qmini.tasks.humanoid.gr1t.mdp import symmetry

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class HIMActorCfg(RslRlMLPModelCfg):
    class_name: str = "Qmini.tasks.humanoid.gr1t.algo.himloco.model:HIMActor"
    latent_dim: int = 16
    encoder_hidden_dims: list[int] = [512, 256, 128]
    target_hidden_dims: list[int] = [128, 64]
    num_prototypes: int = 16
    temperature: float = 3.0
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 3


@configclass
class HIMLocoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "Qmini.tasks.humanoid.gr1t.algo.himloco.ppo:HIMLocoPPO"
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 10.0
    velocity_loss_coef: float = 1.0
    swap_loss_coef: float = 1.0
    hio_epochs: int = 1


@configclass
class HIMLocoRunnerCfg(PPORunnerCfg):
    experiment_name = "flat_himloco"
    actor = HIMActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = HIMLocoAlgorithmCfg(
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
