"""Distill a matching AS2W PPO checkpoint into a smaller student.

Both networks consume the environment's ``policy`` observation group. This
is policy compression, not privileged-observation distillation. Keep the
teacher's observation ordering, normalization, action ordering and scaling
identical to the PPO training environment. Load a trained PPO checkpoint
before starting distillation; these configurations do not provide weights.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from .rsl_rl_ppo_cfg import AS2WFlatPPORunnerCfg, AS2WRoughPPORunnerCfg


@configclass
class AS2WRoughDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 1000
    save_interval = 50
    experiment_name = "as2w_rough_distillation"
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    # Copy the PPO actor so architecture and checkpoint parameter shapes agree.
    teacher = AS2WRoughPPORunnerCfg().actor.copy()
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class AS2WFlatDistillationRunnerCfg(AS2WRoughDistillationRunnerCfg):
    experiment_name = "as2w_flat_distillation"
    teacher = AS2WFlatPPORunnerCfg().actor.copy()
