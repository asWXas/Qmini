"""Single-optimizer, joint PPO and CENet training for DreamWaQ."""

from __future__ import annotations

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .model import DreamWaQActor, to_frames


class DreamWaQPPO(PPO):
    def __init__(
        self,
        *args,
        velocity_loss_coef: float = 1.0,
        reconstruction_loss_coef: float = 1.0,
        kl_loss_coef: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, DreamWaQActor):
            raise TypeError("DreamWaQPPO requires DreamWaQActor")
        if self.rnd is not None:
            raise ValueError("DreamWaQPPO currently requires rnd_cfg=None")
        if self.symmetry is not None and (not self.symmetry.use_data_augmentation or self.symmetry.use_mirror_loss):
            raise ValueError("DreamWaQPPO supports symmetry data augmentation, without mirror loss")
        if self.use_mixed_precision:
            raise ValueError("DreamWaQPPO currently requires use_mixed_precision=False")
        if self.actor.obs_normalization:
            raise ValueError("DreamWaQ next-frame targets require obs_normalization=False")
        self.velocity_loss_coef = velocity_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.kl_loss_coef = kl_loss_coef
        shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
        device = self.storage.device
        self.storage.observations.set("dreamwaq_next", torch.zeros(*shape, self.actor.frame_dim, device=device))
        self.storage.observations.set("dreamwaq_velocity", torch.zeros(*shape, 3, device=device))
        self.storage.observations.set("dreamwaq_valid", torch.zeros(*shape, 1, dtype=torch.bool, device=device))

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> DreamWaQPPO:
        manager = env.unwrapped.observation_manager
        names = manager.active_terms["policy"]
        dims = manager.group_obs_term_dim["policy"]
        policy_cfg = env.unwrapped.cfg.observations.policy
        history_length = policy_cfg.history_length
        if history_length is None or history_length < 2:
            raise ValueError("DreamWaQ policy observations need history_length >= 2")
        if policy_cfg.history_order != "term" or not policy_cfg.flatten_history_dim:
            raise ValueError("DreamWaQ requires flattened term-major policy history")
        term_widths = []
        for name, shape in zip(names, dims, strict=True):
            total = 1
            for dimension in shape:
                total *= dimension
            if total % history_length:
                raise ValueError(f"Observation {name} is not a complete history stack")
            term_widths.append(int(total // history_length))
        if manager.active_terms["critic"][0] != "base_lin_vel":
            raise ValueError("The first critic term must be unscaled base_lin_vel")
        cfg["actor"]["history_length"] = history_length
        cfg["actor"]["term_widths"] = term_widths
        return PPO.construct_algorithm(obs, env, cfg, device)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        # Capture the pre-step privileged velocity before PPO clears the transition.
        velocity = self.transition.observations["critic"][:, :3].clone()
        step = self.storage.step
        super().process_env_step(obs, rewards, dones, extras)
        next_frame = to_frames(obs["policy"], self.actor.term_widths, self.actor.history_length)[:, -1]
        self.storage.observations["dreamwaq_next"][step].copy_(next_frame)
        self.storage.observations["dreamwaq_velocity"][step].copy_(velocity)
        # dones[t] marks (obs[t], action[t], returned_obs[t+1]). On termination
        # or timeout the returned observation is already reset, so exclude this
        # pre-reset sample from ALL auxiliary losses. Do not shift the mask to
        # t-1 or discard the first valid transition of the new episode.
        self.storage.observations["dreamwaq_valid"][step].copy_((~dones.bool()).reshape(-1, 1))

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("DreamWaQPPO supports feedforward actor and critic only")
        totals = dict(value=0.0, surrogate=0.0, entropy=0.0, velocity=0.0, reconstruction=0.0, kl=0.0)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            original_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry is not None:
                self.symmetry.augment_batch(batch, original_size)

            self.actor(batch.observations, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy[:original_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,
                        tuple(param[:original_size] for param in self.actor.output_distribution_params),
                    ).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl > self.desired_kl * 2:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0 < kl < self.desired_kl / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(rate, src=0)
                        self.learning_rate = rate.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantage = batch.advantages.squeeze(-1)
            surrogate_loss = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).square(), (clipped - batch.returns).square()).mean()
            else:
                value_loss = (values - batch.returns).square().mean()

            velocity_loss, reconstruction_loss, kl_loss = self.actor.auxiliary_loss(
                batch.observations,
                batch.observations["dreamwaq_next"],
                batch.observations["dreamwaq_velocity"],
                batch.observations["dreamwaq_valid"],
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.velocity_loss_coef * velocity_loss
                + self.reconstruction_loss_coef * reconstruction_loss
                + self.kl_loss_coef * kl_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            for name, term in (
                ("value", value_loss),
                ("surrogate", surrogate_loss),
                ("entropy", entropy.mean()),
                ("velocity", velocity_loss),
                ("reconstruction", reconstruction_loss),
                ("kl", kl_loss),
            ):
                totals[name] += term.item()

        # RSL-RL 5.x updates normalization once per rollout, after optimization.
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.storage.clear()
        count = self.num_learning_epochs * self.num_mini_batches
        return {name: value / count for name, value in totals.items()}
