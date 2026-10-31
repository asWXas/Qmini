"""HIMLoco: HIO first, frozen-estimator PPO second."""

from __future__ import annotations

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_optimizer
from tensordict import TensorDict

from .model import HIMActor, to_frames


class HIMLocoPPO(PPO):
    def __init__(
        self,
        *args,
        estimator_learning_rate: float = 1e-3,
        estimator_max_grad_norm: float = 10.0,
        velocity_loss_coef: float = 1.0,
        swap_loss_coef: float = 1.0,
        hio_epochs: int = 1,
        **kwargs,
    ) -> None:
        optimizer_name = kwargs.get("optimizer", "adam")
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, HIMActor):
            raise TypeError("HIMLocoPPO requires HIMActor")
        if self.rnd is not None:
            raise ValueError("HIMLocoPPO currently requires rnd_cfg=None")
        if self.symmetry is not None and (not self.symmetry.use_data_augmentation or self.symmetry.use_mirror_loss):
            raise ValueError("HIMLocoPPO supports symmetry data augmentation, without mirror loss")
        if self.actor.obs_normalization:
            raise ValueError("HIMLoco successor targets require obs_normalization=False")
        if hio_epochs <= 0:
            raise ValueError("hio_epochs must be positive")

        # The parent PPO initially includes every actor parameter. Replace that
        # optimizer so PPO cannot change the estimator or its prototypes.
        self.optimizer = resolve_optimizer(optimizer_name)(
            chain(self.actor.policy_parameters(), self.critic.parameters()), lr=self.learning_rate
        )
        self.estimator_optimizer = torch.optim.Adam(self.actor.estimator_parameters(), lr=estimator_learning_rate)
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.velocity_loss_coef = velocity_loss_coef
        self.swap_loss_coef = swap_loss_coef
        self.hio_epochs = hio_epochs

        shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
        device = self.storage.device
        self.storage.observations.set("him_next", torch.zeros(*shape, self.actor.frame_dim, device=device))
        self.storage.observations.set("him_velocity", torch.zeros(*shape, 3, device=device))
        self.storage.observations.set("him_valid", torch.zeros(*shape, 1, dtype=torch.bool, device=device))

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> HIMLocoPPO:
        manager = env.unwrapped.observation_manager
        policy_cfg = env.unwrapped.cfg.observations.policy
        history_length = policy_cfg.history_length
        if history_length is None or history_length < 2:
            raise ValueError("HIMLoco policy observations need at least two history frames")
        if policy_cfg.history_order != "term" or not policy_cfg.flatten_history_dim:
            raise ValueError("HIMLoco requires flattened term-major policy history")
        term_widths = []
        for name, shape in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"], strict=True):
            total = 1
            for dimension in shape:
                total *= dimension
            if total % history_length:
                raise ValueError(f"Policy term {name} has an incomplete history")
            term_widths.append(int(total // history_length))
        if manager.active_terms["critic"][0] != "base_lin_vel":
            raise ValueError("The first critic term must be unscaled base_lin_vel")
        cfg["actor"]["history_length"] = history_length
        cfg["actor"]["term_widths"] = term_widths
        return PPO.construct_algorithm(obs, env, cfg, device)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        velocity = self.transition.observations["critic"][:, :3].clone()
        step = self.storage.step
        super().process_env_step(obs, rewards, dones, extras)
        next_frame = to_frames(obs["policy"], self.actor.term_widths, self.actor.history_length)[:, -1]
        self.storage.observations["him_next"][step].copy_(next_frame)
        self.storage.observations["him_velocity"][step].copy_(velocity)
        # dones[t] excludes the pre-reset obs[t] -> returned reset_obs[t+1]
        # pair from both HIO losses, including timeouts. Do not shift to t-1;
        # the first nonterminal transition after reset is valid. PPO keeps
        # terminal transitions for learning termination rewards.
        self.storage.observations["him_valid"][step].copy_((~dones.bool()).reshape(-1, 1))

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("HIMLocoPPO supports feedforward actor and critic only")
        velocity_total = 0.0
        swap_total = 0.0
        hio_updates = 0

        # Complete HIO before entering PPO. The actor's get_latent uses no_grad,
        # and the PPO optimizer excludes every HIM parameter.
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.hio_epochs)
        for batch in generator:
            observations = batch.observations
            if self.symmetry is not None:
                observations, _ = self.symmetry.data_augmentation_func(
                    env=self.symmetry.env, obs=observations, actions=None
                )
            valid = observations["him_valid"].reshape(-1).bool()
            has_local_samples = bool(valid.any())
            if self.is_multi_gpu:
                valid_count = valid.sum()
                torch.distributed.all_reduce(valid_count, op=torch.distributed.ReduceOp.SUM)
                # Even a zero-gradient Adam step changes parameters when old
                # momentum exists. Skip HIO on all ranks if none has valid data.
                if valid_count.item() == 0:
                    continue
            elif not has_local_samples:
                continue
            selected = observations[valid]
            if has_local_samples:
                velocity_loss, swap_loss = self.actor.hio_loss(selected, selected["him_next"], selected["him_velocity"])
                loss = self.velocity_loss_coef * velocity_loss + self.swap_loss_coef * swap_loss
                velocity_total += velocity_loss.item()
                swap_total += swap_loss.item()
                hio_updates += 1
            else:
                loss = sum(parameter.sum() * 0.0 for parameter in self.actor.estimator_parameters())

            self.estimator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                # All ranks must participate even if one rank has no valid successor.
                for parameter in self.actor.estimator_parameters():
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                    parameter.grad /= self.gpu_world_size
            torch.nn.utils.clip_grad_norm_(self.actor.estimator_parameters(), self.estimator_max_grad_norm)
            self.estimator_optimizer.step()

        # Prevent PPO's gradient reduction from seeing stale estimator gradients.
        self.estimator_optimizer.zero_grad(set_to_none=True)
        losses = super().update()
        losses["him_velocity"] = velocity_total / max(hio_updates, 1)
        losses["him_swap"] = swap_total / max(hio_updates, 1)
        return losses

    def save(self) -> dict:
        state = super().save()
        state["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if (load_cfg is None or load_cfg.get("optimizer", False)) and "estimator_optimizer_state_dict" in loaded_dict:
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        return restore_iteration
