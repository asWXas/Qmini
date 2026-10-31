"""Context-aided estimator and deployable DreamWaQ actor."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F


def to_frames(flat: torch.Tensor, term_widths: list[int], history_length: int) -> torch.Tensor:
    """Isaac Lab stacks each term separately, oldest to newest."""
    parts = []
    offset = 0
    for width in term_widths:
        size = width * history_length
        parts.append(flat[..., offset : offset + size].reshape(-1, history_length, width))
        offset += size
    if offset != flat.shape[-1]:
        raise ValueError(f"Policy observation width {flat.shape[-1]} != expected {offset}")
    return torch.cat(parts, dim=-1)


class DreamWaQActor(MLPModel):
    """Policy with shared velocity/VAE encoder; policy always uses deterministic means."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_length: int = 5,
        term_widths: list[int] | None = None,
        latent_dim: int = 16,
        encoder_hidden_dims: list[int] = (128, 64),
        decoder_hidden_dims: list[int] = (64, 128),
    ) -> None:
        if not term_widths or history_length < 2:
            raise ValueError("DreamWaQ needs per-term widths and at least two history frames")
        self.history_length = history_length
        self.term_widths = [int(width) for width in term_widths]
        self.frame_dim = int(sum(self.term_widths))
        self.context_dim = latent_dim
        if sum(term_widths) * history_length != obs["policy"].shape[-1]:
            raise ValueError("Policy history layout does not match the observation tensor")
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        if self.obs_groups != ["policy"]:
            raise ValueError("DreamWaQ actor requires only the policy observation group")
        self.encoder = MLP(self.obs_dim, encoder_hidden_dims[-1], encoder_hidden_dims[:-1], activation)
        self.velocity_head = nn.Linear(encoder_hidden_dims[-1], 3)
        self.context_head = nn.Linear(encoder_hidden_dims[-1], 2 * latent_dim)
        self.decoder = MLP(self.frame_dim + 3 + latent_dim, self.frame_dim, decoder_hidden_dims, activation)

    def _get_latent_dim(self) -> int:
        return self.frame_dim + 3 + self.context_dim

    def _encode(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.obs_normalizer(obs["policy"])
        frames = to_frames(flat, self.term_widths, self.history_length)
        hidden = self.encoder(flat)
        velocity = self.velocity_head(hidden)
        mu, logvar = self.context_head(hidden).chunk(2, dim=-1)
        logvar = logvar.clamp(-10.0, 10.0)
        return frames[:, -1], velocity, mu, logvar

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        current, velocity, mu, _ = self._encode(obs)
        return torch.cat((current, velocity, mu), dim=-1)

    def auxiliary_loss(
        self, obs: TensorDict, next_frame: torch.Tensor, velocity_target: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Filter before encoding: multiplying a terminal loss by zero would
        # still allow NaNs in invalid observations/targets to enter gradients.
        valid = valid.reshape(-1).bool()
        if not valid.any():
            # Keep zero gradients for every parameter even when a distributed
            # worker has no valid samples, so gradient reduction stays aligned.
            zero = sum(parameter.sum() * 0.0 for parameter in self.parameters())
            return zero, zero, zero
        current, velocity, mu, logvar = self._encode(obs[valid])
        velocity_loss = F.mse_loss(velocity, velocity_target[valid])
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        predicted = self.decoder(torch.cat((current, velocity, z), dim=-1))
        reconstruction_loss = F.mse_loss(predicted, next_frame[valid])
        kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
        return velocity_loss, reconstruction_loss, kl_loss

    def as_jit(self) -> nn.Module:
        return _ExportActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportActor(self)


class _ExportActor(nn.Module):
    is_recurrent = False

    def __init__(self, model: DreamWaQActor) -> None:
        super().__init__()
        self.term_widths = model.term_widths
        self.history_length = model.history_length
        self.input_size = model.obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.velocity_head = copy.deepcopy(model.velocity_head)
        self.context_head = copy.deepcopy(model.context_head)
        self.mlp = copy.deepcopy(model.mlp)
        self.output_layer = model.distribution.as_deterministic_output_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        current = to_frames(x, self.term_widths, self.history_length)[:, -1]
        hidden = self.encoder(x)
        velocity = self.velocity_head(hidden)
        mu = self.context_head(hidden).chunk(2, dim=-1)[0]
        return self.output_layer(self.mlp(torch.cat((current, velocity, mu), dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["observations"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
