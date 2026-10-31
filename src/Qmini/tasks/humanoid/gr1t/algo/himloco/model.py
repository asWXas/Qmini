"""History based hybrid internal model and deployable policy."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F


def to_frames(flat: torch.Tensor, term_widths: list[int], history_length: int) -> torch.Tensor:
    """Unpack Isaac Lab's term-major flattened history into [batch, time, frame]."""
    parts = []
    offset = 0
    for width in term_widths:
        size = width * history_length
        parts.append(flat[:, offset : offset + size].reshape(-1, history_length, width))
        offset += size
    if offset != flat.shape[-1]:
        raise ValueError(f"Policy observation width {flat.shape[-1]} != expected {offset}")
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def sinkhorn(logits: torch.Tensor, epsilon: float = 0.05, iterations: int = 3) -> torch.Tensor:
    """Balanced prototype assignments over the local minibatch."""
    scores = logits - logits.max()
    q = torch.exp(scores / epsilon).T
    q /= q.sum().clamp_min(1e-12)
    prototypes, samples = q.shape
    for _ in range(iterations):
        q /= q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= samples
    return (q * samples).T


class HIMActor(MLPModel):
    """Actor whose internal model is frozen during PPO updates."""

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
        encoder_hidden_dims: list[int] = (512, 256, 128),
        target_hidden_dims: list[int] = (128, 64),
        num_prototypes: int = 16,
        temperature: float = 3.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
    ) -> None:
        if not term_widths or history_length < 2 or latent_dim <= 0 or num_prototypes < 2:
            raise ValueError("HIMLoco needs history, a latent dimension, and at least two prototypes")
        if not encoder_hidden_dims or not target_hidden_dims or temperature <= 0 or sinkhorn_epsilon <= 0:
            raise ValueError("Invalid HIMLoco estimator configuration")
        self.history_length = int(history_length)
        self.term_widths = [int(width) for width in term_widths]
        self.frame_dim = int(sum(self.term_widths))
        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        if self.frame_dim * self.history_length != obs["policy"].shape[-1]:
            raise ValueError("Policy history layout does not match the observation tensor")
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        if self.obs_groups != ["policy"]:
            raise ValueError("HIM actor requires only the policy observation group")
        self.source_encoder = MLP(self.obs_dim, 3 + latent_dim, encoder_hidden_dims, activation)
        self.target_encoder = MLP(self.frame_dim, latent_dim, target_hidden_dims, activation)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))

    def _get_latent_dim(self) -> int:
        return self.frame_dim + 3 + self.latent_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        flat = self.obs_normalizer(obs["policy"])
        current = to_frames(flat, self.term_widths, self.history_length)[:, -1]
        # HIO and PPO are separate phases. The policy never changes the HIM encoder.
        with torch.no_grad():
            embedding = self.source_encoder(flat)
            velocity = embedding[:, :3]
            context = F.normalize(embedding[:, 3:], dim=-1)
        return torch.cat((current, velocity, context), dim=-1)

    def estimator_parameters(self):
        return list(self.source_encoder.parameters()) + list(self.target_encoder.parameters()) + [self.prototypes]

    def policy_parameters(self):
        estimator_ids = {id(parameter) for parameter in self.estimator_parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in estimator_ids)

    def hio_loss(
        self, obs: TensorDict, next_frame: torch.Tensor, velocity_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self.obs_normalizer(obs["policy"])
        source = self.source_encoder(flat)
        velocity_loss = F.mse_loss(source[:, :3], velocity_target)
        if flat.shape[0] < 2:
            return velocity_loss, velocity_loss.new_zeros(())

        source_context = F.normalize(source[:, 3:], dim=-1)
        target_context = F.normalize(self.target_encoder(next_frame), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        source_scores = source_context @ prototypes.T
        target_scores = target_context @ prototypes.T
        with torch.no_grad():
            source_assignment = sinkhorn(source_scores.detach(), self.sinkhorn_epsilon, self.sinkhorn_iterations)
            target_assignment = sinkhorn(target_scores.detach(), self.sinkhorn_epsilon, self.sinkhorn_iterations)
        source_log_prob = F.log_softmax(source_scores / self.temperature, dim=-1)
        target_log_prob = F.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = (
            -0.5
            * (
                (target_assignment * source_log_prob).sum(dim=-1) + (source_assignment * target_log_prob).sum(dim=-1)
            ).mean()
        )
        return velocity_loss, swap_loss

    def as_jit(self) -> nn.Module:
        return _ExportActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportActor(self)


class _ExportActor(nn.Module):
    is_recurrent = False

    def __init__(self, model: HIMActor) -> None:
        super().__init__()
        self.input_size = model.obs_dim
        self.history_length = model.history_length
        self.term_widths = model.term_widths
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.source_encoder = copy.deepcopy(model.source_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.output_layer = model.distribution.as_deterministic_output_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        current = to_frames(x, self.term_widths, self.history_length)[:, -1]
        embedding = self.source_encoder(x)
        velocity = embedding[:, :3]
        context = F.normalize(embedding[:, 3:], dim=-1)
        return self.output_layer(self.mlp(torch.cat((current, velocity, context), dim=-1)))

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
