"""COO voxel features: (batch,x,y,z,time) coordinates and three centroid offsets."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def sparse_voxelize(points, size, resolution):
    coordinates = (points + size / 2) * (resolution / size)
    valid = torch.isfinite(coordinates).all(1) & (coordinates >= 0).all(1) & (coordinates < resolution).all(1)
    coordinates = coordinates[valid]
    cells, inverse, counts = torch.unique(coordinates.floor().int(), dim=0, return_inverse=True, return_counts=True)
    features = torch.zeros((len(cells), 3), dtype=points.dtype, device=points.device)
    features.index_add_(0, inverse, coordinates - coordinates.floor())
    features /= counts[:, None].clamp_min(1)
    return cells, features


def temporal_input(observed_local, previous_local, size, resolution, observation_noise=None):
    """Legacy v1: t=0 current measurement, t=1 aligned prior map (possibly warm-started)."""
    all_coordinates, all_features = [], []
    for time, points in enumerate((observed_local, previous_local)):
        cells, features = sparse_voxelize(points, size, resolution)
        if time == 0 and observation_noise:
            dropout = observation_noise.get("voxel_dropout", 0)
            jitter = observation_noise.get("voxel_noise_std", 0)
            if dropout:
                keep = torch.rand(len(cells), device=cells.device) >= dropout
                cells, features = cells[keep], features[keep]
            if jitter:
                features = (features + torch.randn_like(features) * jitter).clamp(0, 1 - 1e-6)
        batch = torch.zeros((len(cells), 1), dtype=torch.int32, device=cells.device)
        times = torch.full_like(batch, time)
        all_coordinates.append(torch.cat((batch, cells, times), dim=1))
        all_features.append(features)
    return torch.cat(all_coordinates), torch.cat(all_features)


def coordinate_keys(cells, resolution):
    return (cells[:, 0].long() * resolution + cells[:, 1]) * resolution + cells[:, 2]


def match_target(coordinates, target_cells, resolution, stride=1):
    """Map candidate XYZ cells at any spatial stride to target surface voxels."""
    coarse_target = torch.div(target_cells, stride, rounding_mode="floor") * stride
    target_keys = coordinate_keys(coarse_target, resolution)
    keys, order = torch.sort(target_keys)
    candidate_keys = coordinate_keys(coordinates[:, 1:4].to(target_cells.device), resolution)
    if not len(keys):
        return torch.zeros(len(coordinates), dtype=torch.bool, device=target_cells.device), torch.zeros_like(
            candidate_keys
        )
    index = torch.searchsorted(keys, candidate_keys).clamp_max(len(keys) - 1)
    return keys[index] == candidate_keys, order[index]


@dataclass
class SparsePrediction:
    # Every decoder stage is supervised BEFORE probability pruning.
    stages: list
    coordinates: torch.Tensor
    logits: torch.Tensor
    offsets: torch.Tensor


def sparse_reconstruction_loss(
    prediction, target_cells, target_offsets, resolution, offset_weight=1.0, occupancy_loss="bce"
):
    if occupancy_loss not in ("bce", "balanced"):
        raise ValueError(f"Unknown occupancy_loss: {occupancy_loss}")
    losses = []
    for coordinates, logits, stride in prediction.stages:
        occupied, _ = match_target(coordinates, target_cells, resolution, stride)
        elementwise = F.binary_cross_entropy_with_logits(logits.flatten(), occupied.to(logits.dtype), reduction="none")
        if occupancy_loss == "balanced" and bool(occupied.any()) and bool((~occupied).any()):
            losses.append((elementwise[occupied].mean() + elementwise[~occupied].mean()) / 2)
        else:
            losses.append(elementwise.mean())
    # Also keep an empty-output loss attached to the graph (e.g. an empty input).
    zero = prediction.offsets.sum() * 0
    occupancy_loss = torch.stack(losses).mean() if losses else zero
    occupied, target_ids = match_target(prediction.coordinates, target_cells, resolution)
    if occupied.any():
        distance = torch.linalg.vector_norm(prediction.offsets[occupied] - target_offsets[target_ids[occupied]], dim=1)
        offset_loss = distance.mean()
    else:
        offset_loss = zero
    return occupancy_loss + offset_weight * offset_loss


@torch.no_grad()
def decode_sparse(prediction, size, resolution, threshold, max_points=0):
    scores = prediction.logits.detach().flatten().sigmoid()
    ids = torch.where(scores >= threshold)[0]
    if max_points > 0 and len(ids) > max_points:
        ids = ids[torch.topk(scores[ids], max_points).indices]
    cells = prediction.coordinates[ids, 1:4].to(prediction.offsets.device)
    return (cells + prediction.offsets.detach()[ids]) * (size / resolution) - size / 2
