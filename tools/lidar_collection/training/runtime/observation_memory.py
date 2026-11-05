"""Bounded, measurement-only surface memory, independent of network pruning."""

import math
from dataclasses import dataclass

import torch

from ..geometry.sparse_voxel import coordinate_keys, sparse_voxelize
from ..models.architectures import is_recurrent


def validate_memory_config(config):
    if not config.get("observation_memory", False):
        return
    if not is_recurrent(config):
        raise ValueError("Observation memory is supported by sparse4d_v3 only")
    size = config.get("memory_size", config["map_size"] * 4)
    if not math.isfinite(size) or size < config["map_size"]:
        raise ValueError("memory_size must be finite and at least map_size")
    if config.get("memory_max_voxels", 200000) < 1:
        raise ValueError("memory_max_voxels must be positive")
    if config.get("max_points", 0):
        raise ValueError("Observation memory requires max_points=0 to preserve observed surface cells")


@dataclass
class ObservedSurfaceMemory:
    # Fixed anchor grid prevents sub-voxel motion from being repeatedly rounded away.
    anchor_from_current: torch.Tensor | None = None
    points: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    last_seen: torch.Tensor | None = None
    frame: int = 0

    @torch.no_grad()
    def update(self, measured_local, current_from_previous, config):
        validate_memory_config(config)
        device = measured_local.device
        dtype = torch.float64
        if self.anchor_from_current is None:
            self.anchor_from_current = torch.eye(4, device=device, dtype=dtype)
        else:
            t = current_from_previous.to(dtype)
            inverse = torch.eye(4, device=device, dtype=dtype)
            inverse[:3, :3] = t[:3, :3].T
            inverse[:3, 3] = -t[:3, :3].T @ t[:3, 3]
            self.anchor_from_current = self.anchor_from_current @ inverse
        self.frame += 1
        pose = self.anchor_from_current
        incoming = measured_local.detach().to(dtype)
        incoming = incoming[torch.isfinite(incoming).all(1)]
        incoming = incoming @ pose[:3, :3].T + pose[:3, 3]
        weights = torch.ones(len(incoming), dtype=dtype, device=device)
        seen = torch.full((len(incoming),), self.frame, device=device, dtype=torch.int64)
        if self.points is not None:
            incoming = torch.cat((self.points, incoming))
            weights = torch.cat((self.weights, weights))
            seen = torch.cat((self.last_seen, seen))
        local = (incoming - pose[:3, 3]) @ pose[:3, :3]
        half = config.get("memory_size", config["map_size"] * 4) / 2
        valid = ((local >= -half) & (local < half)).all(1)
        incoming, weights, seen = incoming[valid], weights[valid], seen[valid]
        spacing = config["map_size"] / config["resolution"] / 2
        cells = torch.floor(incoming / spacing).long()
        unique, inverse = torch.unique(cells, dim=0, return_inverse=True)
        total = weights.new_zeros(len(unique)).index_add_(0, inverse, weights)
        sums = incoming.new_zeros((len(unique), 3)).index_add_(0, inverse, incoming * weights[:, None])
        timestamps = seen.new_zeros(len(unique)).scatter_reduce_(0, inverse, seen, reduce="amax", include_self=True)
        points = sums / total[:, None].clamp_min(1)
        limit = config.get("memory_max_voxels", 200000)
        selected = timestamps.argsort(descending=True, stable=True)[:limit]
        self.points = points[selected]
        self.weights = total[selected].clamp_max(64)
        self.last_seen = timestamps[selected]
        return self.local_points(config, measured_local.dtype)

    @torch.no_grad()
    def local_points(self, config, dtype=torch.float32):
        pose = self.anchor_from_current
        local = ((self.points - pose[:3, 3]) @ pose[:3, :3]).to(dtype)
        cells, offsets = sparse_voxelize(local, config["map_size"], config["resolution"])
        return (cells + offsets) * (config["map_size"] / config["resolution"]) - config["map_size"] / 2


@torch.no_grad()
def preserve_observed_surface(predicted, measured_memory, config):
    """Measured centroids win in shared cells; network-only surface remains available."""
    size, resolution = config["map_size"], config["resolution"]
    p_cells, p_offsets = sparse_voxelize(predicted, size, resolution)
    m_cells, m_offsets = sparse_voxelize(measured_memory, size, resolution)
    occupied = coordinate_keys(m_cells, resolution)
    keep = ~torch.isin(coordinate_keys(p_cells, resolution), occupied)
    cells = torch.cat((m_cells, p_cells[keep]))
    offsets = torch.cat((m_offsets, p_offsets[keep]))
    return (cells + offsets) * (size / resolution) - size / 2
