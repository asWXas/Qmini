"""Shared plane initialization, world-axis geometry and observation voxelization."""

import math

import torch

from .sparse_voxel import sparse_voxelize

# AS2W nominal body-to-ground height; an explicit prior, never measured from GT.
DEFAULT_PLANE_Z = -0.38


def canvas_voxels(config, reference):
    """Active query support, never a sensor observation or a ground-truth label."""
    mode = config.get("canvas_mode", "plane")
    if mode == "plane":
        return plane_voxels(config, reference)
    if mode != "volume":
        raise ValueError(f"Unknown canvas_mode: {mode}")
    axis = torch.arange(config["resolution"], device=reference.device, dtype=torch.int32)
    cells = torch.cartesian_prod(axis, axis, axis)
    return cells, reference.new_full((len(cells), 3), 0.5)


def plane_voxels(config, reference):
    """One horizontal occupied layer across the local XY map, with exact surface Z.

    plane_z is world-up height relative to the first frame's base position.
    This is a surface prior, not a solid slab, and does not rotate with body tilt.
    """
    size, resolution = config["map_size"], config["resolution"]
    height = config.get("plane_z", DEFAULT_PLANE_Z)
    if not math.isfinite(height) or not -size / 2 <= height < size / 2:
        raise ValueError("plane_z must be finite and inside [-map_size/2, map_size/2)")
    z = (height + size / 2) / (size / resolution)
    layer = min(math.floor(z), resolution - 1)
    axis = torch.arange(resolution, dtype=torch.int32, device=reference.device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    cells = torch.stack((x.reshape(-1), y.reshape(-1), torch.full_like(x.reshape(-1), layer)), dim=1)
    offsets = reference.new_full((len(cells), 3), 0.5)
    offsets[:, 2] = min(z - layer, 1 - 1e-6)
    return cells, offsets


def rotation_matrix(quaternion):
    q = quaternion / quaternion.norm().clamp_min(1e-8)
    x, y, z, w = q.unbind()
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)


def observed_voxels(points_local, size, resolution, noise=None):
    cells, features = sparse_voxelize(points_local, size, resolution)
    if noise:
        dropout, jitter = noise.get("voxel_dropout", 0), noise.get("voxel_noise_std", 0)
        if dropout:
            keep = torch.rand(len(cells), device=cells.device) >= dropout
            cells, features = cells[keep], features[keep]
        if jitter:
            features = (features + torch.randn_like(features) * jitter).clamp(0, 1 - 1e-6)
    return cells, features
