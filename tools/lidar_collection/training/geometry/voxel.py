"""World-aligned, base-centered surface voxels and detached prediction feedback."""

import torch


def voxelize(points, size, resolution):
    """Return [occupancy, centroid offsets xyz] in CXYZ order.

    Points are measured in meters relative to the current base position. An empty
    cell means no sampled surface, not a ray-certified free-space measurement.
    """
    grid = torch.zeros((4, resolution**3), dtype=points.dtype, device=points.device)
    coordinates = (points + size / 2) * (resolution / size)
    valid = torch.isfinite(coordinates).all(1) & (coordinates >= 0).all(1) & (coordinates < resolution).all(1)
    coordinates = coordinates[valid]
    cells = coordinates.floor().long()
    ids = (cells[:, 0] * resolution + cells[:, 1]) * resolution + cells[:, 2]
    counts = torch.bincount(ids, minlength=resolution**3)
    grid[0] = counts > 0
    grid[1:].scatter_add_(1, ids[None].expand(3, -1), (coordinates - cells).T)
    grid[1:] /= counts.clamp_min(1)[None]
    return grid.reshape(4, resolution, resolution, resolution)


@torch.no_grad()
def decode(logits, offsets, size, threshold=0.5, max_points=20000):
    """Decode one prediction to local points; keep highest-confidence cells if capped."""
    resolution = logits.shape[-1]
    scores = logits.detach().sigmoid().reshape(-1)
    ids = torch.where(scores >= threshold)[0]
    if max_points > 0 and len(ids) > max_points:
        ids = ids[torch.topk(scores[ids], max_points).indices]
    cells = torch.stack((ids // resolution**2, ids // resolution % resolution, ids % resolution), dim=1)
    positions = cells + offsets.detach().reshape(3, -1)[:, ids].T
    return positions * (size / resolution) - size / 2


def model_input(observed, position, quaternion, previous_world, size, resolution):
    """Both clouds share world axes; translation suffices to recenter the history.

    Sensor/body rotation is already included in the recorded world-frame points.
    The sign-canonicalized xyzw orientation additionally conditions the network.
    """
    current = voxelize(observed - position, size, resolution)
    history = voxelize(previous_world - position, size, resolution)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1e-8)
    largest = quaternion.abs().argmax()
    quaternion = torch.where(quaternion[largest] < 0, -quaternion, quaternion)
    pose = quaternion[:, None, None, None].expand(4, resolution, resolution, resolution)
    return torch.cat((current, history, pose), dim=0)[None]
