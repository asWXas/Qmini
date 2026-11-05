"""Paper-style corruption of measurements/odometry; ground truth stays clean.

Noise bounds and trajectory mirroring follow III-E. Patch sizes/counts and
outlier density are explicit implementation choices, not published constants.
"""

import numpy as np
import torch


def mirror_sequence(sequence, signs):
    """Reflect a whole trajectory, including pose, using one fixed diagonal S."""
    signs = np.asarray(signs, dtype=np.float32)
    result = []
    for frame in sequence:
        mirrored = dict(frame)
        for key in ("lidar_points_w", "gt_points_w", "base_pos_w"):
            if key in frame:
                mirrored[key] = frame[key] * signs
        if "initial_history_w" in frame:
            mirrored["initial_history_w"] = frame["initial_history_w"] * signs
        quaternion = frame["base_quat_xyzw"].copy()
        quaternion[:3] *= float(np.prod(signs)) * signs  # R' = S R S remains a proper rotation
        mirrored["base_quat_xyzw"] = quaternion
        result.append(mirrored)
    return result


def augment_measurement(points_world, true_position, config):
    recipe = config.get("augmentation", "basic")
    if recipe == "none":
        return points_world, true_position
    device = points_world.device
    if recipe == "basic":
        points = points_world[torch.rand(len(points_world), device=device) >= config["point_dropout"]]
        return points + torch.randn_like(points) * config["noise_std"], true_position
    if recipe != "paper":
        raise ValueError(f"Unknown augmentation recipe: {recipe}")
    points = points_world - true_position
    points = points + (torch.rand_like(points) * 2 - 1) * config.get("point_noise_bound", 0.05)
    direction = torch.rand((), device=device) * (2 * torch.pi)
    axis = torch.stack((direction.cos(), direction.sin(), torch.zeros_like(direction)))
    angle = (torch.rand((), device=device) * 2 - 1) * (torch.pi / 180)
    points = (
        points * angle.cos()
        + torch.linalg.cross(axis.expand_as(points), points) * angle.sin()
        + (points @ axis)[:, None] * axis * (1 - angle.cos())
    )
    # Two height patches and one missing patch, with sizes chosen for this local map.
    for _ in range(2):
        center = (torch.rand(2, device=device) - 0.5) * config["map_size"]
        radius = 0.2 + torch.rand((), device=device) * 0.4
        mask = torch.linalg.vector_norm(points[:, :2] - center, dim=1) < radius
        points[mask, 2] += (torch.rand((), device=device) * 2 - 1) * 0.05
    center = (torch.rand(2, device=device) - 0.5) * config["map_size"]
    radius = 0.15 + torch.rand((), device=device) * 0.45
    points = points[torch.linalg.vector_norm(points[:, :2] - center, dim=1) >= radius]
    if len(points):
        center = points[torch.randint(len(points), (), device=device)]
        center = center + (torch.rand(3, device=device) - 0.5) * 0.4
        outliers = center + torch.randn((32, 3), device=device) * 0.03
        points = torch.cat((points, outliers))
    # Current points and map origin share the same pose error, so it cancels for
    # current local measurements; old predicted maps retain their previous error.
    estimated_position = true_position + (torch.rand_like(true_position) * 2 - 1) * 0.05
    return points + estimated_position, estimated_position
