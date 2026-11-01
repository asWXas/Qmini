"""Local geometry for the four world-aligned 3 by 3 wheel scans."""

import torch


def scan_step_height(hits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the largest second height difference and complete-scan validity.

    A plane (including a slope) has zero second difference. An axis-aligned
    step has a difference equal to its height. This is an edge indicator, not
    an estimate of traversability; diagonal/rounded edges can give smaller values.
    Missing rays must never become an artificial obstacle or a NaN reward.
    """
    if hits.shape[-2:] != (9, 3):
        raise ValueError("Expected a 3 by 3 wheel scan (9 XYZ hits)")
    valid = torch.isfinite(hits).all(dim=(-2, -1))
    z = torch.where(torch.isfinite(hits[..., 2]), hits[..., 2], 0.0).reshape(-1, 3, 3)
    dx = (z[:, 0, :] - 2.0 * z[:, 1, :] + z[:, 2, :]).abs().amax(dim=-1)
    dy = (z[:, :, 0] - 2.0 * z[:, :, 1] + z[:, :, 2]).abs().amax(dim=-1)
    return torch.where(valid, torch.maximum(dx, dy), 0.0), valid


def terrain_motion_scale(env, sensor_names: list[str], rough_scale: float) -> torch.Tensor:
    """Relax style penalties continuously between 3 and 8 cm local edges."""
    if not 0.0 <= rough_scale <= 1.0:
        raise ValueError("rough_scale must be in [0, 1]")
    steps = [scan_step_height(env.scene.sensors[name].data.ray_hits_w.torch)[0] for name in sensor_names]
    roughness = torch.stack(steps, dim=-1).amax(dim=-1)
    gate = ((roughness - 0.03) / 0.05).clamp(0.0, 1.0)
    return 1.0 - (1.0 - rough_scale) * gate
