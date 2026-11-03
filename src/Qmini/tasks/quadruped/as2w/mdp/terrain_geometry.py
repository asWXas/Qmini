"""Height-scan geometry shared by gait and terrain-dependent style rewards."""

import math

import torch


class GridScanEdgeDetector:
    """Local accumulated height curvature on an evenly spaced rectangular scan.

    Cache ray ordering once. Sum three adjacent absolute second differences to
    retain edges split across rays by slanted or rounded stair risers. A tilted
    plane still has zero curvature. This is an obstacle-strength metric in
    meters, not a physical step height: a sharp edge can contribute twice its
    height. Keep individual differences for small grids and incomplete windows.
    Invalid triplets contribute nothing; accumulated windows require all five
    hits, and a scan without any valid triplet is unknown.
    """

    def __init__(self, pattern, device):
        starts, _ = pattern.func(pattern, device="cpu")
        xs, xi = starts[:, 0].unique(sorted=True, return_inverse=True)
        ys, yi = starts[:, 1].unique(sorted=True, return_inverse=True)
        self._shape = (len(xs), len(ys))
        keys = xi * len(ys) + yi
        if min(self._shape) < 3 or len(keys.unique()) != len(xs) * len(ys) or len(keys) != len(keys.unique()):
            raise ValueError("Terrain detection requires a complete rectangular grid of at least 3x3")
        for axis in (xs, ys):
            if not torch.allclose(axis.diff(), axis.diff()[0].expand_as(axis.diff()), atol=1e-6, rtol=1e-5):
                raise ValueError("Terrain detection requires evenly spaced rays")
        self._grid_ids = keys.argsort().to(device)

    def __call__(self, hits):
        grid = hits[:, self._grid_ids].reshape(-1, *self._shape, 3)
        finite = torch.isfinite(grid).all(dim=-1)
        z = grid[..., 2].masked_fill(~finite, 0.0)
        valid_x = finite[:, 2:, :] & finite[:, 1:-1, :] & finite[:, :-2, :]
        valid_y = finite[:, :, 2:] & finite[:, :, 1:-1] & finite[:, :, :-2]
        dx = (z[:, 2:, :] - 2 * z[:, 1:-1, :] + z[:, :-2, :]).abs().masked_fill(~valid_x, 0.0)
        dy = (z[:, :, 2:] - 2 * z[:, :, 1:-1] + z[:, :, :-2]).abs().masked_fill(~valid_y, 0.0)
        valid = valid_x.any(dim=(1, 2)) | valid_y.any(dim=(1, 2))
        edge = torch.maximum(dx.amax(dim=(1, 2)), dy.amax(dim=(1, 2)))
        for curvature, valid_triplets, axis in ((dx, valid_x, 1), (dy, valid_y, 2)):
            if curvature.shape[axis] >= 3:
                local_curvature = curvature.unfold(axis, 3, 1).sum(dim=-1)
                valid_window = valid_triplets.unfold(axis, 3, 1).all(dim=-1)
                local_curvature = local_curvature.masked_fill(~valid_window, 0.0)
                edge = torch.maximum(edge, local_curvature.amax(dim=(1, 2)))
        return edge, valid


def terrain_step_gate(edge, step_threshold: float, step_full_height: float):
    """Smooth obstacle strength from zero to one, shared by gait and style costs."""
    if not 0 <= step_threshold < step_full_height or not math.isfinite(step_full_height):
        raise ValueError("Require 0 <= step_threshold < finite step_full_height")
    blend = ((edge - step_threshold) / (step_full_height - step_threshold)).clamp(0.0, 1.0)
    return blend.square() * (3.0 - 2.0 * blend)
