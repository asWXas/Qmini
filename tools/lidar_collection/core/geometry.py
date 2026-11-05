"""Local mesh supervision without simulator dependencies."""

import numpy as np
import trimesh


def sample_local_mesh(mesh, bounds, center, size, density, rng):
    """Sample all mesh surfaces in a world-aligned local cube, including occluded faces."""
    lo, hi = center - size / 2, center + size / 2
    selected = np.all(bounds[:, 1] >= lo, axis=1) & np.all(bounds[:, 0] <= hi, axis=1)
    if not selected.any():
        return np.empty((0, 3), dtype=np.float32)
    local = mesh.submesh([np.flatnonzero(selected)], append=True, repair=False)
    # Clip before sampling so large triangles do not waste the sample budget.
    for axis in range(3):
        normal = np.eye(3)[axis]
        local = local.slice_plane(lo, normal).slice_plane(hi, -normal)
    count = int(np.ceil(local.area * density))
    if count == 0:
        return np.empty((0, 3), dtype=np.float32)
    points, _ = trimesh.sample.sample_surface(local, count, seed=rng)
    return points.astype(np.float32)
