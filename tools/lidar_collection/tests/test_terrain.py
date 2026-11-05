"""Verify geometry generation without the simulator."""

import numpy as np
import pytest
import trimesh

from ..terrains.terrain import TERRAIN_KINDS, build_terrain, terrain_size


@pytest.mark.parametrize("kind", TERRAIN_KINDS)
def test_terrain_bounds_and_clear_spawn(kind):
    meshes, origin = build_terrain(kind, seed=7, difficulty=1.0)
    mesh = trimesh.util.concatenate(meshes)
    assert np.isfinite(mesh.vertices).all()
    assert np.all(mesh.bounds[0, :2] >= -1e-6)
    size = np.asarray(terrain_size(kind))
    assert np.all(mesh.bounds[1, :2] <= size + 1e-6)
    np.testing.assert_allclose(origin, [*list(size / 2), 0])
    for obstacle in meshes[1:]:
        # Every obstacle must lie outside the central spawn square.
        bounds = obstacle.bounds[:, :2]
        assert np.any(bounds[1] <= origin[:2] - 1.2) or np.any(bounds[0] >= origin[:2] + 1.2)
    repeated, _ = build_terrain(kind, seed=7, difficulty=1.0)
    np.testing.assert_array_equal(mesh.vertices, trimesh.util.concatenate(repeated).vertices)


def test_seed_changes_geometry():
    first, _ = build_terrain(seed=1)
    second, _ = build_terrain(seed=2)
    assert not np.array_equal(trimesh.util.concatenate(first).vertices, trimesh.util.concatenate(second).vertices)
