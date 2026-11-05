"""Run: .venv/bin/python -m pytest tools/lidar_collection/tests/test_geometry.py"""

import numpy as np
import trimesh

from ..core.geometry import sample_local_mesh


def bounds_for(mesh):
    return np.stack((mesh.triangles.min(axis=1), mesh.triangles.max(axis=1)), axis=1)


def test_local_sampling_preserves_vertical_surfaces_and_bounds():
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    center = np.array([0.4, 0.0, 0.0])
    points = sample_local_mesh(mesh, bounds_for(mesh), center, 0.8, 1000, np.random.default_rng(1))
    assert len(points) > 0
    assert np.all(np.abs(points - center) <= 0.400001)
    # Only the box's vertical x=0.5 face intersects this cube.
    np.testing.assert_allclose(points[:, 0], 0.5)
    repeated = sample_local_mesh(mesh, bounds_for(mesh), center, 0.8, 1000, np.random.default_rng(1))
    np.testing.assert_array_equal(points, repeated)


def test_empty_region():
    mesh = trimesh.creation.box()
    points = sample_local_mesh(mesh, bounds_for(mesh), np.full(3, 10.0), 1.0, 1000, np.random.default_rng(1))
    assert points.shape == (0, 3)
