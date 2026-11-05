"""Validate actual course surfaces and the 15 cm riser constraint."""

import numpy as np
import pytest

from ..terrains.terrain import build_terrain


def top_height(mesh, x, y):
    """Independent vertical intersection using barycentric coordinates, no rtree needed."""
    tri = mesh.triangles
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = np.abs(det) > 1e-10
    delta = np.array([x, y]) - a[valid, :2]
    u = (delta[:, 0] * v1[valid, 1] - delta[:, 1] * v1[valid, 0]) / det[valid]
    v = (v0[valid, 0] * delta[:, 1] - v0[valid, 1] * delta[:, 0]) / det[valid]
    inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
    if not inside.any():
        return None
    heights = a[valid, 2] + u * (b[valid, 2] - a[valid, 2]) + v * (c[valid, 2] - a[valid, 2])
    return heights[inside].max()


@pytest.mark.parametrize("difficulty", [0, 0.5, 1])
def test_stair_risers_and_slope_endpoints(difficulty):
    meshes, _ = build_terrain("research", seed=42, difficulty=difficulty)
    flights = [mesh for mesh in meshes if mesh.metadata["role"] == "stairs"]
    assert len(flights) == 2
    for stairs, expected_steps in zip(flights, (6, 4), strict=True):
        boundaries = np.unique(stairs.vertices[:, 0])
        mid_y = stairs.bounds[:, 1].mean()
        heights = np.array([top_height(stairs, x, mid_y) for x in (boundaries[:-1] + boundaries[1:]) / 2])
        jumps = np.diff(np.r_[0, heights, 0])
        assert np.max(np.abs(jumps)) <= 0.15 + 1e-8
        assert np.count_nonzero(jumps > 0.01) == expected_steps
        assert np.count_nonzero(jumps < -0.01) == expected_steps
    slope = next(mesh for mesh in meshes if mesh.metadata["role"] == "slope")
    assert top_height(slope, 18.5, 9) == pytest.approx(0)
    assert top_height(slope, 18.5, 15) == pytest.approx(0)
    grade = top_height(slope, 18.5, 11) - top_height(slope, 18.5, 10)
    assert grade == pytest.approx(np.tan(np.deg2rad(8 + 7 * difficulty)))


def test_surface_regions_and_ground_seams():
    meshes, _ = build_terrain("research", seed=11, difficulty=1)
    roles = {mesh.metadata["role"] for mesh in meshes}
    assert roles == {"ground", "stairs", "slope", "gravel", "rubble", "wall", "box", "pole"}
    ground = meshes[0]
    assert top_height(ground, 12, 12) == 0
    assert top_height(ground, 12, 6) is None  # no buried plane under gravel
    for role in ("gravel", "rubble"):
        mesh = next(mesh for mesh in meshes if mesh.metadata["role"] == role)
        vertices = mesh.vertices
        lower, upper = mesh.bounds
        edge = np.any(np.isclose(vertices[:, :2], lower[:2]) | np.isclose(vertices[:, :2], upper[:2]), axis=1)
        np.testing.assert_allclose(vertices[edge, 2], 0, atol=1e-8)
        assert mesh.bounds[1, 2] > (0.03 if role == "gravel" else 0.15)
        assert np.all(mesh.face_normals[:, 2] > 0)


@pytest.mark.parametrize("difficulty", [0, 0.7, 1])
def test_dense_layout_limits_empty_ground_and_preserves_spawn(difficulty):
    meshes, _ = build_terrain("research", seed=42, difficulty=difficulty)
    ground = meshes[0]
    flat = np.all(np.abs(ground.triangles[:, :, 2]) < 1e-9, axis=1)
    assert ground.area_faces[flat].sum() / (24 * 24) < 0.25
    for x in (10.8, 12, 13.2):
        for y in (10.8, 12, 13.2):
            assert top_height(ground, x, y) == pytest.approx(0)
    assert sum(mesh.metadata["role"] == "slope" for mesh in meshes) == 2
    rubble = next(mesh for mesh in meshes if mesh.metadata["role"] == "rubble")
    assert np.prod(rubble.extents[:2]) > 40
