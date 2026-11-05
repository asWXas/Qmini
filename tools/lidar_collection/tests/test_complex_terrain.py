"""Check exposed block geometry, spacing and preservation of the original features."""

import numpy as np
import pytest

from ..terrains.complex_terrain import footprint, overlaps
from ..terrains.terrain import build_terrain
from .test_research_terrain import top_height


def union_area(rectangles):
    """Independent sweep: original corner walls have overlapping footprints."""
    x_edges = sorted({x for rect in rectangles for x in rect[:2]})
    area = 0.0
    for left, right in zip(x_edges[:-1], x_edges[1:], strict=True):
        x = (left + right) / 2
        intervals = sorted((r[2], r[3]) for r in rectangles if r[0] < x < r[1])
        end, length = 0.0, 0.0
        for low, high in intervals:
            length += max(0.0, high - max(low, end))
            end = max(end, high)
        area += (right - left) * length
    return area


@pytest.mark.parametrize("seed,difficulty", [(42, 0.7), (7, 1), (11, 0)])
def test_complex_blocks_and_original_features(seed, difficulty):
    meshes, _ = build_terrain("research_complex", seed, difficulty)
    original, _ = build_terrain("research", seed, difficulty)
    # All existing stairs, ramps and rough regions survive exactly.
    for before, after in zip(original[1:], meshes[1 : len(original)], strict=True):
        np.testing.assert_array_equal(before.vertices, after.vertices)
        np.testing.assert_array_equal(before.faces, after.faces)
    ground = meshes[0]
    additions = meshes[len(original) :]
    blocks = [m for m in additions if m.metadata["variant"] == "ground_block"]
    pillars = [m for m in additions if m.metadata["variant"] == "square_pillar"]
    assert len(blocks) >= 100
    assert len(pillars) == 10
    assert sum(np.prod(m.extents[:2]) for m in blocks) > 125
    assert ground.area / 576 < 0.5
    for x in (10.8, 12, 13.2):
        for y in (10.8, 12, 13.2):
            assert top_height(ground, x, y) == pytest.approx(0)
    for i, mesh in enumerate(additions):
        center = mesh.bounds[:, :2].mean(axis=0)
        assert top_height(ground, *center) is None
        assert not np.any(mesh.face_normals[:, 2] < -0.5)  # no buried bottom labels
        for other in [*original[1:], *additions[:i]]:
            assert not overlaps(footprint(mesh), footprint(other), gap=0.099)
        if mesh.metadata["variant"] == "ground_block":
            height = top_height(mesh, *center)
            assert 0.03 <= height <= 0.15
            assert height == pytest.approx(mesh.metadata["height_m"])
            assert len(np.unique(mesh.vertices[:, 2])) == 2  # flat top, vertical sides
    # Cutting around all rectangular footprints leaves no hidden floor under them.
    rect_area = union_area(
        [
            footprint(m)
            for m in meshes[1:]
            if m.metadata["role"] != "pole" or m.metadata.get("variant") == "square_pillar"
        ]
    )
    assert ground.area + rect_area == pytest.approx(576, abs=1e-6)
