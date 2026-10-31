"""Geometry checks for AS2W stair profiles."""

import numpy as np
import pytest

from Qmini.tasks.quadruped.as2w.mdp.terrains import ROUGH_TERRAINS_CFG, ShapedStairsTerrainCfg
from Qmini.tasks.quadruped.as2w.mdp.terrains.stairs import shaped_stairs


@pytest.mark.parametrize("profile", ["slanted", "rounded"])
@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_stair_geometry(profile, inverted, difficulty):
    cfg = ShapedStairsTerrainCfg(size=(8.0, 8.0), profile=profile, inverted=inverted)
    meshes, origin = shaped_stairs(difficulty, cfg)
    mesh = meshes[0]
    assert np.isfinite(mesh.vertices).all()
    assert (mesh.area_faces > 0).all()
    rings = mesh.vertices.reshape(-1, 4, 3)
    np.testing.assert_allclose(rings[-1, :, 2], origin[2])
    assert (origin[2] < 0) == inverted
    assert (np.diff(rings[:, 0, 0]) >= 0).all()
    # Each tread wraps around all four sides, with a level central platform.
    np.testing.assert_allclose(rings[:, 0, 0], rings[:, 0, 1])
    np.testing.assert_allclose(rings[:, 0, 0] + rings[:, 2, 0], 8.0)
    np.testing.assert_allclose(rings[:, 0, 1] + rings[:, 2, 1], 8.0)
    assert (mesh.face_normals[:, 2] >= -1e-12).all()
    assert 8 - 2 * rings[-1, 0, 0] >= cfg.platform_width
    if profile == "rounded":
        r = min(cfg.corner_radius, (0.05 + difficulty * 0.18) * 0.49)
        arc = rings[1:cfg.arc_segments + 2, 0]
        np.testing.assert_allclose((arc[:, 0] - cfg.border_width) ** 2 +
                                   (arc[:, 2] * (-1 if inverted else 1) - r) ** 2, r**2)


def test_terrain_mix():
    assert len(ROUGH_TERRAINS_CFG.sub_terrains) == 11
    assert sum(c.proportion for c in ROUGH_TERRAINS_CFG.sub_terrains.values()) == pytest.approx(1.0)
    assert "boxes" not in ROUGH_TERRAINS_CFG.sub_terrains


def test_as2w_starts_upright():
    from Qmini.assets.as2w import AS2W_CFG
    from scipy.spatial.transform import Rotation

    np.testing.assert_allclose(Rotation.from_quat(AS2W_CFG.init_state.rot).apply([0, 0, 1]), [0, 0, 1])


def test_stair_category_weights():
    from Qmini.tasks.quadruped.as2w.mdp.terrains.stair_category import StairCategoryCfg

    category = StairCategoryCfg(proportion=0.6, shapes={"right_angle": 2, "rounded": 1}, directions={"up": 1, "down": 0})
    terrains = category.build()
    assert set(terrains) == {"stairs_up", "rounded_stairs_up"}
    assert terrains["stairs_up"].proportion == pytest.approx(0.4)
    assert terrains["rounded_stairs_up"].proportion == pytest.approx(0.2)
    assert terrains["rounded_stairs_up"].inverted
    with pytest.raises(ValueError):
        StairCategoryCfg(shapes={"rounded": -1}).build()


def test_stairs_mix_within_direction_columns():
    from types import SimpleNamespace
    from Qmini.tasks.quadruped.as2w.mdp.terrains.stair_category import StairCategoryCfg
    from Qmini.tasks.quadruped.as2w.mdp.terrains.generator import MixedStairsTerrainGenerator

    category = StairCategoryCfg(shapes={"right_angle": 0.5, "slanted": 0.25, "rounded": 0.25})
    generator = MixedStairsTerrainGenerator.__new__(MixedStairsTerrainGenerator)
    generator.cfg = SimpleNamespace(sub_terrains=category.build(), num_cols=4, num_rows=10, difficulty_range=(0, 1))
    generator.np_rng = np.random.default_rng(42)
    cells = []
    generator._get_terrain_mesh = lambda difficulty, cfg: (difficulty, None)
    generator._add_sub_terrain = lambda difficulty, origin, row, col, cfg: cells.append((row, col, difficulty, cfg))
    generator._generate_curriculum_terrains()
    configs = list(generator.cfg.sub_terrains.values())
    for direction, layout in generator.stair_shape_layout.items():
        assert layout.shape == (2, 10)
        names = list(generator.cfg.sub_terrains)
        assert all(names[i].endswith(direction) for i in layout.flat)
        counts = sorted(np.unique(layout, return_counts=True)[1])
        assert counts == [5, 5, 10]
        assert len(set(layout[0])) > 1
    assert len(cells) == 40
    assert all(row / 10 <= difficulty < (row + 1) / 10 for row, col, difficulty, cfg in cells)
    before = {k: v.copy() for k, v in generator.stair_shape_layout.items()}
    generator.np_rng = np.random.default_rng(42)
    generator._generate_curriculum_terrains()
    for name, layout in before.items():
        np.testing.assert_array_equal(layout, generator.stair_shape_layout[name])


def test_slope_directions_mix_at_equal_row_difficulty():
    from types import SimpleNamespace
    from Qmini.tasks.quadruped.as2w.mdp.terrains.slope_category import SlopeCategoryCfg
    from Qmini.tasks.quadruped.as2w.mdp.terrains.generator import MixedStairsTerrainGenerator

    generator = MixedStairsTerrainGenerator.__new__(MixedStairsTerrainGenerator)
    generator.cfg = SimpleNamespace(sub_terrains=SlopeCategoryCfg().build(), num_cols=2, num_rows=10, difficulty_range=(0, 1))
    generator.np_rng = np.random.default_rng(42)
    cells = {}
    generator._get_terrain_mesh = lambda difficulty, cfg: (difficulty, None)
    generator._add_sub_terrain = lambda difficulty, origin, row, col, cfg: cells.update({(row, col): difficulty})
    generator._generate_curriculum_terrains()
    layout = generator.stair_shape_layout['slopes']
    assert layout.shape == (2, 10)
    np.testing.assert_array_equal(np.unique(layout, return_counts=True)[1], [10, 10])
    assert all(len(set(column)) == 2 for column in layout)
    for row in range(10):
        assert cells[row, 0] == cells[row, 1]
        assert row / 10 <= cells[row, 0] < (row + 1) / 10
    with pytest.raises(ValueError):
        SlopeCategoryCfg(directions={'up': -1}).build()
