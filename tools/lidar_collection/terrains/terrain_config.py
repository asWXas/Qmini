"""Adapt standalone terrain geometry to the existing AS2W simulation."""

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .terrain import build_terrain, terrain_size


def structured_terrain(difficulty, cfg):
    # The research builder derives variation from both seed and difficulty, just
    # like the offline exporter. Identical seed/difficulty gives identical geometry.
    return build_terrain(cfg.kind, int(cfg.seed or 0), difficulty, cfg.size)


@configclass
class StructuredTerrainCfg(SubTerrainBaseCfg):
    function = structured_terrain
    kind: str = "mixed"


class RecordingTerrainGenerator(TerrainGenerator):
    """Keep the exact final mesh, including tile transformations and border."""

    mesh = None

    def __init__(self, cfg, device):
        super().__init__(cfg, device=device)
        RecordingTerrainGenerator.mesh = self.terrain_mesh.copy()


def configure_terrain(cfg, kind, seed, rows, cols, difficulty=None):
    RecordingTerrainGenerator.mesh = None
    if kind == "existing":
        cfg.scene.terrain.terrain_generator.class_type = RecordingTerrainGenerator
        return
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        class_type=RecordingTerrainGenerator,
        seed=seed,
        size=terrain_size(kind),
        num_rows=rows,
        num_cols=cols,
        border_width=5.0,
        curriculum=False,
        use_cache=False,
        difficulty_range=(difficulty, difficulty) if difficulty is not None else (0.0, 1.0),
        sub_terrains={kind: StructuredTerrainCfg(kind=kind, proportion=1.0)},
    )
