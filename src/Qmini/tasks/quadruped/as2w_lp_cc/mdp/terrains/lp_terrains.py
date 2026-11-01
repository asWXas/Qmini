"""Fixed type columns and four exact geometry levels, matching paper Appendix C.

From the central spawn platform, inverted pyramids are ascending tasks;
non-inverted pyramids are descending tasks. Geometry does not imply an order
in the task sampler. Every type is present, independent of proportions.
"""

import isaaclab.terrains as terrains


class LPTaskTerrainGenerator(terrains.TerrainGenerator):
    def _generate_curriculum_terrains(self):
        configs = list(self.cfg.sub_terrains.values())
        if self.cfg.num_cols % len(configs):
            raise ValueError("LP terrain columns must be a multiple of terrain type count")
        for col in range(self.cfg.num_cols):
            cfg = configs[col % len(configs)]
            for row in range(self.cfg.num_rows):
                difficulty = row / max(self.cfg.num_rows-1, 1)
                mesh, origin = self._get_terrain_mesh(difficulty, cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)


def lp_terrain_cfg():
    return terrains.TerrainGeneratorCfg(
        class_type=LPTaskTerrainGenerator, curriculum=True,
        size=(12., 12.), border_width=20., num_rows=4, num_cols=5,
        horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75,
        use_cache=False,
        sub_terrains={
            "stairs_up": terrains.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1., step_height_range=(.05, .20), step_width=.3,
                platform_width=2., border_width=1.),
            "stairs_down": terrains.MeshPyramidStairsTerrainCfg(
                proportion=1., step_height_range=(.05, .20), step_width=.3,
                platform_width=2., border_width=1.),
            "rough": terrains.HfRandomUniformTerrainCfg(
                proportion=1., noise_range=(.02, .10), noise_step=.02, border_width=.25),
            "slope_up": terrains.HfInvertedPyramidSlopedTerrainCfg(
                proportion=1., slope_range=(0., .4), platform_width=2., border_width=.25),
            "slope_down": terrains.HfPyramidSlopedTerrainCfg(
                proportion=1., slope_range=(0., .4), platform_width=2., border_width=.25),
        },
    )
