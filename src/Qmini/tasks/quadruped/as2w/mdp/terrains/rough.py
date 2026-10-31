# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AS2W 混合地形配置。"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .stair_category import StairCategoryCfg
from .slope_category import SlopeCategoryCfg
from .generator import MixedStairsTerrainGenerator

STAIRS_CFG = StairCategoryCfg(
    proportion=0.50,
    shapes={"right_angle": 1 / 3, "slanted": 1 / 3, "rounded": 1 / 3},
    directions={"up": 0.5, "down": 0.5},
    step_height_range=(0.05, 0.23),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    riser_angle_deg=60.0,
    corner_radius=0.02,
    arc_segments=8,
)

SLOPES_CFG = SlopeCategoryCfg(
    proportion=0.2,
    directions={"up": 0.5, "down": 0.5},
    angle_range_deg=(10.0, 45.0),
    platform_width=2.0,
    border_width=0.25,
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=MixedStairsTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        **STAIRS_CFG.build(),
        **SLOPES_CFG.build(),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=0.10,
            amplitude_range=(0.02, 0.10),
            num_waves=3,
            border_width=0.25,
        )
    },
)
