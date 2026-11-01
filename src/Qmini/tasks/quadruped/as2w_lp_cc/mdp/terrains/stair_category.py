"""将楼梯大类按形状和行进方向展开为原生地形配置。"""

import math
from dataclasses import dataclass, field

import isaaclab.terrains as terrain_gen

from .stairs import ShapedStairsTerrainCfg


@dataclass
class StairCategoryCfg:
    proportion: float = 0.6
    shapes: dict[str, float] = field(default_factory=lambda: {"right_angle": 1 / 3, "slanted": 1 / 3, "rounded": 1 / 3})
    directions: dict[str, float] = field(default_factory=lambda: {"up": 0.5, "down": 0.5})
    step_height_range: tuple[float, float] = (0.05, 0.23)
    step_width: float = 0.3
    platform_width: float = 3.0
    border_width: float = 1.0
    riser_angle_deg: float = 60.0
    corner_radius: float = 0.02
    arc_segments: int = 8

    def build(self):
        if not math.isfinite(self.proportion) or not 0 <= self.proportion <= 1:
            raise ValueError("Stair proportion must be in [0, 1]")
        for weights, allowed in ((self.shapes, {"right_angle", "slanted", "rounded"}),
                                 (self.directions, {"up", "down"})):
            if not weights or set(weights) - allowed:
                raise ValueError(f"Expected keys from {sorted(allowed)}")
            if any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError("Category weights must be nonnegative with a positive sum")
        terrains = {}
        for shape, shape_weight in self.shapes.items():
            for direction, direction_weight in self.directions.items():
                proportion = self.proportion * shape_weight / sum(self.shapes.values()) * direction_weight / sum(self.directions.values())
                if proportion == 0:
                    continue
                params = dict(proportion=proportion, step_height_range=self.step_height_range,
                              step_width=self.step_width, platform_width=self.platform_width, border_width=self.border_width)
                # 从中央出生平台向外行进：凹形为上楼梯，凸形为下楼梯。
                if shape == "right_angle":
                    cls = terrain_gen.MeshInvertedPyramidStairsTerrainCfg if direction == "up" else terrain_gen.MeshPyramidStairsTerrainCfg
                    cfg = cls(**params, holes=False)
                    name = f"stairs_{direction}"
                else:
                    cfg = ShapedStairsTerrainCfg(**params, profile=shape, inverted=direction == "up",
                                                riser_angle_deg=self.riser_angle_deg, corner_radius=self.corner_radius,
                                                arc_segments=self.arc_segments)
                    name = f"{shape}_stairs_{direction}"
                terrains[name] = cfg
        return terrains
