"""斜坡总占比、方向比例和共用坡度范围。"""

import math
from dataclasses import dataclass, field

import isaaclab.terrains as terrain_gen


@dataclass
class SlopeCategoryCfg:
    proportion: float = 0.25
    directions: dict[str, float] = field(default_factory=lambda: {"up": 0.5, "down": 0.5})
    angle_range_deg: tuple[float, float] = (10.0, 45.0)
    platform_width: float = 2.0
    border_width: float = 0.25

    def build(self):
        if not math.isfinite(self.proportion) or not 0 <= self.proportion <= 1:
            raise ValueError("Slope proportion must be in [0, 1]")
        if not self.directions or set(self.directions) - {"up", "down"}:
            raise ValueError("Slope directions must be up/down")
        weights = list(self.directions.values())
        if any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("Direction weights must be nonnegative with a positive sum")
        lower, upper = self.angle_range_deg
        if not 0 <= lower <= upper < 90:
            raise ValueError("Expected 0 <= minimum angle <= maximum angle < 90 degrees")
        slopes = tuple(math.tan(math.radians(angle)) for angle in self.angle_range_deg)
        return {
            f"slope_{direction}": (
                terrain_gen.HfInvertedPyramidSlopedTerrainCfg if direction == "up"
                else terrain_gen.HfPyramidSlopedTerrainCfg
            )(
                proportion=self.proportion * weight / sum(weights), slope_range=slopes,
                slope_threshold=max(1.1, slopes[1] + 0.1),
                platform_width=self.platform_width, border_width=self.border_width,
            )
            for direction, weight in self.directions.items() if weight > 0 and self.proportion > 0
        }
