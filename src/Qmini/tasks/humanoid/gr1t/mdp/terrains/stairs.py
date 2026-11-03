"""与原生金字塔楼梯布局一致的倾斜立面、圆角台阶。"""

import math
from typing import Literal

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def shaped_stairs(difficulty, cfg):
    height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if not 0 < cfg.platform_width < cfg.size[0] or height <= 0 or cfg.step_width <= 0:
        raise ValueError("Invalid stair size, height or platform width")
    if cfg.profile == "slanted":
        if not 0 < cfg.riser_angle_deg < 90:
            raise ValueError("Riser angle must be between 0 and 90 degrees")
        run = height / math.tan(math.radians(cfg.riser_angle_deg))
        rise = [(run, height)]
    elif cfg.profile == "rounded":
        if cfg.corner_radius <= 0 or cfg.arc_segments < 2:
            raise ValueError("Positive corner radius and at least two arc segments required")
        radius = min(cfg.corner_radius, height * 0.49)
        theta = np.linspace(0, math.pi / 2, cfg.arc_segments + 1)[1:]
        rise = [(radius * math.sin(t), radius * (1 - math.cos(t))) for t in theta]
        rise.append((radius, height - radius))
        rise.extend((radius + radius * (1 - math.cos(t)), height - radius + radius * math.sin(t)) for t in theta)
        run = 2 * radius
    else:
        raise ValueError(f"Unknown stair profile: {cfg.profile}")
    if run >= cfg.step_width:
        raise ValueError("Riser run must be smaller than step_width to retain a horizontal tread")

    length, width = cfg.size
    count = int(
        min(
            (side - 2 * cfg.border_width - cfg.platform_width - 2 * run) // (2 * cfg.step_width) + 1
            for side in cfg.size
        )
    )
    if count < 1 or cfg.border_width < 0:
        raise ValueError("Terrain must fit at least one step and a nonnegative border")
    profile = [(0.0, 0.0)]
    if cfg.border_width > 0:
        profile.append((cfg.border_width, 0.0))
    for index in range(count):
        inset, z = cfg.border_width + index * cfg.step_width, index * height
        profile.extend((inset + dx, z + dz) for dx, dz in rise)
        if index < count - 1:
            profile.append((inset + cfg.step_width, (index + 1) * height))
    sign = -1 if cfg.inverted else 1
    vertices = []
    for inset, z in profile:
        vertices.extend(
            (
                (inset, inset, sign * z),
                (length - inset, inset, sign * z),
                (length - inset, width - inset, sign * z),
                (inset, width - inset, sign * z),
            )
        )
    faces = []
    for i in range(len(profile) - 1):
        for side in range(4):
            a, b = 4 * i + side, 4 * i + (side + 1) % 4
            faces.extend(((a, b, b + 4), (a, b + 4, a + 4)))
    a = 4 * (len(profile) - 1)
    faces.extend(((a, a + 1, a + 2), (a, a + 2, a + 3)))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    origin = np.array([length / 2, width / 2, sign * count * height])
    return [mesh], origin


@configclass
class ShapedStairsTerrainCfg(SubTerrainBaseCfg):
    function = shaped_stairs
    profile: Literal["slanted", "rounded"] = "slanted"
    step_height_range: tuple[float, float] = (0.05, 0.23)
    step_width: float = 0.3
    platform_width: float = 3.0
    border_width: float = 1.0
    riser_angle_deg: float = 60.0
    corner_radius: float = 0.02
    arc_segments: int = 8
    inverted: bool = False
