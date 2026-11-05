"""Deterministic structured terrain geometry; usable without Isaac Lab or a GPU."""

import numpy as np
import trimesh

from .complex_terrain import build_complex_terrain
from .research_terrain import COURSE_SIZE, build_research_terrain

RESEARCH_KINDS = ("research", "research_complex")
TERRAIN_KINDS = ("flat", "stairs", "boxes", "corridor", "poles", "mixed", *RESEARCH_KINDS)


def terrain_size(kind):
    return COURSE_SIZE if kind in RESEARCH_KINDS else (8.0, 8.0)


def build_terrain(kind="mixed", seed=42, difficulty=0.5, size=None):
    """Return meshes and a clear central spawn point in positive tile coordinates.

    Difficulty varies dimensions; seed varies placement. A central 2.4 m square is
    kept clear. Primitives are concatenated, not boolean-unioned.
    """
    if kind not in TERRAIN_KINDS:
        raise ValueError(f"Unknown terrain kind: {kind}")
    if not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("difficulty must be in [0, 1]")
    size = np.asarray(terrain_size(kind) if size is None else size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 8):
        raise ValueError("Terrain dimensions must be finite and at least 8 meters")
    if kind in RESEARCH_KINDS:
        if not np.array_equal(size, COURSE_SIZE):
            raise ValueError("Research course requires a 24 x 24 meter tile")
        builder = build_research_terrain if kind == "research" else build_complex_terrain
        return builder(seed, difficulty)
    rng = np.random.default_rng(seed)
    cx, cy = size / 2
    meshes = []

    def box(extents, position):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(position)
        meshes.append(mesh)

    box((size[0], size[1], 0.1), (cx, cy, -0.05))
    if kind in ("stairs", "mixed"):
        step_height = min(0.15, 0.08 + 0.07 * difficulty)
        tread = rng.uniform(0.25, 0.4)
        # Four treads approach from the clear central platform toward +x.
        for index in range(4):
            height = step_height * (index + 1)
            box((tread, 1.6, height), (cx + 1.4 + (index + 0.5) * tread, cy, height / 2))
    if kind in ("boxes", "mixed"):
        for y in (-2.2, 0.0, 2.2):
            width, length = rng.uniform(0.35, 0.8, size=2)
            height = rng.uniform(0.08, 0.12 + 0.13 * difficulty)
            box((width, length, height), (cx - 2.4 + rng.uniform(-0.2, 0.2), cy + y, height / 2))
    if kind in ("corridor", "mixed"):
        half_width = 1.6 - 0.3 * difficulty
        for side in (-1, 1):
            box((size[0] - 0.4, 0.15, 1.0), (cx, cy + side * (half_width + 0.075), 0.5))
    if kind in ("poles", "mixed"):
        for x in (-2.4, 0.0, 2.4):
            radius = rng.uniform(0.08, 0.16)
            mesh = trimesh.creation.cylinder(radius=radius, height=1.2, sections=16)
            mesh.apply_translation((cx + x, cy + 2.7, 0.6))
            meshes.append(mesh)
    return meshes, np.array([cx, cy, 0.0])
