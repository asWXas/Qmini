"""24 m structured-terrain collection course, inspired by arXiv:2206.08077.

Slopes, gravel and rubble extend the paper's urban scene distribution. All surfaces
are static. The rough patches represent exposed upper surfaces, not loose bodies.
"""

import numpy as np
import trimesh

COURSE_SIZE = (24.0, 24.0)
COLORS = {
    "ground": (185, 192, 195, 255),
    "stairs": (216, 160, 83, 255),
    "slope": (104, 160, 184, 255),
    "gravel": (138, 140, 126, 255),
    "rubble": (126, 111, 103, 255),
    "wall": (111, 129, 144, 255),
    "box": (173, 128, 91, 255),
    "pole": (158, 171, 120, 255),
}


def tagged(mesh, role, **properties):
    mesh.metadata.update(role=role, **properties)
    mesh.visual.face_colors = COLORS[role]
    return mesh


def surface_grid(x, y, z, role):
    """Single upward-facing surface; no buried box bottoms in rough-patch labels."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ids = np.arange(xx.size).reshape(xx.shape)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()
    faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    return tagged(trimesh.Trimesh(np.column_stack((xx.ravel(), yy.ravel(), z.ravel())), faces, process=False), role)


def extruded_profile(profile, width, center, role, along_y=False, **properties):
    """Expose treads/risers and sides of a longitudinal profile, with no internal boxes."""
    vertices, faces = [], []

    def quad(points):
        start = len(vertices)
        vertices.extend(points)
        faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))

    for (x0, z0), (x1, z1) in zip(profile[:-1], profile[1:], strict=True):
        quad([(x0, -width / 2, z0), (x1, -width / 2, z1), (x1, width / 2, z1), (x0, width / 2, z0)])
        if x1 > x0 and max(z0, z1) > 0:
            quad([(x0, -width / 2, 0), (x1, -width / 2, 0), (x1, -width / 2, z1), (x0, -width / 2, z0)])
            quad([(x1, width / 2, 0), (x0, width / 2, 0), (x0, width / 2, z0), (x1, width / 2, z1)])
    vertices = np.asarray(vertices, dtype=float)
    if along_y:
        vertices[:, :2] = np.column_stack((-vertices[:, 1], vertices[:, 0]))
    vertices[:, :2] += center
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return tagged(mesh, role, **properties)


def build_research_terrain(seed, difficulty):
    rng = np.random.default_rng([seed, round(difficulty * 1_000_000)])
    meshes, patches = [], []

    rise = min(0.15, 0.08 + 0.07 * difficulty)

    def stairs(center, width, count):
        tread = float(rng.uniform(0.30, 0.45))
        landing = 1.4
        length = 2 * count * tread + landing
        x = -length / 2
        profile = [(x, 0.0)]
        for step in range(1, count + 1):
            profile.extend(((x, step * rise), (x + tread, step * rise)))
            x += tread
        x += landing
        profile.append((x, count * rise))
        for step in range(count - 1, -1, -1):
            profile.extend(((x, step * rise), (x + tread, step * rise)))
            x += tread
        meshes.append(
            extruded_profile(
                profile,
                width,
                center,
                "stairs",
                riser_m=rise,
                tread_m=tread,
                rises=count,
                profile_points=profile,
                lane_center=center,
            )
        )
        cx, cy = center
        patches.append((cx - length / 2, cx + length / 2, cy - width / 2, cy + width / 2))

    # Two complete flights; both return to the surrounding surface without an end drop.
    stairs((12, 18.5), 4.4, 6)
    stairs((12, 8.5), 2.6, 4)

    angle = 8.0 + 7.0 * difficulty
    for center, width, run in (((18.5, 12), 3.8, 2.5), ((8, 12), 2.4, 2.0)):
        height = run * np.tan(np.deg2rad(angle))
        half = run + 0.5
        profile = [(-half, 0), (-0.5, height), (0.5, height), (half, 0)]
        meshes.append(extruded_profile(profile, width, center, "slope", along_y=True, angle_deg=angle))
        cx, cy = center
        patches.append((cx - width / 2, cx + width / 2, cy - half, cy + half))

    # South: fine faceted gravel, tapered at every boundary for smooth entry.
    x, y = np.linspace(7.4, 16.6, 116), np.linspace(1.8, 6.4, 59)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    edge = np.clip(np.minimum.reduce((xx - 7.4, 16.6 - xx, yy - 1.8, 6.4 - yy)) / 0.4, 0, 1)
    height = (0.02 + 0.04 * difficulty) * rng.uniform(0.05, 1.0, xx.shape) * edge
    meshes.append(surface_grid(x, y, height, "gravel"))
    patches.append((7.4, 16.6, 1.8, 6.4))

    # West: irregular rock upper envelopes, on a low mound. Static and single-valued.
    x, y = np.linspace(1.8, 6.4, 93), np.linspace(7.2, 16.8, 193)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    edge = np.clip(np.minimum.reduce((xx - 1.8, 6.4 - xx, yy - 7.2, 16.8 - yy)) / 0.45, 0, 1)
    stones = np.zeros_like(xx)
    for _ in range(220):
        cx, cy = rng.uniform(2.0, 6.2), rng.uniform(7.4, 16.6)
        rx, ry = rng.uniform(0.16, 0.42, 2)
        theta = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        u, v = np.cos(theta) * dx + np.sin(theta) * dy, -np.sin(theta) * dx + np.cos(theta) * dy
        rock = np.sqrt(np.clip(1 - (u / rx) ** 2 - (v / ry) ** 2, 0, 1))
        stones = np.maximum(stones, rock * rng.uniform(0.08, 0.16 + 0.16 * difficulty))
    mound = (0.03 + 0.09 * difficulty) * np.exp(-(((xx - 4.1) / 1.6) ** 2) - ((yy - 12) / 3) ** 2)
    meshes.append(surface_grid(x, y, (stones + mound) * edge, "rubble"))
    patches.append((1.8, 6.4, 7.2, 16.8))

    def box(extents, position, role):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(position)
        meshes.append(tagged(mesh, role))
        # Ground under solid obstacles is not an exposed surface.
        patches.append(
            (
                position[0] - extents[0] / 2,
                position[0] + extents[0] / 2,
                position[1] - extents[1] / 2,
                position[1] + extents[1] / 2,
            )
        )

    # Paper-inspired corner occlusions. Open entries and links to the four lanes.
    for y in (2.8, 5.4):
        box((4.5, 0.18, 1.1), (4.5, y, 0.55), "wall")
    box((0.18, 1.6, 1.1), (2.25, 3.5, 0.55), "wall")
    for center in (
        (3.5, 19),
        (5.2, 20.5),
        (7, 21),
        (20, 4),
        (21, 6.5),
        (14.9, 11),
        (14.8, 14.4),
        (18.1, 3.2),
        (3.8, 22),
    ):
        width, length = rng.uniform(0.4, 1.3, 2)
        height = float(rng.uniform(0.08, 0.25))
        box((width, length, height), (*center, height / 2), "box")
    box((3.8, 0.18, 1.1), (19.5, 21.5, 0.55), "wall")
    box((0.18, 2.8, 1.1), (21.3, 20.1, 0.55), "wall")
    box((2.4, 0.18, 1.1), (9.7, 15.2, 0.55), "wall")
    box((0.18, 2.0, 1.1), (21.8, 10.5, 0.55), "wall")
    for center in ((9.8, 11), (14.6, 9.8), (15.1, 15.3), (17.3, 19), (18.9, 18), (20.5, 19.5)):
        mesh = trimesh.creation.cylinder(radius=float(rng.uniform(0.1, 0.18)), height=1.0, sections=20)
        mesh.apply_translation((*center, 0.5))
        meshes.append(tagged(mesh, "pole"))

    # Floor is cut around obstacle/patch footprints, not buried beneath their top surfaces.
    x = np.unique([*np.linspace(0, 24, 121), 10.6, 13.4, *(v for p in patches for v in p[:2])])
    y = np.unique([*np.linspace(0, 24, 121), 10.6, 13.4, *(v for p in patches for v in p[2:])])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    # Replace the former empty plaza with low roughness. Keep only the 2.8 m
    # spawn square, 1.2 m connecting strips and a narrow outside boundary flat.
    spawn_distance = np.maximum(np.abs(xx - 12) - 1.4, np.abs(yy - 12) - 1.4)
    path_distance = np.minimum(np.abs(xx - 12), np.abs(yy - 12)) - 0.6
    border_distance = np.minimum.reduce((xx, 24 - xx, yy, 24 - yy)) - 0.25
    blend = np.clip(np.minimum.reduce((spawn_distance, path_distance, border_distance)) / 0.35, 0, 1)
    for x0, x1, y0, y1 in patches:
        dx = np.maximum(np.maximum(x0 - xx, xx - x1), 0)
        dy = np.maximum(np.maximum(y0 - yy, yy - y1), 0)
        blend = np.minimum(blend, np.clip(np.hypot(dx, dy) / 0.3, 0, 1))
    noise = rng.uniform(0.15, 1.0, xx.shape)
    waves = 0.5 + 0.5 * np.sin(xx * 1.8) * np.cos(yy * 1.4)
    height = ((0.015 + 0.025 * difficulty) * noise + 0.025 * waves) * blend
    ground = surface_grid(x, y, height, "ground")
    centers = ground.triangles_center
    keep = np.ones(len(centers), dtype=bool)
    for x0, x1, y0, y1 in patches:
        keep &= ~((centers[:, 0] > x0) & (centers[:, 0] < x1) & (centers[:, 1] > y0) & (centers[:, 1] < y1))
    ground.update_faces(keep)
    ground.remove_unreferenced_vertices()
    flat_faces = np.all(np.abs(ground.triangles[:, :, 2]) < 1e-9, axis=1)
    flat_area = float(ground.area_faces[flat_faces].sum())
    ground.metadata.update(flat_ground_area_m2=flat_area, flat_ground_fraction=flat_area / (24 * 24), layout_version=2)
    meshes.insert(0, ground)
    return meshes, np.array([12.0, 12.0, 0.0])
