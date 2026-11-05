"""A separate block-filled course; the original research layout stays unchanged."""

import numpy as np
import trimesh

from .research_terrain import build_research_terrain, tagged


def footprint(mesh):
    low, high = mesh.bounds
    return (low[0], high[0], low[1], high[1])


def overlaps(a, b, gap=0.0):
    return a[0] < b[1] + gap and a[1] > b[0] - gap and a[2] < b[3] + gap and a[3] > b[2] - gap


def cut_floor(rectangles, cut):
    """Subtract one footprint, keeping disjoint rectangular pieces without a global grid."""
    result = []
    for x0, x1, y0, y1 in rectangles:
        if not overlaps((x0, x1, y0, y1), cut):
            result.append((x0, x1, y0, y1))
            continue
        a, b = max(x0, cut[0]), min(x1, cut[1])
        c, d = max(y0, cut[2]), min(y1, cut[3])
        result.extend(
            rect
            for rect in ((x0, a, y0, y1), (b, x1, y0, y1), (a, b, y0, c), (a, b, d, y1))
            if rect[1] - rect[0] > 1e-9 and rect[3] - rect[2] > 1e-9
        )
    return result


def build_complex_terrain(seed, difficulty):
    original, origin = build_research_terrain(seed, difficulty)
    meshes = original[1:]
    rng = np.random.default_rng([seed, round(difficulty * 1_000_000), 360])
    occupied = [footprint(mesh) for mesh in meshes]
    # Preserve the same spawn and keep NEW objects off the approach strips.
    reserved = [(10.6, 13.4, 10.6, 13.4), (11.4, 12.6, 0, 24), (0, 24, 11.4, 12.6)]
    # Existing cylinders keep their floor: cutting their bounding boxes would leave holes.
    cuts = [footprint(mesh) for mesh in meshes if mesh.metadata["role"] != "pole"]

    def add_box(rect, height, role, variant):
        x0, x1, y0, y1 = rect
        mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, height))
        mesh.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, height / 2))
        # Exposed top and vertical walls only: no buried bottom in supervision.
        mesh.update_faces(mesh.face_normals[:, 2] > -0.5)
        mesh.remove_unreferenced_vertices()
        meshes.append(tagged(mesh, role, variant=variant, height_m=height))
        occupied.append(rect)
        cuts.append(rect)

    def candidate(width, length):
        x, y = rng.uniform((0.3, 0.3), (23.7 - width, 23.7 - length))
        return (x, x + width, y, y + length)

    # Tall square pillars add occlusion, independently of traversable low blocks.
    pillars = 0
    for _ in range(1000):
        rect = candidate(*rng.uniform(0.25, 0.45, 2))
        if any(overlaps(rect, other, 0.3) for other in occupied + reserved):
            continue
        add_box(rect, float(rng.uniform(0.8, 1.4)), "pole", "square_pillar")
        pillars += 1
        if pillars == 10:
            break

    count, block_area = 0, 0.0
    for attempt in range(12000):
        # Broad slabs first; smaller blocks then fill residual spaces.
        dims = rng.uniform(0.7, 2.1, 2) if attempt < 4000 else rng.uniform(0.45, 1.2, 2)
        rect = candidate(*dims)
        if any(overlaps(rect, other, 0.10) for other in occupied + reserved):
            continue
        height = float(rng.uniform(0.03, 0.08 + 0.07 * difficulty))
        add_box(rect, height, "box", "ground_block")
        count += 1
        block_area += float(np.prod(dims))
        if count >= 180:
            break

    floor = [(0.0, 24.0, 0.0, 24.0)]
    for rect in cuts:
        floor = cut_floor(floor, rect)
    vertices, faces = [], []
    for x0, x1, y0, y1 in floor:
        i = len(vertices)
        vertices.extend(((x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)))
        faces.extend(((i, i + 1, i + 2), (i, i + 2, i + 3)))
    ground = tagged(trimesh.Trimesh(vertices, faces, process=False), "ground")
    ground.metadata.update(
        layout_version=1,
        course="research_complex",
        flat_ground_area_m2=float(ground.area),
        flat_ground_fraction=float(ground.area / 576),
        ground_block_count=count,
        ground_block_area_m2=block_area,
        added_pillar_count=pillars,
    )
    return [ground, *meshes], origin
