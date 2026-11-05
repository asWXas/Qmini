"""Render actual generated geometry to a standalone overview artifact."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_overview(meshes, origin, size, path):
    fig = plt.figure(figsize=(16, 8), facecolor="#f4f6f8")
    ax = fig.add_subplot(121, projection="3d")
    plan = fig.add_subplot(122)
    for mesh in meshes:
        rgba = mesh.visual.face_colors.astype(float) / 255
        light = np.clip(mesh.face_normals @ np.array([0.3, -0.4, 0.85]), 0, 1)
        colors = rgba.copy()
        colors[:, :3] *= 0.55 + 0.45 * light[:, None]
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=colors, edgecolors="none", antialiaseds=False))
        top = mesh.face_normals[:, 2] > 0.01
        plan.add_collection(
            PolyCollection(mesh.triangles[top, :, :2], facecolors=colors[top], edgecolors="none", antialiaseds=False)
        )
    ax.set(xlim=(0, size[0]), ylim=(0, size[1]), zlim=(0, 1.6), xlabel="x / m", ylabel="y / m", zlabel="z / m")
    ax.set_box_aspect((size[0], size[1], 5))
    ax.view_init(elev=42, azim=-55)
    ax.set_zticks([0, 0.5, 1.0, 1.5])
    ax.set_title("Surface geometry (vertical scale exaggerated)", fontsize=12)
    plan.set(xlim=(0, size[0]), ylim=(0, size[1]), xlabel="x / m", ylabel="y / m")
    plan.set_aspect("equal")
    plan.scatter(*origin[:2], c="#217c50", s=65, marker="*", zorder=5)
    plan.text(origin[0], origin[1] + 0.6, "SPAWN / TURN", ha="center", fontsize=9, color="#217c50")
    if tuple(size) == (24.0, 24.0):
        for x, y, label in (
            (12, 21, "STAIRS  <= 15 cm / riser"),
            (12, 8.5, "SECOND STAIR FLIGHT"),
            (8, 12, "SECOND RAMP"),
            (20.7, 12, "RAMP"),
            (12, 3.6, "GRAVEL"),
            (3.5, 12, "RUBBLE"),
            (4.5, 1.8, "OCCLUDED CORRIDOR"),
            (5, 22.5, "BOXES"),
            (19, 23, "WALL + POLES"),
        ):
            plan.text(
                x,
                y,
                label,
                ha="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
            )
    plan.set_title("Top view / accessible links between zones", fontsize=12)
    fig.suptitle("AS2W terrain collection scene", fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "Static geometry | stairs + ramps + gravel + rubble + structured occlusions",
        ha="center",
        fontsize=11,
        color="#52606d",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
