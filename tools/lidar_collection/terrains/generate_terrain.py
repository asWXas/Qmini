"""Export a single structured terrain tile without starting a simulator."""

import argparse
import json
from pathlib import Path

import trimesh

from .terrain import TERRAIN_KINDS, build_terrain, terrain_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=TERRAIN_KINDS, default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--preview", action="store_true", help="Also render overview.png without a simulator")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    try:
        meshes, origin = build_terrain(args.kind, args.seed, args.difficulty)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    trimesh.util.concatenate(meshes).export(args.output / "terrain.ply")
    metadata = {
        "kind": args.kind,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "size_m": list(terrain_size(args.kind)),
        "spawn_position": origin.tolist(),
        "coordinates": "tile-local, x/y in [0, size], ground z=0, meters",
        "components": [
            {
                "bounds": mesh.bounds.tolist(),
                **{
                    key: mesh.metadata[key]
                    for key in (
                        "role",
                        "variant",
                        "course",
                        "height_m",
                        "ground_block_count",
                        "ground_block_area_m2",
                        "added_pillar_count",
                        "riser_m",
                        "tread_m",
                        "rises",
                        "profile_points",
                        "angle_deg",
                        "layout_version",
                        "flat_ground_area_m2",
                        "flat_ground_fraction",
                    )
                    if key in mesh.metadata
                },
            }
            for mesh in meshes
        ],
    }
    (args.output / "terrain.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.preview:
        from .terrain_preview import render_overview

        render_overview(meshes, origin, terrain_size(args.kind), args.output / "overview.png")
    print(f"Terrain exported to {args.output}")


if __name__ == "__main__":
    main()
