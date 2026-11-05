"""Command-line entry point; implementation lives in terrains/generate_terrain.py."""

import sys
from pathlib import Path

# Also support invoking this file by absolute path from outside the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.lidar_collection.terrains.generate_terrain import main  # noqa: E402

if __name__ == "__main__":
    main()
