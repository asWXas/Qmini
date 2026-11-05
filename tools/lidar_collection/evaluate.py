"""Offline reconstruction evaluation entry point."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.lidar_collection.training.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
