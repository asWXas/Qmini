"""Episode-safe sequence loading and terrain-disjoint dataset splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_runs(root, map_size):
    root = Path(root).resolve()
    runs = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "metadata.json").is_file():
            continue
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata.get("point_frame") != "world" or metadata.get("quaternion_order") != "xyzw":
            raise ValueError(f"{folder}: expected world-frame points and xyzw quaternions")
        recorded_size = metadata.get("arguments", {}).get("map_size")
        if recorded_size is None or not 0 < map_size <= float(recorded_size):
            raise ValueError(f"{folder}: requested map exceeds recorded ground-truth coverage")
        episodes = []
        for episode in sorted(folder.glob("env_*/episode_*")):
            frames = sorted(episode.glob("frame_*.npz"))
            if frames:
                episodes.append({"path": str(episode), "frames": [str(p) for p in frames]})
        if episodes:
            runs.append({"name": folder.name, "terrain_hash": file_hash(folder / "terrain.ply"), "episodes": episodes})
    if not runs:
        raise ValueError(f"No recorded frame NPZs found under {root}")
    return runs


def split_runs(runs, val_runs, seed):
    """Hold out mesh hashes, including other runs of the identical scene."""
    hashes = sorted({run["terrain_hash"] for run in runs})
    if len(hashes) < 2:
        raise ValueError("Need at least two different terrain meshes for train/validation; collect another scene")
    if val_runs:
        unknown = set(val_runs) - {run["name"] for run in runs}
        if unknown:
            raise ValueError(f"Unknown validation runs: {sorted(unknown)}")
        held_out = {run["terrain_hash"] for run in runs if run["name"] in val_runs}
    else:
        held_out = {hashes[int(np.random.default_rng(seed).integers(len(hashes)))]}
    train = [run for run in runs if run["terrain_hash"] not in held_out]
    validation = [run for run in runs if run["terrain_hash"] in held_out]
    if not train:
        raise ValueError("Validation selection leaves no training terrain")
    return {"train": train, "validation": validation}


def load_frame(path):
    with np.load(path, allow_pickle=False) as data:
        frame = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in ("lidar_points_w", "gt_points_w", "base_pos_w", "base_quat_xyzw")
        }
        for key in ("step", "time_s", "env_id", "episode_id"):
            frame[key] = data[key].item()
            if not np.isfinite(frame[key]):
                raise ValueError(f"{path}: non-finite {key}")
    for key in ("lidar_points_w", "gt_points_w"):
        if frame[key].ndim != 2 or frame[key].shape[1] != 3:
            raise ValueError(f"{path}: invalid {key} shape")
        frame[key] = frame[key][np.isfinite(frame[key]).all(axis=1)]
    if not len(frame["gt_points_w"]):
        raise ValueError(f"{path}: no finite ground truth")
    if frame["base_pos_w"].shape != (3,) or frame["base_quat_xyzw"].shape != (4,):
        raise ValueError(f"{path}: invalid pose shape")
    pose = np.r_[frame["base_pos_w"], frame["base_quat_xyzw"]]
    if not np.isfinite(pose).all() or np.linalg.norm(frame["base_quat_xyzw"]) < 1e-6:
        raise ValueError(f"{path}: invalid pose")
    frame["path"] = str(path)
    return frame


def check_successor(previous, current):
    if previous is not None and (
        current["env_id"] != previous["env_id"]
        or current["episode_id"] != previous["episode_id"]
        or current["step"] <= previous["step"]
        or current["time_s"] <= previous["time_s"]
    ):
        raise ValueError(f"Non-chronological or cross-reset sequence: {current['path']}")

    if previous is not None:
        before, after = Path(previous["path"]), Path(current["path"])
        if before.stem.startswith("frame_") and after.stem.startswith("frame_"):
            before_id, after_id = before.stem.removeprefix("frame_"), after.stem.removeprefix("frame_")
            if before_id.isdigit() and after_id.isdigit() and int(after_id) != int(before_id) + 1:
                raise ValueError(f"Non-contiguous recording (missing/skipped frame): {before} -> {after}")


class SequenceDataset(Dataset):
    """Episode-local windows; plane startup needs no observation before the window."""

    def __init__(
        self,
        runs,
        sequence_length,
        sampling="sequential",
        history_init="empty",
        seed=42,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if sampling not in ("sequential", "random") or history_init not in ("empty", "previous_gt", "plane", "current"):
            raise ValueError("Unsupported window sampling or history initialization")
        self.sequence_length, self.sampling, self.history_init = sequence_length, sampling, history_init
        self.seed, self.epoch = seed, 0
        self.windows = []
        self.sources = []
        for run in runs:
            for episode in run["episodes"]:
                frames = episode["frames"]
                first = 1 if history_init == "previous_gt" and len(frames) > 1 else 0
                for start in range(first, len(frames), sequence_length):
                    self.windows.append(frames[start : start + sequence_length])
                    self.sources.append((frames, start, first))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        paths, start, first = self.sources[index]
        if self.sampling == "random":
            # Per-index/epoch seeds also reproduce under different worker scheduling.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            start = int(rng.integers(first, max(first, len(paths) - self.sequence_length) + 1))
        frames = [load_frame(path) for path in paths[start : start + self.sequence_length]]
        for previous, current in zip(frames[:-1], frames[1:], strict=True):
            check_successor(previous, current)
        for step, frame in enumerate(frames):
            frame["sequence_step"] = step
            frame["sequence_time_s"] = frame["time_s"] - frames[0]["time_s"]
        if self.history_init == "previous_gt" and start > 0:
            initial = load_frame(paths[start - 1])
            check_successor(initial, frames[0])
            frames[0]["initial_history_w"] = initial["gt_points_w"].copy()
            frames[0]["initial_history_path"] = initial["path"]
        return frames


def collate_sequences(sequences):
    """Keep variable point counts and final short windows without padding."""
    return sequences
