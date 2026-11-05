"""Training data boundaries, coordinate transforms, leakage and gradient checks."""

import json

import numpy as np
import pytest
import torch

from ..training.datasets.augmentation import mirror_sequence
from ..training.datasets.sequence import SequenceDataset, check_successor, discover_runs, load_frame, split_runs
from ..training.runtime.engine import forward_frame
from ..training.models.dense3d import TerrainUNet
from ..training.geometry.voxel import decode, model_input, voxelize


def test_voxel_centroids_boundaries_and_translation_feedback():
    points = torch.tensor([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [1.6, 0, 0], [float("nan"), 0, 0]])
    grid = voxelize(points, 3.2, 64)
    assert grid[0].sum() == 1
    torch.testing.assert_close(grid[1:, 32, 32, 32], torch.full((3,), 0.4))
    recovered = decode(grid[:1] * 20 - 10, grid[1:], 3.2)
    torch.testing.assert_close(recovered, torch.tensor([[0.02, 0.02, 0.02]]))
    position = torch.tensor([0.5, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 0.7071068, 0.7071068])
    inputs = model_input(torch.empty((0, 3)), position, q, recovered, 3.2, 64)
    history = inputs[0, 4:8]
    local = decode(history[:1] * 20 - 10, history[1:], 3.2)
    torch.testing.assert_close(local + position, recovered)
    opposite = model_input(torch.empty((0, 3)), position, -q, recovered, 3.2, 64)
    torch.testing.assert_close(inputs, opposite)


def write_run(root, name, terrain, episodes):
    run = root / name
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"point_frame": "world", "quaternion_order": "xyzw", "arguments": {"map_size": 3.2}})
    )
    (run / "terrain.ply").write_bytes(terrain)
    for episode_id, count in enumerate(episodes):
        folder = run / "env_000" / f"episode_{episode_id:06d}"
        folder.mkdir(parents=True)
        for step in range(count):
            np.savez(
                folder / f"frame_{step:06d}.npz",
                lidar_points_w=[[0, 0, 0]],
                gt_points_w=[[0, 0, 0]],
                base_pos_w=[0, 0, 0],
                base_quat_xyzw=[0, 0, 0, 1],
                env_id=0,
                episode_id=episode_id,
                step=step + 1,
                time_s=(step + 1) * 0.1,
            )


def test_split_groups_identical_terrain_and_windows_never_cross_reset(tmp_path):
    write_run(tmp_path, "a", b"same terrain", [3, 1])
    write_run(tmp_path, "b", b"same terrain", [2])
    write_run(tmp_path, "c", b"other terrain", [2])
    runs = discover_runs(tmp_path, 3.2)
    split = split_runs(runs, ["a"], seed=42)
    assert [r["name"] for r in split["validation"]] == ["a", "b"]
    assert [r["name"] for r in split["train"]] == ["c"]
    sequences = SequenceDataset([runs[0]], 2)
    assert [len(sequences[i]) for i in range(len(sequences))] == [2, 1, 1]
    assert [sequences[i][0]["episode_id"] for i in range(len(sequences))] == [0, 0, 1]
    with pytest.raises(ValueError, match="cross-reset"):
        check_successor(sequences[0][0], sequences[2][0])
    with pytest.raises(ValueError, match="coverage"):
        discover_runs(tmp_path, 4)
    with pytest.raises(ValueError, match="two different"):
        split_runs(runs[:2], None, 42)


def test_prediction_does_not_use_gt_and_loss_backpropagates(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(1)
    write_run(tmp_path, "a", b"terrain", [1])
    path = next(tmp_path.glob("a/env_*/episode_*/*.npz"))
    frame = load_frame(path)
    model = TerrainUNet(4)
    config = {
        "map_size": 3.2,
        "resolution": 16,
        "voxel_size": 0.2,
        "offset_weight": 1.0,
        "threshold": 0.05,
        "max_points": 100,
        "point_dropout": 0,
        "noise_std": 0,
    }
    empty = torch.empty((0, 3))
    loss, prediction, _, _ = forward_frame(model, frame, empty, config, torch.device("cpu"))
    different_gt = {**frame, "gt_points_w": np.array([[0.8, 0.8, 0.8]], dtype=np.float32)}
    _, other_prediction, _, _ = forward_frame(model, different_gt, empty, config, torch.device("cpu"))
    torch.testing.assert_close(prediction, other_prediction)
    assert not prediction.requires_grad
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0


def test_paper_augmentation_keeps_ground_truth_in_true_local_frame(tmp_path):
    torch.set_num_threads(2)
    write_run(tmp_path, "a", b"terrain", [1])
    frame = load_frame(next(tmp_path.glob("a/env_*/episode_*/*.npz")))
    frame["base_pos_w"] = np.array([0.25, -0.5, 0.1], dtype=np.float32)
    original_gt = frame["gt_points_w"].copy()
    config = {
        "map_size": 3.2,
        "resolution": 16,
        "offset_weight": 1.0,
        "threshold": 0.5,
        "max_points": 100,
        "augmentation": "paper",
    }
    model = TerrainUNet(4)
    _, _, _, target = forward_frame(model, frame, torch.empty((0, 3)), config, torch.device("cpu"), augment=True)
    from ..training.geometry.voxel import voxelize

    expected = voxelize(torch.from_numpy(original_gt - frame["base_pos_w"]), 3.2, 16)
    torch.testing.assert_close(target, expected)
    np.testing.assert_array_equal(frame["gt_points_w"], original_gt)
    # A fixed reflection must act on the complete sequence, and be reversible.
    mirrored = mirror_sequence([frame, frame], [-1, 1, 1])
    restored = mirror_sequence(mirrored, [-1, 1, 1])
    for key in ("gt_points_w", "lidar_points_w", "base_pos_w", "base_quat_xyzw"):
        np.testing.assert_array_equal(restored[0][key], frame[key])
        np.testing.assert_array_equal(mirrored[0][key], mirrored[1][key])


@pytest.mark.parametrize("sampling", ["random", "sequential"])
def test_recurrent_windows_are_contiguous_with_independent_episode_clocks(tmp_path, sampling):
    write_run(tmp_path, "a", b"terrain", [17, 9])
    runs = discover_runs(tmp_path, 3.2)
    dataset = SequenceDataset(runs, 4, sampling, "plane")
    for epoch in range(3):
        dataset.epoch = epoch
        for index in range(len(dataset)):
            frames = dataset[index]
            assert len({(f["env_id"], f["episode_id"]) for f in frames}) == 1
            assert frames[0]["sequence_time_s"] == 0
            assert [f["sequence_step"] for f in frames] == list(range(len(frames)))
            assert all(b["step"] == a["step"] + 1 for a, b in zip(frames[:-1], frames[1:], strict=True))


def test_missing_recording_is_rejected_instead_of_joining_discontinuous_frames(tmp_path):
    write_run(tmp_path, "a", b"terrain", [4])
    next(tmp_path.glob("a/env_*/episode_*/frame_000001.npz")).unlink()
    dataset = SequenceDataset(discover_runs(tmp_path, 3.2), 4, "random", "plane")
    with pytest.raises(ValueError, match="Non-contiguous"):
        dataset[0]
