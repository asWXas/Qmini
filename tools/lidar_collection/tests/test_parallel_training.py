"""Unequal/empty rank batches, sequence initialization, and observation-only noise."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..training.datasets.augmentation import mirror_sequence
from ..training.datasets.sequence import SequenceDataset, discover_runs, load_frame
from ..training.runtime.distributed import ParallelBatchSampler, merge_validation, synchronize_gradients, validation_shards
from ..training.geometry.sparse_voxel import temporal_input
from .test_training import write_run


@pytest.mark.parametrize("size", [1, 5, 11])
def test_rank_sampler_no_duplicates_or_drops(size):
    ranks = [list(ParallelBatchSampler(size, 2, rank, 2, 42)) for rank in range(2)]
    assert len(ranks[0]) == len(ranks[1])
    samples = [index for rank in ranks for batch in rank for index in batch]
    assert sorted(samples) == list(range(size))


def gradient_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    model = torch.nn.Linear(2, 1, bias=False)
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
    with torch.no_grad():
        model.weight.fill_(0.3)
    x = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, -1.0], [3.0, 2.0]])
    for sizes in ((3, 1), (4, 0)):
        model.zero_grad(set_to_none=True)
        start = 0 if rank == 0 else sizes[0]
        local = x[start : start + sizes[rank]]
        if len(local):
            (model(local).square().sum() / len(x)).backward()
        synchronize_gradients(model, bucket_bytes=4)
        reference = model.weight.detach().clone().requires_grad_()
        (torch.nn.functional.linear(x, reference).square().mean()).backward()
        torch.testing.assert_close(model.weight.grad, reference.grad)
        assert model.unused.grad is None
    dist.destroy_process_group()


def test_gradient_sum_equals_global_mean_with_unequal_and_empty_ranks(tmp_path):
    mp.spawn(gradient_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def test_random_slices_previous_gt_once_clock_zero_and_mirror(tmp_path):
    write_run(tmp_path, "a", b"terrain", [30, 1])
    for path in tmp_path.glob("a/env_*/episode_*/*.npz"):
        with np.load(path) as data:
            frame = dict(data)
        frame["gt_points_w"] = np.array([[float(frame["step"]), 1, 2]])
        np.savez(path, **frame)
    runs = discover_runs(tmp_path, 3.2)
    dataset = SequenceDataset(runs, 4, "random", "previous_gt", seed=42)
    previous_starts = []
    for epoch in range(3):
        dataset.epoch = epoch
        sequence = dataset[0]
        assert sequence[0]["sequence_time_s"] == 0
        assert [f["sequence_step"] for f in sequence] == list(range(4))
        initial = load_frame(sequence[0]["initial_history_path"])
        assert initial["step"] + 1 == sequence[0]["step"]
        np.testing.assert_array_equal(sequence[0]["initial_history_w"], initial["gt_points_w"])
        assert all("initial_history_w" not in frame for frame in sequence[1:])
        assert [f["path"] for f in dataset[0]] == [f["path"] for f in sequence]
        mirrored = mirror_sequence(sequence, [-1, 1, 1])
        np.testing.assert_array_equal(mirrored[0]["initial_history_w"], initial["gt_points_w"] * [-1, 1, 1])
        previous_starts.append(sequence[0]["path"])
    assert len(set(previous_starts)) > 1
    singleton = dataset[len(dataset) - 1]
    assert len(singleton) == 1 and "initial_history_w" not in singleton[0]
    assert Path(singleton[0]["path"]).parent.name == "episode_000001"


def test_voxel_corruption_only_touches_current_observation():
    observed = torch.tensor([[0.01, 0.01, 0.01], [0.11, 0.01, 0.01]])
    history = observed.clone()
    clean_coords, clean_features = temporal_input(observed, history, 3.2, 64)
    coords, features = temporal_input(observed, history, 3.2, 64, {"voxel_noise_std": 0.2})
    torch.testing.assert_close(coords, clean_coords)
    torch.testing.assert_close(features[coords[:, 4] == 1], clean_features[clean_coords[:, 4] == 1])
    assert not torch.equal(features[coords[:, 4] == 0], clean_features[clean_coords[:, 4] == 0])
    assert ((features >= 0) & (features < 1)).all()
    dropped, _ = temporal_input(observed, history, 3.2, 64, {"voxel_dropout": 1.0})
    assert (dropped[:, 4] == 1).all() and len(dropped) == len(history)
    clean_again = temporal_input(observed, history, 3.2, 64)
    torch.testing.assert_close(clean_again[1], clean_features)


def test_validation_shards_and_weighted_metrics():
    runs = [{"name": "a", "episodes": [{"frames": list(range(4))}, {"frames": list(range(3))}]}]
    shards = validation_shards(runs, 2)
    assert [sum(len(ep["frames"]) for run in shard for ep in run["episodes"]) for shard in shards] == [4, 3]
    limited = validation_shards(runs, 2, 5)
    assert sum(len(ep["frames"]) for shard in limited for run in shard for ep in run["episodes"]) == 5
    results = []
    for n, tp, fp, fn, error in [(4, 2, 1, 3, 1.5), (3, 1, 3, 0, 3.0)]:
        metrics = dict(tp=tp, fp=fp, fn=fn, matched_centroid_error_cm=error)
        results.append(dict(frames=n, episodes_started=1, loss=2.0, prediction=metrics, observed_only=metrics))
    merged = merge_validation(results, 7)
    assert not merged["partial_evaluation"]
    assert merged["prediction"]["f1"] == pytest.approx(6 / 13)
    assert merged["prediction"]["matched_centroid_error_cm"] == 2.0


def test_sparse_minibatch_matches_independent_eval_and_preserves_empty_sample(tmp_path):
    pytest.importorskip("MinkowskiEngine")
    from ..training.models.architectures import SPARSE_ARCHITECTURE, make_model
    from ..training.runtime.batched import forward_sparse_batch
    from ..training.runtime.engine import forward_frame

    torch.set_num_threads(2)
    write_run(tmp_path, "a", b"terrain", [2])
    frames = [load_frame(path) for path in sorted(tmp_path.glob("a/env_*/episode_*/*.npz"))]
    frames[1]["lidar_points_w"] = np.array([[0.6, -0.4, 0.1]], dtype=np.float32)
    config = dict(
        architecture=SPARSE_ARCHITECTURE,
        base_channels=4,
        map_size=3.2,
        resolution=16,
        threshold=0.5,
        max_points=0,
        offset_weight=1.0,
        augmentation="none",
    )
    model = make_model(config).eval()
    histories = [torch.empty(0, 3) for _ in frames]
    with torch.no_grad():
        losses, predictions = forward_sparse_batch(model, frames, histories, config, torch.device("cpu"), augment=False)
        for frame, history, loss, prediction in zip(frames, histories, losses, predictions, strict=True):
            expected_loss, expected, _, _ = forward_frame(model, frame, history, config, torch.device("cpu"))
            torch.testing.assert_close(loss, expected_loss)

            def ordered(points):
                array = points.numpy()
                cells = np.floor((array + 1.6) / 0.2).astype(int)
                return array[np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))]

            np.testing.assert_allclose(ordered(prediction), ordered(expected), atol=1e-5)
        frames[0]["lidar_points_w"] = np.empty((0, 3), dtype=np.float32)
        _, predictions = forward_sparse_batch(model, frames, histories, config, torch.device("cpu"), augment=False)
        assert len(predictions[0]) == 0 and len(predictions[1]) == 16**3
    model.train()
    losses, _ = forward_sparse_batch(model, frames, histories, config, torch.device("cpu"), augment=False)
    sum(losses).backward()
    assert model.offset_head.kernel.grad.abs().sum() > 0
