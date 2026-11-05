"""Canvas support, evidence provenance, motion, recurrence and loss regression checks."""

import numpy as np
import pytest
import torch

from tools.lidar_collection.training.geometry.sparse_voxel import (
    SparsePrediction,
    sparse_reconstruction_loss,
)
from tools.lidar_collection.training.models.architectures import RECURRENT_ARCHITECTURE, make_model
from tools.lidar_collection.training.runtime.observation_memory import (
    ObservedSurfaceMemory,
    preserve_observed_surface,
    validate_memory_config,
)
from tools.lidar_collection.training.runtime.recurrent import recorded_batch, relative_batch
from tools.lidar_collection.training.runtime.state import RecurrentState


def config(**kwargs):
    return dict(
        architecture=RECURRENT_ARCHITECTURE,
        base_channels=4,
        resolution=16,
        map_size=3.2,
        voxel_size=0.2,
        map_frame="world",
        threshold=0.5,
        max_points=0,
        relative_pose=False,
        history_init="plane",
        plane_z=-0.38,
        augmentation="none",
        offset_weight=1.0,
        canvas=True,
        canvas_mode="plane",
        observation_memory=True,
        **kwargs,
    )


def ordered(points):
    points = torch.as_tensor(points)
    # Input clouds in these checks have distinct XYZ; lexicographic ordering is sufficient.
    return points[points[:, 0].argsort()]


def test_memory_survives_subvoxel_motion_exit_return_and_respects_bounds():
    cfg = config()
    memory = ObservedSurfaceMemory()
    point = torch.tensor([[0.11, 0.17, -0.39]])
    empty = point[:0]
    identity = torch.eye(4)
    memory.update(point, identity, cfg)
    move = identity.clone()
    move[0, 3] = -0.003
    for _ in range(100):
        actual = memory.update(empty, move, cfg)
    torch.testing.assert_close(actual, point - torch.tensor([0.3, 0.0, 0.0]), atol=2e-6, rtol=0)
    move[0, 3] = -3
    assert len(memory.update(empty, move, cfg)) == 0
    assert len(memory.points) == 1
    move[0, 3] = 3.3
    torch.testing.assert_close(memory.update(empty, move, cfg), point, atol=2e-6, rtol=0)
    move[0, 3] = -20
    assert len(memory.update(empty, move, cfg)) == len(memory.points) == 0
    capped = dict(cfg, memory_max_voxels=1)
    memory = ObservedSurfaceMemory()
    memory.update(point, identity, capped)
    newer = point + torch.tensor([0.5, 0.0, 0.0])
    torch.testing.assert_close(memory.update(newer, identity, capped), newer)


def test_memory_se3_and_measured_centroid_priority():
    cfg = config()
    memory = ObservedSurfaceMemory()
    point = torch.tensor([[0.11, 0.17, -0.39]])
    memory.update(point, torch.eye(4), cfg)
    t = torch.eye(4)
    t[:3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t[0, 3] = 0.3
    actual = memory.update(point[:0], t, cfg)
    torch.testing.assert_close(actual, point @ t[:3, :3].T + t[:3, 3])
    predicted = torch.tensor([[0.15, 0.19, -0.35], [0.73, 0.13, 0.2]])
    merged = preserve_observed_surface(predicted, point, cfg)
    torch.testing.assert_close(ordered(merged), ordered(torch.cat((point, predicted[1:]))))
    with pytest.raises(ValueError, match="max_points"):
        validate_memory_config(dict(cfg, max_points=10))
    with pytest.raises(ValueError, match="memory_size"):
        validate_memory_config(dict(cfg, memory_size=1))


def test_collapse_reads_all_three_sources_and_returns_zero_source():
    ME = pytest.importorskip("MinkowskiEngine")
    from tools.lidar_collection.training.models.v1 import TemporalCollapse

    torch.set_num_threads(2)
    merge = TemporalCollapse(1, source_planes=3)
    with torch.no_grad():
        merge.conv.kernel.fill_(1)
        merge.conv.bias.zero_()
    features = torch.tensor([[1.0], [10.0], [100.0]], requires_grad=True)
    coords = torch.tensor([[0, 8, 8, 8, source] for source in range(3)], dtype=torch.int32)
    result = merge(ME.SparseTensor(features=features, coordinates=coords))
    torch.testing.assert_close(result.F, torch.tensor([[111.0]]))
    assert result.C.tolist() == [[0, 8, 8, 8, 0]]
    result.F.sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features))


@pytest.mark.parametrize("mode,count", [("plane", 16**2), ("volume", 16**3)])
def test_canvas_returns_each_frame_and_never_becomes_measured_truth(mode, count):
    pytest.importorskip("MinkowskiEngine")
    torch.set_num_threads(2)
    cfg = dict(config(), canvas_mode=mode)
    model = make_model(cfg).eval()
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    state = RecurrentState()
    empty = torch.empty(0, 3)
    with torch.no_grad():
        # Kill every coarse branch and all final outputs. Fine support must still survive.
        for head in model.classifiers:
            head.kernel.zero_()
            head.bias.fill_(-10)
        for _ in range(2):
            outputs, predictions = relative_batch(model, [empty], [torch.eye(4)], [state], cfg)
            assert len(predictions[0]) == 0
            assert len(state.retained_observations) == len(state.observation_memory.points) == 0
            assert len(outputs[0].coordinates) >= count
            if mode == "volume":
                assert len(outputs[0].coordinates) == 16**3
        # A positive residual can retain positions beyond the initial horizontal plane.
        model.classifiers[-1].bias.fill_(3)
        _, predictions = relative_batch(model, [empty], [torch.eye(4)], [state], cfg)
        assert len(predictions[0]) >= count
    hook.remove()
    assert all(int((coords[:, 4] == 2).sum()) == count for coords in seen)
    assert len(state.observation_memory.points) == 0


def test_memory_isolated_across_sequences_even_if_network_deletes_everything():
    pytest.importorskip("MinkowskiEngine")
    torch.set_num_threads(2)
    cfg = config()
    model = make_model(cfg).eval()
    with torch.no_grad():
        model.classifiers[-1].bias.fill_(-10)
        states = [RecurrentState(), RecurrentState()]
        points = [torch.tensor([[x, 0.13, -0.4]]) for x in (-0.7, 0.7)]
        relative_batch(model, points, [torch.eye(4)] * 2, states, cfg)
        _, predictions = relative_batch(model, [points[0][:0]] * 2, [torch.eye(4)] * 2, states[::-1], cfg)
        for predicted, measured in zip(predictions, points[::-1], strict=True):
            torch.testing.assert_close(predicted, measured)
        assert all(len(state.network_prediction) == 0 for state in states)
        fresh = RecurrentState()
        _, predictions = relative_batch(model, [points[0][:0]], [torch.eye(4)], [fresh], cfg)
        assert len(predictions[0]) == 0 and fresh.observation_memory.frame == 1


@pytest.mark.parametrize("memory_enabled", [False, True])
def test_canvas_training_gradients_and_gt_does_not_enter_evidence(memory_enabled):
    pytest.importorskip("MinkowskiEngine")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = dict(config(), canvas_mode="volume", occupancy_loss="balanced", observation_memory=memory_enabled)
    model = make_model(cfg).train()
    with torch.no_grad():
        model.classifiers[-1].kernel.normal_(0, 0.1)
    frame = dict(
        lidar_points_w=np.array([[0.13, 0.17, -0.37]], dtype=np.float32),
        gt_points_w=np.array([[0.13, 0.17, -0.37], [-0.7, 0.2, 0.4]], dtype=np.float32),
        base_pos_w=np.zeros(3, dtype=np.float32),
        base_quat_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
    )
    state = RecurrentState()
    recorded_batch(model, [frame], [state], cfg, "cpu", augment=False)
    hidden = [m.features for m in state.hidden]
    for h in hidden:
        h.retain_grad()
    loss, _, _ = recorded_batch(model, [frame], [state], cfg, "cpu", augment=False)
    loss[0].backward()
    assert all(h.grad is not None and h.grad.abs().sum() > 0 for h in hidden)
    if memory_enabled:
        torch.testing.assert_close(state.retained_observations, torch.from_numpy(frame["lidar_points_w"]))
    else:
        assert state.retained_observations is None and state.observation_memory.frame == 0
        assert state.prediction is state.network_prediction
    model.eval()
    with torch.no_grad():
        a, b = RecurrentState(), RecurrentState()
        recorded_batch(model, [frame], [a], cfg, "cpu", augment=False)
        recorded_batch(model, [dict(frame, gt_points_w=frame["gt_points_w"] + 1)], [b], cfg, "cpu", augment=False)
        torch.testing.assert_close(a.prediction, b.prediction)
        torch.testing.assert_close(a.retained_observations, b.retained_observations)


@pytest.mark.parametrize("canvas,mode", [(False, "plane"), (True, "plane"), (True, "volume")])
def test_network_alone_controls_output_and_feedback_without_measurement_copy(monkeypatch, canvas, mode):
    pytest.importorskip("MinkowskiEngine")
    from tools.lidar_collection.training.runtime import recurrent

    def forbidden(*args, **kwargs):
        raise AssertionError("Network-only inference must never update/fuse observation memory")

    monkeypatch.setattr(ObservedSurfaceMemory, "update", forbidden)
    monkeypatch.setattr(recurrent, "preserve_observed_surface", forbidden)
    torch.set_num_threads(2)
    cfg = dict(config(), canvas=canvas, canvas_mode=mode, observation_memory=False)
    model = make_model(cfg).eval()
    captured_prior = []
    hook = model.register_forward_pre_hook(
        lambda module, args: captured_prior.append(args[0][args[0][:, 4] == 1].clone())
    )
    state = RecurrentState()
    point = torch.tensor([[0.13, 0.17, -0.37]])
    with torch.no_grad():
        # Force the network to reject even observed/initial-plane cells.
        model.classifiers[-1].kernel.zero_()
        model.classifiers[-1].bias.fill_(-10)
        for _ in range(2):
            _, predictions = relative_batch(model, [point], [torch.eye(4)], [state], cfg)
            assert predictions[0] is state.network_prediction is state.prediction
            assert len(predictions[0]) == 0
            assert len(state.hidden) == 5
            assert state.retained_observations is None
            assert state.observation_memory.points is None and state.observation_memory.frame == 0
    hook.remove()
    assert len(captured_prior[0]) == cfg["resolution"] ** 2  # Explicit first-frame plane prior.
    assert len(captured_prior[1]) == 0  # No plane or sensor points silently copied into feedback.


def test_balanced_loss_is_not_dominated_by_empty_volume():
    coords = torch.tensor([[0, x, 0, 0, 0] for x in range(10)])
    logits = torch.full((10, 1), -2.0, requires_grad=True)
    offsets = torch.full((10, 3), 0.5)
    output = SparsePrediction([(coords, logits, 1)], coords, logits, offsets)
    loss = sparse_reconstruction_loss(output, coords[:1, 1:4], offsets[:1], 16, occupancy_loss="balanced")
    loss.backward()
    # Summed positive gradient outweighs nine negative cells at this negative initialization.
    assert -logits.grad[0].item() > logits.grad[1:].sum().item()
    for target in (coords[:0, 1:4], coords[:, 1:4]):
        loss = sparse_reconstruction_loss(output, target, offsets[: len(target)], 16, occupancy_loss="balanced")
        assert torch.isfinite(loss)


def test_existing_checkpoint_memory_override_relative_api_and_reset(tmp_path):
    pytest.importorskip("MinkowskiEngine")
    from tools.lidar_collection.training.predictor import Reconstructor, RelativeReconstructor

    torch.set_num_threads(2)
    cfg = dict(config(), canvas=False, observation_memory=False)
    model = make_model(cfg).eval()
    with torch.no_grad():
        model.classifiers[-1].bias.fill_(-10)
    checkpoint = tmp_path / "old.pt"
    torch.save(dict(config=cfg, model=model.state_dict()), checkpoint)
    relative = RelativeReconstructor(checkpoint, observation_memory=True)
    absolute = Reconstructor(checkpoint, observation_memory=True)
    point = np.array([[0.11, 0.17, -0.39]], dtype=np.float32)
    np.testing.assert_allclose(relative.update(point), point, atol=1e-6)
    np.testing.assert_allclose(absolute.update(point, [0, 0, 0], [0, 0, 0, 1]), point, atol=1e-6)
    t = np.eye(4, dtype=np.float32)
    t[0, 3] = -0.3
    next_relative = relative.update(point[:0], t)
    next_world = absolute.update(point[:0], [0.3, 0, 0], [0, 0, 0, 1])
    np.testing.assert_allclose(next_relative + [0.3, 0, 0], next_world, atol=1e-6)
    relative.reset()
    absolute.reset()
    assert relative.state.observation_memory.points is None
    assert absolute.recurrent_state.observation_memory.points is None
    assert len(relative.update(point[:0])) == 0


def test_memory_cache_uses_perturbed_observation_not_clean_sensor():
    pytest.importorskip("MinkowskiEngine")
    from tools.lidar_collection.training.geometry.sparse_voxel import sparse_voxelize

    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = config()
    model = make_model(cfg).eval()
    captured = []

    def capture(module, args):
        coords, features = args[:2]
        current = coords[:, 4] == 0
        captured.append((coords[current, 1:4] + features[current]) * 0.2 - 1.6)

    hook = model.register_forward_pre_hook(capture)
    points = torch.tensor([[0.11, 0.17, -0.39], [0.45, 0.45, -0.39], [0.9, 0.9, -0.39]])
    state = RecurrentState()
    with torch.no_grad():
        relative_batch(
            model, [points], [torch.eye(4)], [state], cfg, noise=dict(voxel_dropout=0.2, voxel_noise_std=0.1)
        )
    hook.remove()
    torch.testing.assert_close(ordered(state.retained_observations), ordered(captured[0]))
    clean_offsets = sparse_voxelize(points, 3.2, 16)[1]
    assert len(captured[0]) < len(points) or not torch.allclose(sparse_voxelize(captured[0], 3.2, 16)[1], clean_offsets)
