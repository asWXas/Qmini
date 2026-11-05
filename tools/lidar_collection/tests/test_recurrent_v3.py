"""V3 relative geometry, persistent spatial memory, temporal gradients and unrestricted edits."""

import numpy as np
import pytest
import torch

from ..training.models.architectures import RECURRENT_ARCHITECTURE, make_model
from ..training.runtime.recurrent import recorded_batch, relative_batch
from ..training.runtime.state import FeatureMemory, RecurrentState, coverage_mask, warp_features
from ..training.geometry.temporal import rotation_matrix
from .test_plane_startup import ordered
from .test_training import write_run


def config():
    return dict(
        architecture=RECURRENT_ARCHITECTURE,
        base_channels=4,
        resolution=16,
        map_size=3.2,
        voxel_size=0.2,
        gt_map_size=3.2,
        map_frame="world",
        threshold=0.5,
        max_points=0,
        relative_pose=False,
        history_init="plane",
        plane_z=-0.38,
        augmentation="none",
        offset_weight=1.0,
    )


def test_feature_rotation_translation_collision_crop_and_gradients():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    coords = torch.tensor([[0, 8, 8, 8], [0, 9, 8, 8]], dtype=torch.int32)
    memory = FeatureMemory(coords, values, 1)
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform[0, 3] = 0.2
    moved = warp_features(memory, transform, 3.2, 16)
    assert moved.coordinates.tolist() == [[0, 8, 8, 8], [0, 8, 9, 8]]
    torch.testing.assert_close(moved.features, values)
    moved.features.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    transform[0, 3] = 10
    assert len(warp_features(memory, transform, 3.2, 16).coordinates) == 0
    # A 45-degree rotation can send multiple feature cells into one target cell.
    angle = torch.tensor(torch.pi / 4)
    transform = torch.eye(4)
    transform[:2, :2] = torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
    many = torch.tensor([[0, x, y, 8] for x in range(6, 10) for y in range(6, 10)], dtype=torch.int32)
    memory = FeatureMemory(many, torch.ones(len(many), 2), 1)
    moved = warp_features(memory, transform, 3.2, 16)
    assert len(moved.coordinates) < len(many)
    torch.testing.assert_close(moved.features, torch.ones_like(moved.features))


def test_rotated_gt_coverage_is_not_a_prediction_mask():
    cfg = dict(config(), map_frame="body")
    cells = torch.tensor([[0, 0, 0], [8, 8, 8], [15, 15, 15]], dtype=torch.int32)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(coverage_mask(cells, 1, cfg, identity), torch.ones(3, dtype=torch.bool))
    q = torch.tensor([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)], dtype=torch.float32)
    assert coverage_mask(cells, 1, cfg, q).tolist() == [False, True, False]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_plane_startup_and_unrestricted_add_delete(device):
    pytest.importorskip("MinkowskiEngine")
    torch.set_num_threads(2)
    cfg = config()
    model = make_model(cfg).to(device).eval()
    empty = torch.empty(0, 3, device=device)
    transform = torch.eye(4, device=device)
    with torch.no_grad():
        _, predictions = relative_batch(model, [empty], [transform], [RecurrentState()], cfg)
        assert len(predictions[0]) == 16**2
        torch.testing.assert_close(predictions[0][:, 2], torch.full((16**2,), -0.38, device=device))
        # No observations at all: removal must still work, without a spatial hard mask.
        model.classifiers[-1].bias.fill_(-3)
        _, predictions = relative_batch(model, [empty], [transform], [RecurrentState()], cfg)
        assert len(predictions[0]) == 0
        model.classifiers[-1].bias.fill_(3)
        _, predictions = relative_batch(
            model, [empty], [transform], [RecurrentState()], dict(cfg, history_init="empty")
        )
        assert len(predictions[0]) == 16**3


def test_hidden_is_real_memory_and_later_loss_reaches_previous_frame():
    pytest.importorskip("MinkowskiEngine")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cfg = config()
    model = make_model(cfg).eval()
    with torch.no_grad():
        model.classifiers[-1].kernel.normal_(0.0, 0.2)
    first = torch.tensor([[0.1, 0.1, 0.1]])
    state = RecurrentState()
    relative_batch(model, [first], [torch.eye(4)], [state], cfg)
    first_hidden = [memory.features for memory in state.hidden]
    for features in first_hidden:
        features.retain_grad()
    # Identical explicit map/current observation; only hidden memory differs.
    without_memory = RecurrentState(
        prediction=state.prediction.clone(), started=True, orientation=state.orientation.clone()
    )
    next_points = torch.tensor([[-0.5, 0.3, 0.1]])
    output, _ = relative_batch(model, [next_points], [torch.eye(4)], [state], cfg)
    other, _ = relative_batch(model, [next_points], [torch.eye(4)], [without_memory], cfg)

    def sorted_logits(prediction):
        c = prediction.coordinates
        keys = (c[:, 1].long() * 16 + c[:, 2]) * 16 + c[:, 3]
        return prediction.logits[keys.argsort()]

    assert not torch.allclose(sorted_logits(output[0]), sorted_logits(other[0]))
    output[0].logits.square().mean().backward()
    assert all(h.grad is not None and h.grad.abs().sum() > 0 for h in first_hidden)
    saved = [h.features.detach().clone() for h in state.hidden]
    state.detach_hidden()
    for previous, memory in zip(saved, state.hidden, strict=True):
        torch.testing.assert_close(previous, memory.features)
        assert memory.features.grad_fn is None
    # Truncation must release the earlier graph while keeping values for the next chunk.
    output, _ = relative_batch(model, [next_points], [torch.eye(4)], [state], cfg)
    output[0].logits.square().mean().backward()


def test_batched_memory_isolation_with_reordered_active_sequences():
    pytest.importorskip("MinkowskiEngine")
    torch.set_num_threads(2)
    cfg = config()
    model = make_model(cfg).eval()
    with torch.no_grad():
        model.classifiers[-1].kernel.normal_(0.0, 0.1)
    states = [RecurrentState(), RecurrentState()]
    independent = [RecurrentState(), RecurrentState()]
    points = [torch.tensor([[x, 0.1, 0.1]]) for x in (-0.7, 0.7)]
    for order in ([0, 1], [1, 0], [1]):
        transforms = [torch.eye(4) for _ in order]
        with torch.no_grad():
            outputs, predictions = relative_batch(
                model, [points[i] for i in order], transforms, [states[i] for i in order], cfg
            )
            for slot, index in enumerate(order):
                single, prediction = relative_batch(
                    model, [points[index]], [transforms[slot]], [independent[index]], cfg
                )
                np.testing.assert_allclose(ordered(predictions[slot]), ordered(prediction[0]), atol=1e-5)
                assert len(states[index].hidden) == 5


def test_relative_deployment_matches_recorded_adapter_reset_and_gt_independence(tmp_path):
    pytest.importorskip("MinkowskiEngine")
    from ..training.predictor import Reconstructor, RelativeReconstructor

    torch.set_num_threads(2)
    cfg = config()
    model = make_model(cfg).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(config=cfg, model=model.state_dict()), checkpoint)
    initial_q = [0.0, 0.0, np.sin(0.3), np.cos(0.3)]
    predictor = RelativeReconstructor(checkpoint, initial_quaternion_xyzw=initial_q)
    world_predictor = Reconstructor(checkpoint)
    state = RecurrentState()
    previous_rotation, previous_position = None, None
    for step in range(3):
        q = torch.tensor([0.0, 0.0, np.sin(0.3 + step * 0.2), np.cos(0.3 + step * 0.2)], dtype=torch.float32)
        rotation = rotation_matrix(q)
        position = torch.tensor([step * 0.1, 0.0, 0.5])
        world_points = torch.tensor([[0.3, 0.2, 0.0], [0.5, 0.1, 0.2]])
        points = (world_points - position) @ rotation
        frame = dict(
            lidar_points_w=world_points.numpy(),
            gt_points_w=world_points.numpy(),
            base_pos_w=position.numpy(),
            base_quat_xyzw=q.numpy(),
        )
        transform = torch.eye(4)
        if step:
            transform[:3, :3] = rotation.T @ previous_rotation
            transform[:3, 3] = rotation.T @ (previous_position - position)
        predicted = predictor.update(points.numpy(), transform.numpy())
        world_predicted = world_predictor.update(world_points.numpy(), position.numpy(), q.numpy())
        np.testing.assert_allclose(ordered(predicted), ordered(world_predicted - position.numpy()), atol=2e-6)
        with torch.no_grad():
            _, recorded, _ = recorded_batch(model, [frame], [state], cfg, "cpu", augment=False)
        np.testing.assert_allclose(ordered(predicted), ordered(recorded[0] - position), atol=2e-6)
        previous_rotation, previous_position = rotation, position
    with pytest.raises(ValueError, match="requires"):
        predictor.update(points.numpy())
    predictor.reset()
    assert not predictor.state.started and not predictor.state.hidden
    a = predictor.update(points.numpy())
    np.testing.assert_allclose(ordered(a), ordered(RelativeReconstructor(checkpoint).update(points.numpy())))
    bad = np.eye(4, dtype=np.float32)
    bad[0, 0] = 2
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        predictor.update(points.numpy(), bad)
    changed = dict(frame, gt_points_w=frame["gt_points_w"] + 1.0)
    with torch.no_grad():
        _, a, _ = recorded_batch(model, [frame], [RecurrentState()], cfg, "cpu", augment=False)
        _, b, _ = recorded_batch(model, [changed], [RecurrentState()], cfg, "cpu", augment=False)
    np.testing.assert_allclose(ordered(a[0]), ordered(b[0]))


def test_recorded_augmentation_and_full_evaluation_export(tmp_path):
    pytest.importorskip("MinkowskiEngine")
    from ..training.datasets.sequence import discover_runs, load_frame
    from ..training.runtime.engine import evaluate
    from ..training.visualize import read_sample

    torch.set_num_threads(2)
    write_run(tmp_path, "a", b"terrain", [2, 1])
    runs = discover_runs(tmp_path, 3.2)
    cfg = config()
    model = make_model(cfg).eval()
    frame = load_frame(runs[0]["episodes"][0]["frames"][0])
    clean_gt = frame["gt_points_w"].copy()
    losses, _, _ = recorded_batch(model, [frame], [RecurrentState()], dict(cfg, augmentation="paper"), "cpu")
    losses[0].backward()
    np.testing.assert_array_equal(frame["gt_points_w"], clean_gt)
    output = tmp_path / "evaluation"
    result = evaluate(model, runs, cfg, torch.device("cpu"), export=output, export_frames=3)
    assert result["frames"] == 3 and result["episodes_started"] == 2
    samples = [read_sample(path, output, cfg) for path in sorted(output.glob("predictions/*/*/*/*.npz"))]
    for key in ("tp", "fp", "fn"):
        assert sum(sample["record"]["prediction"][key] for sample in samples) == result["prediction"][key]


def test_world_axes_keep_hidden_grid_on_turn_and_translate_with_center():
    pytest.importorskip("MinkowskiEngine")
    from ..training.runtime.state import pack_hidden, relative_transform

    torch.set_num_threads(2)
    cfg = config()
    model = make_model(cfg).eval()
    state = RecurrentState()
    points = torch.tensor([[0.13, 0.27, -0.31]])
    captured = []
    hook = model.register_forward_pre_hook(lambda module, args: captured.append(args))
    old_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    position = torch.zeros(3)
    with torch.no_grad():
        relative_batch(model, [points], [torch.eye(4)], [state], cfg)
        for yaw, delta in ((0.7, 0.0), (1.2, 0.4)):
            old_hidden = state.hidden
            q = torch.tensor([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=torch.float32)
            rotation = rotation_matrix(q)
            next_position = position + torch.tensor([delta, 0.0, 0.0])
            transform = relative_transform(next_position, q, position, old_q)
            translation = torch.eye(4)
            translation[:3, 3] = position - next_position
            expected = pack_hidden([state], [translation], 3.2, 16)
            relative_batch(model, [(points - next_position) @ rotation], [transform], [state], cfg)
            torch.testing.assert_close(state.orientation, rotation)
            for actual, wanted in zip(captured[-1][3], expected, strict=True):
                torch.testing.assert_close(actual.coordinates, wanted.coordinates)
                torch.testing.assert_close(actual.features, wanted.features)
            if delta == 0:
                for actual, before in zip(captured[-1][3], old_hidden, strict=True):
                    torch.testing.assert_close(actual.coordinates, before.coordinates)
                cells, features = captured[0][:2]
                new_cells, new_features = captured[-1][:2]
                torch.testing.assert_close(cells[cells[:, -1] == 0], new_cells[new_cells[:, -1] == 0])
                torch.testing.assert_close(features[cells[:, -1] == 0], new_features[new_cells[:, -1] == 0])
            position, old_q = next_position, q
    hook.remove()
    cells = torch.tensor([[0, 0, 0], [15, 15, 15]], dtype=torch.int32)
    assert coverage_mask(cells, 1, cfg, old_q).all()  # turning does not remove GT corners


@pytest.mark.parametrize("enabled", [False, True])
def test_pose_embedding_checkpoint_choice_and_network_input(tmp_path, enabled):
    pytest.importorskip("MinkowskiEngine")
    from ..training.predictor import RelativeReconstructor

    torch.set_num_threads(2)
    cfg = dict(config(), relative_pose=enabled)
    model = make_model(cfg).eval()
    assert (model.pose_encoder is not None) == enabled
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(config=cfg, model=model.state_dict()), checkpoint)
    predictor = RelativeReconstructor(checkpoint)
    seen = []
    hook = predictor.model.register_forward_pre_hook(lambda module, args: seen.append(args[2]))
    predictor.update(np.array([[0.13, 0.27, -0.31]], dtype=np.float32))
    if enabled:
        assert seen[0].shape == (1, 10)
    else:
        assert seen[0] is None
        assert not any("pose_encoder" in key for key in predictor.model.state_dict())
    hook.remove()


def test_removed_v2_is_rejected_and_surviving_models_are_explicit():
    from ..training.models.architectures import ARCHITECTURES

    assert set(ARCHITECTURES) == {"sparse4d_v1", "sparse4d_v3", "dense3d"}
    with pytest.raises(ValueError, match="v2 has been removed"):
        make_model(dict(config(), architecture="minkowski_sparse_gru_residual_v2"))


@pytest.mark.parametrize("threshold", [0.2, 0.8])
def test_v3_zero_residual_preserves_prior_at_nondefault_threshold(threshold):
    from ..training.geometry.residual import residual_update

    coordinates = torch.tensor([[0, 8, 8, 8, 1], [0, 9, 8, 8, 0]], dtype=torch.int32)
    features = torch.tensor([[0.1, 0.3, 0.7], [0.2, 0.4, 0.6]])
    keep, logits, offsets, editable = residual_update(
        coordinates, torch.zeros(2, 1), torch.zeros(2, 3), coordinates, features, 16, threshold=threshold
    )
    assert keep.all() and editable.all()
    assert (logits.sigmoid().flatten() >= threshold).tolist() == [True, False]
    torch.testing.assert_close(offsets[0], features[0])
