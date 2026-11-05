"""Coordinate semantics, sparse supervision, actual 4D operators and no-label feedback."""

import pytest
import torch

from ..training.models.architectures import SPARSE_ARCHITECTURE, make_model
from ..training.runtime.inference import run_network
from ..training.geometry.sparse_voxel import (
    SparsePrediction,
    decode_sparse,
    match_target,
    sparse_reconstruction_loss,
    sparse_voxelize,
    temporal_input,
)
from ..training.geometry.voxel import voxelize


def test_sparse_centroids_match_dense_and_time_is_a_coordinate():
    points = torch.tensor([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [1.6, 0, 0]])
    cells, features = sparse_voxelize(points, 3.2, 64)
    dense = voxelize(points, 3.2, 64)
    assert cells.tolist() == [[32, 32, 32]]
    torch.testing.assert_close(features[0], dense[1:, 32, 32, 32])
    coordinates, combined = temporal_input(points, points, 3.2, 64)
    assert coordinates.shape == (2, 5)
    assert coordinates[:, 4].tolist() == [0, 1]
    assert combined.shape == (2, 3)  # NOT a dense tensor with stacked time feature channels
    torch.testing.assert_close(combined[0], combined[1])


def test_multiscale_target_matching_and_final_probability_pruning():
    coords = torch.tensor([[0, 0, 0, 0, 0], [0, 8, 8, 8, 0]], dtype=torch.int32)
    target = torch.tensor([[3, 2, 1]], dtype=torch.int32)
    labels, _ = match_target(coords, target, 16, stride=8)
    assert labels.tolist() == [True, False]
    final = torch.tensor([[0, 3, 2, 1, 0], [0, 8, 8, 8, 0]], dtype=torch.int32)
    logits = torch.tensor([[2.0], [-2.0]], requires_grad=True)
    offsets = torch.full((2, 3), 0.25, requires_grad=True)
    output = SparsePrediction([(coords, logits, 8), (final, logits, 1)], final, logits, offsets)
    loss = sparse_reconstruction_loss(output, target, torch.full((1, 3), 0.5), 16)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and offsets.grad[0].abs().sum() > 0
    assert offsets.grad[1].abs().sum() == 0
    decoded = decode_sparse(output, 3.2, 16, 0.5)
    assert decoded.shape == (1, 3)


def devices():
    return ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", devices())
def test_minkowski_time_collapse_and_sparse_generation(device):
    ME = pytest.importorskip("MinkowskiEngine")
    from ..training.models.v1 import SparseSceneNet4D, TemporalCollapse

    torch.set_num_threads(2)
    coordinates = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=torch.int32, device=device)
    features = torch.tensor([[2.0], [5.0]], device=device)
    collapse = TemporalCollapse(1).to(device)
    with torch.no_grad():
        collapse.conv.kernel.fill_(1)
        collapse.conv.bias.zero_()
    collapsed = collapse(ME.SparseTensor(features=features, coordinates=coordinates))
    assert len(collapsed) == 1 and collapsed.C[:, 4].tolist() == [0]
    torch.testing.assert_close(collapsed.F, torch.tensor([[7.0]], device=device))

    model = SparseSceneNet4D(4, 16, 0.5).to(device)
    xyz = torch.tensor([[0.01, 0.01, 0.01], [0.2, 0.1, -0.4]], device=device)
    coordinates, features = temporal_input(xyz, xyz + 0.1, 3.2, 16)
    output = model(coordinates, features)
    assert [stage[2] for stage in output.stages] == [8, 4, 2, 1]
    # Initial classifier biases keep every candidate: four upsamplings must
    # generate the complete 16^3 volume from the 1^3 coarsest lattice.
    assert len(output.coordinates) == 16**3
    assert bool((output.coordinates[:, 4] == 0).all())
    cells, target_features = sparse_voxelize(xyz, 3.2, 16)
    loss = sparse_reconstruction_loss(output, cells, target_features, 16)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.stem[0].kernel.grad.abs().sum() > 0
    assert model.offset_head.kernel.grad.abs().sum() > 0


@pytest.mark.parametrize("device", devices())
def test_sparse_empty_input_and_empty_pruning_are_valid(device):
    pytest.importorskip("MinkowskiEngine")
    config = {
        "architecture": SPARSE_ARCHITECTURE,
        "base_channels": 4,
        "resolution": 16,
        "map_size": 3.2,
        "threshold": 0.5,
        "max_points": 0,
    }
    model = make_model(config).to(device).eval()
    position = torch.zeros(3, device=device)
    quaternion = torch.tensor([0, 0, 0, 1.0], device=device)
    empty = torch.empty((0, 3), device=device)
    points, output, _ = run_network(model, empty, position, quaternion, empty, config)
    assert points.shape == (0, 3)
    loss = sparse_reconstruction_loss(output, torch.empty((0, 3), dtype=torch.int32, device=device), empty, 16)
    loss.backward()
    with torch.no_grad():
        for classifier in model.classifiers:
            classifier.bias.fill_(-100)
    # Every pruning decision is negative, including after singleton observations.
    points, _, _ = run_network(
        model, torch.tensor([[0.1, 0.2, 0.3]], device=device), position, quaternion, empty, config
    )
    assert points.shape == (0, 3)
