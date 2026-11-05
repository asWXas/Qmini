"""Shared plane prior and continuous-window startup checks."""

import numpy as np
import pytest
import torch

from ..training.datasets.sequence import SequenceDataset, discover_runs
from ..training.geometry.temporal import plane_voxels
from .test_training import write_run


def plane_config():
    return dict(map_size=3.2, resolution=16, plane_z=-0.38)


def ordered(points):
    """Sparse coordinate insertion order can differ between single/batched forwards."""
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


def test_world_axis_plane_surface_voxels():
    cells, offsets = plane_voxels(plane_config(), torch.zeros(3))
    local_plane = (cells + offsets) * 0.2 - 1.6
    assert len(cells) == 16**2 and cells[:, 2].unique().numel() == 1
    torch.testing.assert_close(local_plane[:, 2], torch.full((16**2,), -0.38))
    assert local_plane[:, :2].min() == pytest.approx(-1.5)
    assert local_plane[:, :2].max() == pytest.approx(1.5)


@pytest.mark.parametrize("height", [float("nan"), float("inf"), -1.61, 1.6])
def test_plane_rejects_invalid_height(height):
    with pytest.raises(ValueError, match="plane_z"):
        plane_voxels(dict(plane_config(), plane_z=height), torch.zeros(3))


def test_plane_windows_need_no_previous_frame_or_gt_seed(tmp_path):
    write_run(tmp_path, "a", b"terrain", [7, 1])
    runs = discover_runs(tmp_path, 3.2)
    for sampling in ("sequential", "random"):
        dataset = SequenceDataset(runs, 2, sampling, "plane")
        assert len(dataset) == 5
        assert all(first == 0 for _, _, first in dataset.sources)
        for sequence in dataset:
            assert sequence[0]["sequence_time_s"] == 0 and sequence[0]["sequence_step"] == 0
            assert "initial_history_w" not in sequence[0] and "initial_observation" not in sequence[0]
