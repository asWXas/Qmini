"""Visualization uses evaluation geometry; sparse export must retain temporal inference."""

import numpy as np
import pytest
import torch

from ..training.runtime.engine import Scores, evaluate
from ..training.visualize import cloud, comparison, thin
from ..training.geometry.voxel import voxelize
from .test_recurrent_v3 import config as plane_config
from .test_training import write_run


def test_visualization_metrics_match_evaluator_and_ignore_outside_points():
    config = plane_config()
    position = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    predicted = np.array([[0.21, 0.01, 0.42], [0.22, 0.02, 0.43], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([[0.23, 0.02, 0.44], [-0.8, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=np.float32)
    points, errors = comparison(cloud(predicted, position, config), cloud(target, position, config))
    scores = Scores()
    scores.update(
        voxelize(torch.from_numpy(predicted - position), 3.2, 16),
        voxelize(torch.from_numpy(target - position), 3.2, 16),
        0.2,
    )
    for key, value in scores.result().items():
        assert points[key] == pytest.approx(value)
    assert [len(category) for category in errors] == [1, 1, 1]
    # Empty clouds and duplicate samples must not manufacture matches or NaN metrics.
    empty, _ = comparison(cloud(np.empty((0, 3)), position, config), cloud(target, position, config))
    assert empty["f1"] == 0 and empty["matched_centroid_error_cm"] is None
    assert len(thin(predicted, 2)) == 2 and len(thin(predicted, 10)) == 4


def test_sparse_exports_do_not_skip_inference_or_cross_episode_history(tmp_path):
    pytest.importorskip("MinkowskiEngine")
    from ..training.models.architectures import make_model
    from ..training.datasets.sequence import discover_runs

    torch.set_num_threads(2)
    config = plane_config()
    write_run(tmp_path, "a", b"terrain", [3, 3])
    runs = discover_runs(tmp_path, 3.2)
    model = make_model(config).eval()
    expected = evaluate(model, runs, config, torch.device("cpu"))
    output = tmp_path / "exports"
    sampled = evaluate(model, runs, config, torch.device("cpu"), export=output, export_frames=2, export_every=4)
    assert sampled["frames"] == 6 and sampled["episodes_started"] == 2
    assert sampled["prediction"] == expected["prediction"]
    assert sampled["loss"] == pytest.approx(expected["loss"])
    paths = sorted(output.glob("predictions/*/*/*/*.npz"))
    assert len(paths) == 2
    assert [(p.parent.name, p.name) for p in paths] == [
        ("episode_000000", "frame_000000.npz"),
        ("episode_000001", "frame_000001.npz"),
    ]
