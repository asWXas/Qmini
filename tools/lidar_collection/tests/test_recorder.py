"""Exercise sequence boundaries and saved sensor fields without simulation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from ..core.recorder import collect


@pytest.mark.parametrize("scan_every", [1, 5])
def test_reset_splits_trajectory_and_invalid_hits_are_removed(tmp_path, scan_every):
    tensor = torch.tensor
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=tensor([[0.0, 0.0, 0.0]]),
            root_quat_w=tensor([[0.0, 0.0, 0.0, 1.0]]),
            root_ang_vel_b=tensor([[0.0, 0.0, 0.2]]),
        )
    )
    lidar = SimpleNamespace(
        cfg=SimpleNamespace(offset=SimpleNamespace(pos=(0.2, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0))),
        data=SimpleNamespace(
            ray_hits_w=tensor([[[1.0, 0.0, 0.0], [float("inf"), 0.0, 0.0], [20.0, 0.0, 0.0]]]),
            pos_w=tensor([[0.0, 0.0, 0.0]]),
            quat_w=tensor([[0.0, 0.0, 0.0, 1.0]]),
        ),
    )

    class Environment:
        def __init__(self):
            self.unwrapped = SimpleNamespace(
                scene={"robot": robot, "collection_lidar": lidar}, num_envs=1, step_dt=0.02
            )
            self.step_id = 0

        def get_observations(self):
            return None

        def step(self, action):
            self.step_id += 1
            return None, None, tensor([self.step_id == 2]), None

    class Policy:
        def __call__(self, obs):
            return None

        def reset(self, done):
            pass

    args = SimpleNamespace(
        seed=42,
        checkpoint="test.pt",
        scan_every=scan_every,
        steps=3,
        min_range=0.2,
        max_range=10.0,
        map_size=3.2,
        gt_density=10,
    )
    collect(Environment(), Policy(), trimesh.creation.box(), args, tmp_path)
    files = sorted(tmp_path.glob("env_*/*/*.npz"))
    assert len(files) == 2
    assert files[0].parent.name == "episode_000000"
    assert files[1].parent.name == "episode_000001"
    assert files[0].name == files[1].name == "frame_000000.npz"
    with np.load(files[1]) as frame:
        assert frame["step"] == 3
        np.testing.assert_allclose(frame["time_s"], 0.06)
        np.testing.assert_array_equal(frame["lidar_points_w"], [[1, 0, 0]])
        np.testing.assert_array_equal(frame["ray_indices"], [0])
        np.testing.assert_allclose(frame["sensor_pos_w"], [0.2, 0, 0])
        assert frame["episode_step"] == 1
    states = [json.loads(line) for line in (tmp_path / "states.jsonl").read_text().splitlines()]
    assert [state["episode_id"] for state in states] == [0, 1, 1]
    assert states[1]["reset"] is True
    first = files[0].parent
    second = files[1].parent
    assert json.loads((first / "episode.json").read_text())["end_reason"] == "reset"
    assert json.loads((second / "episode.json").read_text())["frame_count"] == 1
    local_states = [json.loads(line) for line in (second / "states.jsonl").read_text().splitlines()]
    assert [state["episode_step"] for state in local_states] == [0, 1]
