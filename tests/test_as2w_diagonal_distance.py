from types import SimpleNamespace as N

import torch

from isaaclab.managers import SceneEntityCfg
from Qmini.tasks.quadruped.as2w.mdp.rewards import diagonal_wheel_distance_difference


def test_diagonal_tolerance_and_xy_projection():
    positions = torch.zeros(5, 4, 3, dtype=torch.float64)
    # Test equal, below threshold, at threshold, above threshold, and reversed diagonals.
    positions[:, 2, 0] = torch.tensor([0., .02, .03, .06, 0.], dtype=torch.float64)
    positions[-1, 3, 1] = .06
    positions[:, 2, 2] = 2.0  # Z must not affect XY lengths.
    env = N(scene={'robot': N(data=N(body_link_pos_w=N(torch=positions)))})
    result = diagonal_wheel_distance_difference(env, SceneEntityCfg('robot'))
    torch.testing.assert_close(result, torch.tensor([0., 0., 0., .03, .03], dtype=torch.float64))
    result = diagonal_wheel_distance_difference(env, SceneEntityCfg('robot'), tolerance=.01)
    torch.testing.assert_close(result, torch.tensor([0., .01, .02, .05, .05], dtype=torch.float64))
