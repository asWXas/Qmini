from types import SimpleNamespace as N

import torch

from isaaclab.managers import SceneEntityCfg
from Qmini.tasks.quadruped.as2w.mdp.rewards import diagonal_wheel_distance_difference


def test_diagonal_terrain_tolerance_and_xy_projection():
    heights = torch.tensor([[0., .02], [0., .04], [0., .03], [0., float('nan')]])
    hits = torch.zeros(4, 2, 3)
    hits[..., 2] = heights
    positions = torch.zeros(4, 4, 3)
    positions[:, 2, 0] = 1.04
    positions[:, 3, 0] = 1.0
    # Large wheel height differences must not change XY diagonal distances.
    positions[:, 2, 2] = 2.0
    class Scene(dict):
        pass
    scene = Scene(robot=N(data=N(body_link_pos_w=N(torch=positions))))
    scene.sensors = {'scan': N(data=N(ray_hits_w=N(torch=hits)))}
    env = N(scene=scene)
    result = diagonal_wheel_distance_difference(env, SceneEntityCfg('robot'), SceneEntityCfg('scan'))
    torch.testing.assert_close(result, torch.tensor([.01, 0., 0., 0.]))
    positions[:, 2, 0] = 1.06
    result = diagonal_wheel_distance_difference(env, SceneEntityCfg('robot'), SceneEntityCfg('scan'))
    torch.testing.assert_close(result, torch.tensor([.03, .01, .01, .01]))
