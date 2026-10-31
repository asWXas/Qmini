from types import SimpleNamespace as N

import torch

from Qmini.tasks.quadruped.as2w.mdp.observations import wheel_scan_height_range


def test_wheel_scan_height_range():
    names = ['fl', 'fr', 'rl', 'rr']
    sensors = {}
    for index, name in enumerate(names):
        hits = torch.zeros(3, 9, 3)
        hits[0, :, 2] = torch.linspace(0., index * .02, 9)
        hits[1, :, 2] = hits[0, :, 2] + 5.0
        hits[2, :, 2] = float('inf')
        sensors[name] = N(data=N(ray_hits_w=N(torch=hits)))
    env = N(scene=N(sensors=sensors))
    result = wheel_scan_height_range(env, names)
    assert result.shape == (3, 4)
    torch.testing.assert_close(result, torch.tensor([
        [0., .02, .04, .06], [0., .02, .04, .06], [1.5, 1.5, 1.5, 1.5],
    ]))
