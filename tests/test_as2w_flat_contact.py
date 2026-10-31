from types import SimpleNamespace as N

import torch

from isaaclab.managers import SceneEntityCfg
from Qmini.tasks.quadruped.as2w.mdp.rewards import flat_terrain_four_wheel_contact


def test_flat_contact_gate():
    # Flat, rough, missing wheel contact, partial scan, invalid scan, threshold, shifted world height.
    heights = torch.tensor([[0., .02], [0., .05], [0., .02], [0., float('inf')],
                            [float('nan'), float('inf')], [0., .04], [-5., -4.98]])
    hits = torch.zeros(7, 2, 3)
    hits[:, :, 2] = heights
    forces = torch.zeros(7, 4, 3)
    forces[:, :, 2] = 10
    forces[2, 0, 2] = 0
    env = N(scene=N(sensors={
        'scan': N(data=N(ray_hits_w=N(torch=hits))),
        'contact': N(data=N(net_forces_w=N(torch=forces))),
    }))
    commands = torch.tensor([[0.3, 0., 0.]]).repeat(7, 1)
    env.command_manager = N(get_command=lambda name: commands)
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result, torch.tensor([0., 0., 0.25, 0., 0., 0., 0.]))
    for count in range(5):
        forces[0] = 0
        forces[0, :count, 2] = 10
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[0], torch.tensor(1.0 - count / 4.0))
    forces[:, 0, 2] = 0
    missing_one = torch.tensor(0.25)
    # Rough terrain disables the penalty even at zero command.
    for command, expected in (([0., 0., 0.], 0.), ([0.05, 0.05, 0.05], 0.),
                              ([0.08, 0.08, 0.], 0.), ([0., 0., 0.1], 0.)):
        commands[1] = torch.tensor(command)
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[1], expected * missing_one)
    commands[4] = 0
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result[4], torch.tensor(0.))  # Invalid scan disables the penalty.
    # Flat terrain allows forward/backward motion, but not lateral/yaw/mixed commands.
    for command, expected in (
        ([1., 0., 0.], 1.), ([-1., 0., 0.], 1.),
        ([3., 0., 0.], 1.), ([-3., 0., 0.], 1.),
        ([0.75, 0., 0.], 1.), ([-0.75, 0., 0.], 1.),
        ([0.74, 0., 0.], 1.), ([-0.74, 0., 0.], 1.),
        ([0.5, 0., 0.], 1.), ([-0.5, 0., 0.], 1.),
        ([0.49, 0., 0.], 1.), ([-0.49, 0., 0.], 1.),
        ([0.15, 0., 0.], 1.), ([0.2, 0., 0.], 1.),
        ([0.21, 0., 0.], 1.), ([-0.21, 0., 0.], 1.),
        ([0., 0.2, 0.], 0.), ([0., 0., -0.2], 0.),
        ([1., 0.2, 0.], 0.), ([1., 0., 0.2], 0.),
        ([0.3, 0.001, -0.001], 1.), ([0.3, 0.002, 0.], 0.),
        ([0., 0.01, 0.], 0.), ([0., 0., 0.01], 0.),
        ([0., 0.001, 0.], 1.), ([0.05, 0.01, 0.], 0.),
        ([0., 0., 0.], 1.),
    ):
        commands[0] = torch.tensor(command)
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[0], expected * missing_one)

    # Height threshold is strict, and configurable; all rows miss one contact.
    commands[:] = 0
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result, torch.tensor([.25, 0., .25, 0., 0., 0., .25]))
    result = flat_terrain_four_wheel_contact(
        env, SceneEntityCfg('scan'), SceneEntityCfg('contact'), height_threshold=0.06
    )
    torch.testing.assert_close(result, torch.tensor([.25, .25, .25, 0., 0., .25, .25]))
