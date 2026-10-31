from types import SimpleNamespace as N

import torch

from isaaclab.managers import SceneEntityCfg
from Qmini.tasks.quadruped.as2w.mdp.rewards import flat_terrain_four_wheel_contact


def test_flat_contact_gate():
    # Flat, rough, missing wheel contact, partial scan, invalid scan, threshold, shifted world height.
    heights = torch.tensor([[0., .02], [0., .12], [0., .02], [0., float('inf')],
                            [float('nan'), float('inf')], [0., .1], [-5., -4.98]])
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
    torch.testing.assert_close(result, torch.tensor([1., 0., 0., 0., 0., 0., 1.]))
    for count in range(5):
        forces[0] = 0
        forces[0, :count, 2] = 10
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[0], torch.tensor(float(count == 4)))
    # Each individual wheel is required, regardless of its index.
    for wheel in range(4):
        forces[0, :, 2] = 10
        forces[0, wheel, 2] = 0
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[0], torch.tensor(0.))
    forces[:, :, 2] = 10
    all_contact = torch.tensor(1.)
    # Zero command enables the reward on rough terrain.
    for command, expected in (([0., 0., 0.], 1.), ([0.05, 0.05, 0.05], 0.),
                              ([0.08, 0.08, 0.], 0.), ([0., 0., 0.1], 0.)):
        commands[1] = torch.tensor(command)
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[1], expected * all_contact)
    commands[4] = 0
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result[4], torch.tensor(1.))  # Zero command bypasses scan validity.
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
        ([0., 0.001, 0.], 0.), ([0.05, 0.01, 0.], 0.),
        ([0., 0., 0.], 1.),
    ):
        commands[0] = torch.tensor(command)
        result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
        torch.testing.assert_close(result[0], expected * all_contact)

    # Height threshold is strict and configurable; all wheels are in contact.
    commands[:] = torch.tensor([0.3, 0., 0.])
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result, torch.tensor([1., 0., 1., 0., 0., 0., 1.]))
    result = flat_terrain_four_wheel_contact(
        env, SceneEntityCfg('scan'), SceneEntityCfg('contact'), height_threshold=0.15
    )
    torch.testing.assert_close(result, torch.tensor([1., 1., 1., 0., 0., 1., 1.]))

    commands[:] = 0
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result, torch.ones(7))
    forces[:, 0, 2] = 0
    result = flat_terrain_four_wheel_contact(env, SceneEntityCfg('scan'), SceneEntityCfg('contact'))
    torch.testing.assert_close(result, torch.zeros(7))
