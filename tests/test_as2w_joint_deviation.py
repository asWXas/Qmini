from types import SimpleNamespace as N

import torch

from isaaclab.managers import SceneEntityCfg
from Qmini.tasks.quadruped.as2w.mdp.rewards import stand_still_joint_deviation


def test_joint_groups_and_standing_gate():
    names = ['FL_calf', 'FL_foot', 'FL_hip', 'FL_thigh', 'RR_hip']
    default = torch.tensor([[-1.6, 0., 0., .8, 0.]]).repeat(4, 1)
    offsets = torch.tensor([[3., 100., -1., 2., 4.]]).repeat(4, 1)
    asset = N(joint_names=names, data=N(joint_pos=N(torch=default + offsets),
                                     default_joint_pos=N(torch=default)))
    commands = torch.tensor([[0., 0., 0.], [.09, 0., 0.], [0., 0., .1], [1., 0., 0.]])
    env = N(scene={'robot': asset}, command_manager=N(get_command=lambda name: commands))
    cfg = SceneEntityCfg('robot', joint_ids=[4, 0, 3, 2])
    result = stand_still_joint_deviation(
        env, cfg, 'base_velocity', moving_hip_scale=.1, moving_thigh_scale=.2, moving_calf_scale=.3
    )
    # Standing: (3+1+2+4)*1.5. Moving: (1+4)*.1 + 2*.2 + 3*.3.
    torch.testing.assert_close(result, torch.tensor([15., 15., 1.8, 1.8]))
    result = stand_still_joint_deviation(env, cfg, 'base_velocity')
    torch.testing.assert_close(result, torch.tensor([15., 15., 2.5, 2.5]))
    cfg.joint_ids = slice(2, 5)
    result = stand_still_joint_deviation(
        env, cfg, 'base_velocity', moving_hip_scale=.1, moving_thigh_scale=.2, moving_calf_scale=.3
    )
    torch.testing.assert_close(result, torch.tensor([10.5, 10.5, .9, .9]))
