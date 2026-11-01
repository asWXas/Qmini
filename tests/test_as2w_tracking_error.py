from types import SimpleNamespace as N

import torch
from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import RewardsCfg


def test_separate_squared_tracking_errors():
    command = torch.tensor([[1., -2., 3.], [0., 0., 0.]])
    data = N(root_lin_vel_b=N(torch=torch.tensor([[-1., -1., 99.], [0., 0., 99.]])),
             root_ang_vel_b=N(torch=torch.tensor([[99., 99., -1.], [99., 99., 0.]])))
    env = N(scene={'robot': N(data=data)}, command_manager=N(get_command=lambda name: command))
    cfg = RewardsCfg()
    for axis, expected in [('x', 4.), ('y', 1.), ('yaw', 16.)]:
        term = getattr(cfg, f'track_{axis}_error_l2')
        result = term.func(env, **term.params)
        torch.testing.assert_close(result, torch.tensor([expected, 0.]))
        assert term.weight == -0.01
        assert (result * term.weight <= 0).all()
