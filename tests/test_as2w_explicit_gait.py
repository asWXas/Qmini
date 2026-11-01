from types import SimpleNamespace as N

import pytest
import torch
from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import RewardsCfg, ObservationsCfg


def test_penalty_internal_phase_and_gate():
    cfg = RewardsCfg().explicit_diagonal_gait
    cfg.params['period'] = 0.4
    cfg.params['frequency'] = None
    assert cfg.weight == -0.5
    cfg.params['sensor_cfg'].body_ids = [0, 1, 2, 3]
    commands = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., .2, 0.], [0., 0., -.2]])
    forces = torch.zeros(4, 4, 3)
    forces[:, [0, 3], 2] = 10.
    env = N(episode_length_buf=torch.zeros(4), step_dt=.02,
            command_manager=N(get_command=lambda name: commands),
            scene=N(sensors={'contact_forces': N(data=N(net_forces_w=N(torch=forces)))}))
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.zeros(4))
    env.episode_length_buf[:] = 10
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([0., 0., 1., 1.]))
    env.episode_length_buf[2] = 0
    assert cfg.func(env, **cfg.params)[2] == 0
    cfg.params['enabled'] = False
    assert (cfg.func(env, **cfg.params) == 0).all()
    cfg.params['enabled'] = True
    cfg.params['frequency'] = 5.0  # at 0.2 seconds, now a full cycle
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.zeros(4))
    cfg.params['frequency'] = 0.
    with pytest.raises(ValueError, match='frequency'):
        cfg.func(env, **cfg.params)
    obs = ObservationsCfg()
    assert not hasattr(obs.policy, 'diagonal_gait')
    assert not hasattr(obs.critic, 'diagonal_gait')
