"""Command gating and diagonal synchronization checks."""

from types import SimpleNamespace as N

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from Qmini.tasks.quadruped.as2w.mdp.rewards import GaitReward, air_time_variance_penalty


def test_air_time_variance_gate_clipping_and_reentry():
    from isaaclab.managers import SceneEntityCfg

    commands = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., .2, 0.],
                             [0., 0., -.2], [1., .08, .08], [0., .1, 0.]])
    # Select only wheel bodies; long completed intervals saturate at 0.5 s.
    air = torch.tensor([[9., .1, .3, .6, 30.]]).repeat(6, 1)
    contact = torch.tensor([[9., .1, .1, .3, .3]]).repeat(6, 1)
    sensor = N(cfg=N(track_air_time=True), data=N(
        last_air_time=N(torch=air), last_contact_time=N(torch=contact),
    ))
    env = N(scene=N(sensors={"contact_forces": sensor}),
            command_manager=N(get_command=lambda name: commands))
    cfg = RewardTermCfg(func=air_time_variance_penalty, weight=-0.25, params={
        "command_name": "base_velocity", "command_threshold": 0.1,
    })
    cfg.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_ids=[1, 2, 3, 4])
    original_air, original_contact = air.clone(), contact.clone()
    expected = torch.tensor([0., 0., .05, .05, .05, 0.])
    torch.testing.assert_close(cfg.func(env, **cfg.params), expected)
    # Entering and re-entering a turn immediately uses the unchanged sensor history.
    commands[0, 2] = .2
    assert cfg.func(env, **cfg.params)[0] > 0
    commands[0, 2] = 0.
    assert cfg.func(env, **cfg.params)[0] == 0
    commands[0, 2] = -.2
    torch.testing.assert_close(cfg.func(env, **cfg.params)[0], torch.tensor(.05))
    torch.testing.assert_close(air, original_air)
    torch.testing.assert_close(contact, original_contact)
    air[:, 1:] = 30.
    contact[:, 1:] = 30.
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.zeros(6))
    sensor.cfg.track_air_time = False
    with pytest.raises(RuntimeError, match="track_air_time"):
        cfg.func(env, **cfg.params)


def test_gait_command_gate_and_pair_timing():
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.2],
            [1.0, 0.2, 0.2],
            [0.0, 0.08, 0.08],
            [0.0, 0.1, 0.0],
        ]
    )
    names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    air = torch.tensor([[0.1, 0.0, 0.0, 0.1]]).repeat(len(commands), 1)
    contact = 0.1 - air
    sensor = N(
        cfg=N(track_air_time=True),
        data=N(current_air_time=N(torch=air), current_contact_time=N(torch=contact)),
        find_sensors=lambda pair: ([names.index(name) for name in pair], pair),
    )
    env = N(scene=N(sensors={"contact_forces": sensor}), command_manager=N(get_command=lambda name: commands))
    cfg = RewardTermCfg(func=GaitReward, weight=0.5, params={
        "std": 0.1, "max_err": 0.2,
        "synced_feet_pair_names": [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]],
        "sensor_cfg": SceneEntityCfg("contact_forces"),
        "command_name": "base_velocity", "command_threshold": 0.1,
    })
    term = GaitReward(cfg, env)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(term(env, **cfg.params), expected)
    # Breaking one synchronized diagonal reduces the active reward.
    air[:, 0] = 0.0
    contact[:, 0] = 0.1
    result = term(env, **cfg.params)
    assert (result[expected == 1] < 1).all()
    assert (result[expected == 0] == 0).all()


def test_air_time_limits():
    from isaaclab.managers import SceneEntityCfg

    from Qmini.tasks.quadruped.as2w.mdp.rewards import gait_air_time_limits

    # Too short, in range, too long at landing, ongoing timeout, reset contact, pure VX.
    last = torch.tensor([[0.05], [0.25], [0.45], [0.0], [0.0], [0.05]])
    current = torch.tensor([[0.0], [0.0], [0.0], [0.5], [0.0], [0.0]])
    landed = torch.tensor([[True], [True], [True], [False], [True], [True]])
    commands = torch.tensor([[0.0, 0.2, 0.0]] * 5 + [[1.0, 0.0, 0.0]])
    sensor = N(
        data=N(last_air_time=N(torch=last), current_air_time=N(torch=current)),
        compute_first_contact=lambda dt: N(torch=landed),
    )
    env = N(
        scene=N(sensors={"contact_forces": sensor}),
        step_dt=0.02,
        command_manager=N(get_command=lambda name: commands),
    )
    result = gait_air_time_limits(env, SceneEntityCfg("contact_forces", body_ids=[0]))
    torch.testing.assert_close(result, torch.tensor([0.1, 0.0, 0.0, 0.15, 0.0, 0.0]))
    # An active swing must not be penalized as too short before it lands.
    current[0] = 0.05
    landed[0] = False
    assert gait_air_time_limits(env, SceneEntityCfg("contact_forces", body_ids=[0]))[0] == 0
    for duration, penalty in ((0.35, 0.0), (0.36, 0.01), (0.40, 0.05)):
        current[0] = duration
        result = gait_air_time_limits(env, SceneEntityCfg("contact_forces", body_ids=[0]))
        torch.testing.assert_close(result[0], torch.tensor(penalty))

    commands[0] = torch.tensor([0.0, 0.08, 0.08])
    current[0] = 0.40
    result = gait_air_time_limits(env, SceneEntityCfg("contact_forces", body_ids=[0]))
    torch.testing.assert_close(result[0], torch.tensor(0.05))
    commands[0] = torch.tensor([0.0, 0.1, 0.0])
    assert gait_air_time_limits(env, SceneEntityCfg("contact_forces", body_ids=[0]))[0] == 0
