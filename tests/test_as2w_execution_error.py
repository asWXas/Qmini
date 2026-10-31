"""Execution-error contracts without launching the simulator."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction, JointPositionAction, JointVelocityAction
from isaaclab.managers import SceneEntityCfg

from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import ActionsCfg, ObservationsCfg
from Qmini.tasks.quadruped.as2w.mdp.observations import (
    joint_position_execution_error,
    joint_velocity_execution_error,
)


@pytest.mark.parametrize(
    "action_type,observe,field,scale,offset,actual,expected,obs_scale",
    [
        (JointPositionAction, joint_position_execution_error, "position", 0.125, 0.8, 0.9, 0.15, 1.0),
        (JointVelocityAction, joint_velocity_execution_error, "velocity", 5.0, 0.0, 7.0, 3.0, 0.05),
    ],
)
def test_processed_target_error(action_type, observe, field, scale, offset, actual, expected, obs_scale):
    # Scrambled joint selection catches accidental use of policy or simulator ordering.
    ids = [3, 1]
    target = torch.full((2, 4), -99.0)
    state = torch.full((2, 4), actual)
    asset = SimpleNamespace(
        actuators=SimpleNamespace(target_command=SimpleNamespace(**{field: SimpleNamespace(torch=target)})),
        data=SimpleNamespace(**{"joint_pos" if field == "position" else "joint_vel": SimpleNamespace(torch=state)}),
    )

    def write_target(*, target, joint_ids):
        # Simulated articulation sink; action processing/application is the real Lab implementation.
        getattr(asset.actuators.target_command, field).torch[:, joint_ids] = target

    setattr(asset, f"set_joint_{field}_target_index", write_target)
    action = action_type.__new__(action_type)
    action._asset = asset
    action.cfg = SimpleNamespace(clip=None)
    action._joint_ids = ids
    action._raw_actions = torch.zeros(2, 2)
    action._scale = scale
    action._offset = offset
    action._clip = None
    JointAction.process_actions(action, torch.tensor([[2.0, -2.0], [2.0, -2.0]]))
    action.apply_actions()
    env = SimpleNamespace(scene={"robot": asset}, episode_length_buf=torch.tensor([5, 0]))
    cfg = SceneEntityCfg("robot", joint_ids=ids)
    saved_target = target.clone()
    saved_state = state.clone()
    error = observe(env, cfg)
    torch.testing.assert_close(error[0, 0], torch.tensor(expected))
    torch.testing.assert_close(error[0, 1], torch.tensor(-2.0 * scale + offset - actual))
    assert torch.count_nonzero(error[1]) == 0
    # Observation scaling must not modify the underlying target or physical state.
    error.mul_(obs_scale)
    torch.testing.assert_close(error[0, 0], torch.tensor(expected * obs_scale))
    torch.testing.assert_close(target, saved_target)
    torch.testing.assert_close(state, saved_state)
    # Changing the latest target must be visible immediately (no stale cached error).
    target[0, ids[0]] += 1.0
    torch.testing.assert_close(observe(env, cfg)[0, 0], torch.tensor(expected + 1.0))


def test_action_and_observation_joint_contract():
    actions, observations = ActionsCfg(), ObservationsCfg()
    for action, term, count, scale in (
        (actions.joint_pos, observations.critic.leg_position_error, 12, 1.0),
        (actions.wheel_vel, observations.critic.wheel_velocity_error, 4, 0.05),
    ):
        selection = term.params["asset_cfg"]
        assert action.joint_names == selection.joint_names
        assert len(selection.joint_names) == count
        assert action.preserve_order and selection.preserve_order
        assert term.scale == scale
    assert not set(actions.joint_pos.joint_names) & set(actions.wheel_vel.joint_names)
