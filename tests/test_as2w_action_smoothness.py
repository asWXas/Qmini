"""Second-order action history and partial-reset checks."""

from types import SimpleNamespace

import torch

from isaaclab.managers import RewardTermCfg

from Qmini.tasks.quadruped.as2w.mdp.rewards import action_rate_l2_second_order


def test_action_second_order_and_partial_reset():
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.zeros(2, 2), prev_action=torch.zeros(2, 2)),
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )
    term = action_rate_l2_second_order(RewardTermCfg(func=action_rate_l2_second_order, weight=-0.01), env)

    def step(actions):
        env.action_manager.prev_action.copy_(env.action_manager.action)
        env.action_manager.action.copy_(torch.tensor(actions, dtype=torch.float))
        env.common_step_counter += 1
        env.episode_length_buf += 1
        result = term(env)
        torch.testing.assert_close(term(env), result)  # Repeated evaluation must not advance history.
        return result

    assert torch.equal(step([[1, 1], [1, 1]]), torch.zeros(2))
    assert torch.equal(step([[2, 2], [2, 2]]), torch.zeros(2))
    # Linear ramp has zero second difference; a jump of two in both channels gives eight.
    assert torch.equal(step([[3, 3], [5, 5]]), torch.tensor([0., 8.]))
    term.reset([1])
    env.episode_length_buf[1] = 0
    env.action_manager.action[1] = 0
    env.action_manager.prev_action[1] = 0
    assert torch.equal(step([[4, 4], [8, 8]]), torch.zeros(2))
    assert torch.equal(step([[5, 5], [8, 8]]), torch.zeros(2))
    assert torch.equal(step([[6, 6], [8, 8]]), torch.zeros(2))
