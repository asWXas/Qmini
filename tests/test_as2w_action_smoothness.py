"""Second-order action history and partial-reset checks."""

from types import SimpleNamespace

import torch

from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import RewardsCfg


def test_action_second_order_and_partial_reset():
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.zeros(2, 2), prev_action=torch.zeros(2, 2)),
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )
    cfg = RewardsCfg().action_second_order_l2
    term = cfg.func(cfg, env)

    def step(actions):
        env.action_manager.prev_action.copy_(env.action_manager.action)
        env.action_manager.action.copy_(torch.tensor(actions, dtype=torch.float))
        env.common_step_counter += 1
        env.episode_length_buf += 1
        result = term(env, **cfg.params)
        torch.testing.assert_close(term(env, **cfg.params), result)  # Repeated evaluation must not advance history.
        first_order = RewardsCfg().action_rate_l2
        expected = (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=-1)
        torch.testing.assert_close(first_order.func(env, **first_order.params), expected)
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
