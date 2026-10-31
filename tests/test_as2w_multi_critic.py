import gymnasium as gym
import pytest

import Qmini.tasks  # noqa: F401
from isaaclab.managers import RewardTermCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from Qmini.tasks.quadruped.as2w.env.multi_critic_env_cfg import group_rewards


@pytest.mark.parametrize('terrain', ['Flat', 'Rough'])
def test_multi_critic_registration_and_reward_preservation(terrain):
    task = f'Qmini-Velocity-{terrain}-AS2W'
    base = load_cfg_from_registry(task, 'env_cfg_entry_point')
    grouped = load_cfg_from_registry(task + '-MultiCritic', 'env_cfg_entry_point')
    agent = load_cfg_from_registry(task + '-MultiCritic', 'rsl_rl_cfg_entry_point')
    assert list(grouped.rewards) == list(agent.algorithm.critic_groups)
    assert list(grouped.rewards) == ['task', 'style', 'safety']
    assert set(grouped.rewards['task']) == {
        'track_lin_vel_xy_exp', 'track_ang_vel_z_exp', 'gravity_z_alignment',
    }
    assert set(grouped.rewards['safety']) == {'undesired_contacts', 'dof_pos_limits', 'dof_torque_limits'}
    assert {'dof_torques_l2', 'action_rate_l2', 'flat_four_wheel_contact', 'turning_lateral_gait'} <= set(
        grouped.rewards['style']
    )
    original = {k: v for k, v in vars(base.rewards).items() if isinstance(v, RewardTermCfg)}
    flattened = {k: v for group in grouped.rewards.values() for k, v in group.items()}
    assert original.keys() == flattened.keys()
    for name in original:
        assert original[name].to_dict() == flattened[name].to_dict()
    assert agent.algorithm.class_name.endswith(':MultiCriticPPO')
    assert gym.spec(task + '-MultiCritic').entry_point.endswith(':MultiCriticManagerBasedRLEnv')
    assert agent.obs_groups == {'actor': ['critic'], 'critic': ['critic']}
    grouped.to_dict()
    agent.to_dict()


def test_unassigned_reward_rejected():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='unassigned'):
        group_rewards(SimpleNamespace(new_term=RewardTermCfg(func=lambda env: 0, weight=1.0)))
