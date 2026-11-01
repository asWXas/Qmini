from types import SimpleNamespace as N

import gymnasium as gym
import pytest
import torch

from Qmini.tasks.quadruped.as2w_lp_cc.mdp.lp_curriculum import (
    LPACRLCfg, LearningProgressSampler, epte_with_stability,
)
from Qmini.tasks.quadruped.as2w_lp_cc.env.lp_env import LPCurriculumController
from Qmini.tasks.quadruped.as2w_lp_cc.env.as2w_env_cfg import AS2WRoughEnvCfg, AS2WFlatEnvCfg
from Qmini.tasks.quadruped.as2w_lp_cc.env.multi_critic_env_cfg import AS2WRoughMultiCriticEnvCfg


def small_cfg(**kwargs):
    return LPACRLCfg(terrain_names=('a',), terrain_levels=1, vx_edges=(0., 1., 2., 3.),
                    yaw_edges=(0., 0.), min_episodes_per_task=1, stage_steps=2, **kwargs)


def test_signed_lp_softmax_and_stage_aggregation():
    s = LearningProgressSampler(small_cfg(temperature=2.))
    s.observe(torch.tensor([0, 0, 1, 2]), torch.tensor([8., 12., 10., 10.]))
    assert not s.advance()
    assert s.advance()
    torch.testing.assert_close(s.probabilities, torch.full((3,), 1/3, dtype=torch.float64))
    s.observe(torch.arange(3), torch.tensor([12., 10., 8.]))
    s.finish_stage()
    torch.testing.assert_close(s.learning_progress, torch.tensor([2., 0., -2.], dtype=torch.float64))
    torch.testing.assert_close(s.probabilities, torch.softmax(torch.tensor([1., 0., -1.], dtype=torch.float64), 0))
    assert s.probabilities[0] > s.probabilities[1] > s.probabilities[2]


def test_missing_task_is_not_zero_return_or_stale_progress():
    s = LearningProgressSampler(small_cfg())
    s.observe(torch.arange(3), torch.tensor([10., 20., 30.]))
    s.finish_stage()
    s.observe(torch.tensor([0]), torch.tensor([12.]))
    s.finish_stage()
    torch.testing.assert_close(s.learning_progress, torch.tensor([2., 0., 0.], dtype=torch.float64))
    s.observe(torch.tensor([1]), torch.tensor([50.]))
    s.finish_stage()
    assert (s.learning_progress == 0).all()


def test_bins_signs_and_serialization():
    cfg = LPACRLCfg()
    s = LearningProgressSampler(cfg)
    assert s.num_tasks == 600
    ids = torch.arange(600).repeat(5)
    commands = s.commands(ids)
    table = s.table[ids]
    for axis, edges, dim in [(0, cfg.vx_edges, 2), (2, cfg.yaw_edges, 3)]:
        edges = torch.tensor(edges)
        assert (commands[:, axis].abs() >= edges[table[:, dim]]).all()
        assert (commands[:, axis].abs() < edges[table[:, dim]+1]).all()
        assert (commands[:, axis] < 0).any() and (commands[:, axis] > 0).any()
    assert commands[:, 1].abs().max() <= .5
    s.observe(torch.tensor([0, 1]), torch.tensor([1., 2.]))
    s.advance()
    state = s.state_dict()
    restored = LearningProgressSampler(cfg)
    restored.load_state_dict(state)
    torch.testing.assert_close(s.sample(500), restored.sample(500))
    torch.testing.assert_close(s.commands(ids), restored.commands(ids))
    assert restored.steps_in_stage == 1
    torch.testing.assert_close(restored.count, s.count)
    mismatch = LearningProgressSampler(LPACRLCfg(temperature=2.))
    with pytest.raises(ValueError, match='temperature'):
        mismatch.load_state_dict(state)


def test_epte_sp():
    errors = torch.tensor([[20., 40., 0.], [2., 4., 0.], [20., 40., 0.]])
    steps = torch.tensor([100, 10, 100])
    result = epte_with_stability(errors, steps, 100, torch.tensor([False, True, True]))
    torch.testing.assert_close(result, torch.tensor([[.2, .4, 0.], [.92, .94, .9], [.2, .4, 0.]]))


class Scene(dict):
    @property
    def terrain(self):
        return self['terrain']


def test_reset_attributes_completed_return_to_old_task_and_moves_origin():
    cfg = small_cfg()
    terrain = N(terrain_origins=torch.tensor([[[4., 5., 6.]]]), terrain_types=torch.zeros(2, dtype=torch.long),
                terrain_levels=torch.zeros(2, dtype=torch.long), env_origins=torch.zeros(2, 3))
    robot = N(data=N(root_lin_vel_b=N(torch=torch.zeros(2, 3)), root_ang_vel_b=N(torch=torch.zeros(2, 3)), root_pos_w=N(torch=torch.zeros(2, 3))))
    env = N(cfg=N(lp_acrl=cfg, scene=N(terrain=N(terrain_generator=N(sub_terrains={'a': None}, num_rows=1, num_cols=1, size=(12., 12.))))),
            device='cpu', num_envs=2, scene=Scene(terrain=terrain, robot=robot),
            episode_length_buf=torch.zeros(2, dtype=torch.long), max_episode_length=10,
            termination_manager=N(terminated=torch.tensor([False, True])))
    c = LPCurriculumController(env)
    c.reset(torch.arange(2))
    assert c.sampler.count.sum() == 0
    c.task_ids[:] = torch.tensor([0, 1])
    c.commands[:] = 0.
    env.episode_length_buf += 1
    c.record_step(torch.tensor([3., -4.]))
    c.in_step = True
    c.forced_task_ids = torch.tensor([2, 2])
    c.reset(torch.tensor([1]))
    torch.testing.assert_close(c.sampler.return_sum, torch.tensor([0., -4., 0.], dtype=torch.float64))
    assert c.task_ids.tolist() == [0, 2]
    assert c.alive_steps.tolist() == [1, 0]
    torch.testing.assert_close(terrain.env_origins[1], torch.tensor([4., 5., 6.]))
    assert c.returns.tolist() == [3., 0.]
    # Random episode timer is censored and must not be considered a full episode.
    env.episode_length_buf[0] = 8
    c.reset(torch.tensor([0]))
    assert c.sampler.count.sum() == 1


def test_independent_task_registration_and_configs():
    assert '.as2w.env.' in gym.spec('Qmini-Velocity-Rough-AS2W').kwargs['env_cfg_entry_point']
    for terrain in ('Flat', 'Rough'):
        spec = gym.spec(f'Qmini-Velocity-{terrain}-AS2W-LP-ACRL')
        assert spec.entry_point.endswith(':LPACRLEnv')
    rough, flat, multi = AS2WRoughEnvCfg(), AS2WFlatEnvCfg(), AS2WRoughMultiCriticEnvCfg()
    assert LearningProgressSampler(rough.lp_acrl).num_tasks == 600
    assert LearningProgressSampler(flat.lp_acrl).num_tasks == 8
    assert rough.curriculum.terrain_levels is None
    assert not rough.commands.base_velocity.heading_command
    assert tuple(rough.scene.terrain.terrain_generator.sub_terrains) == rough.lp_acrl.terrain_names
    assert 'track_lin_vel_xy_exp' in multi.rewards['task']
    rough.play_mode()
    assert not rough.lp_acrl.adaptive
    assert rough.scene.terrain.terrain_generator.num_rows == 4
    assert rough.scene.terrain.terrain_generator.num_cols == 5


def test_uniform_baseline_and_validation():
    s = LearningProgressSampler(small_cfg(adaptive=False))
    for values in ([1., 1., 1.], [5., 1., -10.]):
        s.observe(torch.arange(3), torch.tensor(values))
        s.finish_stage()
    torch.testing.assert_close(s.probabilities, torch.full((3,), 1/3, dtype=torch.float64))
    with pytest.raises(ValueError, match='temperature'):
        LearningProgressSampler(small_cfg(temperature=0.))


def test_minimum_samples_and_stage_state_roundtrip():
    cfg = small_cfg()
    cfg.min_episodes_per_task = 2
    s = LearningProgressSampler(cfg)
    s.observe(torch.tensor([0, 0, 1]), torch.tensor([1., 1., 100.]))
    s.finish_stage()
    s.observe(torch.tensor([0, 0, 1, 1]), torch.tensor([3., 3., 200., 200.]))
    s.finish_stage()
    # Task 1 has no valid previous stage, so its apparent jump is not LP.
    torch.testing.assert_close(s.learning_progress, torch.tensor([2., 0., 0.], dtype=torch.float64))
    restored = LearningProgressSampler(cfg)
    restored.load_state_dict(s.state_dict())
    torch.testing.assert_close(restored.probabilities, s.probabilities)
    torch.testing.assert_close(restored.learning_progress, s.learning_progress)
    assert restored.stage == 2


def test_fixed_terrain_columns_and_geometry_levels():
    from Qmini.tasks.quadruped.as2w_lp_cc.mdp.terrains.lp_terrains import LPTaskTerrainGenerator, lp_terrain_cfg
    generator = object.__new__(LPTaskTerrainGenerator)
    generator.cfg = lp_terrain_cfg()
    seen = []
    generator._get_terrain_mesh = lambda difficulty, cfg: (difficulty, None)
    generator._add_sub_terrain = lambda mesh, origin, row, col, cfg: seen.append((mesh, row, col, cfg))
    generator._generate_curriculum_terrains()
    assert len(seen) == 20
    configs = list(generator.cfg.sub_terrains.values())
    for difficulty, row, col, cfg in seen:
        assert cfg is configs[col]
        assert difficulty == row/3


def test_evaluation_config_loader_roundtrip(tmp_path):
    import yaml
    from Qmini.tasks.quadruped.as2w_lp_cc.evaluate import load_config
    values = {'axes': (0., .5), 'joint_ids': slice(None), 'parts': ['hip', 'thigh']}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.dump(values))
    assert load_config(path) == values
