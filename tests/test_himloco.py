"""HIMLoco migration and episode-boundary regressions without Isaac Sim."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

PACKAGE = Path(__file__).parents[1] / "src/Qmini/tasks/humanoid/gr1t/algo/himloco"
SPEC = importlib.util.spec_from_file_location("_test_himloco", PACKAGE / "__init__.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MODEL = importlib.import_module("_test_himloco.model")
ALGORITHM = importlib.import_module("_test_himloco.ppo")


def observations():
    return TensorDict({"policy": torch.randn(4, 10), "critic": torch.randn(4, 6)}, batch_size=[4])


def augment(env, obs=None, actions=None):
    return obs.repeat(2) if obs is not None else None, actions.repeat(2, 1) if actions is not None else None


def algorithm(symmetry=False, history_order="term", **kwargs):
    env = SimpleNamespace(
        num_envs=4,
        num_actions=2,
        unwrapped=SimpleNamespace(
            observation_manager=SimpleNamespace(
                active_terms={"policy": ["base_ang_vel", "gait_phase"], "critic": ["base_lin_vel"]},
                group_obs_term_dim={"policy": [(6,), (4,)], "critic": [(3,)]},
            ),
            cfg=SimpleNamespace(
                observations=SimpleNamespace(
                    policy=SimpleNamespace(history_length=2, history_order=history_order, flatten_history_dim=True)
                )
            ),
        ),
    )
    cfg = {
        "actor": {
            "class_name": MODEL.HIMActor,
            "hidden_dims": [8],
            "latent_dim": 2,
            "encoder_hidden_dims": [8],
            "target_hidden_dims": [8],
            "num_prototypes": 3,
            "distribution_cfg": {"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 1.0},
        },
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [8], "obs_normalization": True},
        "algorithm": {
            "class_name": ALGORITHM.HIMLocoPPO,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "symmetry_cfg": {"use_data_augmentation": True, "data_augmentation_func": augment} if symmetry else None,
            **kwargs,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 3,
        "multi_gpu": None,
    }
    return ALGORITHM.HIMLocoPPO.construct_algorithm(observations(), env, cfg, "cpu")


def fill_rollout(alg, all_done=False):
    obs = observations()
    done_sequence = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0]])
    if all_done:
        done_sequence.fill_(1)
    for step, dones in enumerate(done_sequence):
        velocity = obs["critic"][:, :3].clone()
        next_obs = observations()
        with torch.inference_mode():
            alg.act(obs)
            alg.process_env_step(next_obs, torch.ones(4), dones, {"time_outs": dones * torch.tensor([0, 0, 1, 0])})
        stored = alg.storage.observations[step]
        torch.testing.assert_close(stored["him_velocity"], velocity)
        torch.testing.assert_close(stored["him_next"], MODEL.to_frames(next_obs["policy"], [3, 2], 2)[:, -1])
        obs = next_obs
    assert torch.equal(alg.storage.observations["him_valid"].squeeze(-1).bool(), ~done_sequence.bool())
    with torch.inference_mode():
        alg.compute_returns(obs)
    # Reset targets must be filtered before loss computation (NaN * 0 is still NaN).
    invalid = done_sequence.bool()
    alg.storage.observations["him_next"][invalid] = float("nan")
    alg.storage.observations["him_velocity"][invalid] = float("nan")


@pytest.mark.parametrize("symmetry", [False, True])
@pytest.mark.parametrize("all_done", [False, True])
def test_boundary_filtering_and_separate_optimizers(monkeypatch, symmetry, all_done):
    alg = algorithm(symmetry)
    fill_rollout(alg, all_done)
    estimator_before = [p.detach().clone() for p in alg.actor.estimator_parameters()]
    policy_before = [p.detach().clone() for p in alg.actor.policy_parameters()]
    parent_update = PPO.update

    def checked_ppo_update(self):
        estimator_after_hio = [p.detach().clone() for p in self.actor.estimator_parameters()]
        changed = any(not torch.equal(a, b) for a, b in zip(estimator_before, estimator_after_hio, strict=True))
        assert changed != all_done
        for old, new in zip(policy_before, self.actor.policy_parameters(), strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        assert all(p.grad is None for p in self.actor.estimator_parameters())
        losses = parent_update(self)
        for old, new in zip(estimator_after_hio, self.actor.estimator_parameters(), strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        assert all(p.grad is None for p in self.actor.estimator_parameters())
        return losses

    monkeypatch.setattr(PPO, "update", checked_ppo_update)
    losses = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    if all_done:
        assert losses["him_velocity"] == losses["him_swap"] == 0
    assert any(not torch.equal(a, b) for a, b in zip(policy_before, alg.actor.policy_parameters(), strict=True))
    assert alg.critic.obs_normalizer.count == 12
    assert alg.storage.step == 0


@pytest.mark.parametrize("distributed", [False, True])
def test_empty_hio_does_not_apply_adam_momentum(monkeypatch, distributed):
    alg = algorithm()
    fill_rollout(alg)
    alg.update()  # Populate Adam's momentum before the all-terminal rollout.
    fill_rollout(alg, all_done=True)
    before = [p.detach().clone() for p in alg.actor.estimator_parameters()]
    steps_before = [state["step"].clone() for state in alg.estimator_optimizer.state.values()]
    if distributed:
        # Simulate a world-size-one collective; exercise the distributed HIO branch only.
        alg.is_multi_gpu = True
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(PPO, "update", lambda self: {})
    losses = alg.update()
    for old, new in zip(before, alg.actor.estimator_parameters(), strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    for step, state in zip(steps_before, alg.estimator_optimizer.state.values(), strict=True):
        torch.testing.assert_close(step, state["step"], rtol=0, atol=0)
    assert losses == {"him_velocity": 0.0, "him_swap": 0.0}


def test_single_valid_sample_only_trains_velocity():
    alg = algorithm()
    model = alg.actor
    velocity, swap = model.hio_loss(observations()[:1], torch.randn(1, 5), torch.randn(1, 3))
    assert swap.item() == 0
    (velocity + swap).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.source_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert model.prototypes.grad is None


def test_empty_worker_participates_when_another_has_valid_samples(monkeypatch):
    alg = algorithm()
    fill_rollout(alg, all_done=True)
    alg.is_multi_gpu = True
    alg.gpu_world_size = 2
    reductions = []

    def other_worker_contributes(tensor, op):
        reductions.append(tensor.clone())
        tensor.add_(1)  # One remote valid sample, then remote nonzero gradients.

    monkeypatch.setattr(torch.distributed, "all_reduce", other_worker_contributes)
    monkeypatch.setattr(PPO, "update", lambda self: {})
    before = [p.detach().clone() for p in alg.actor.estimator_parameters()]
    alg.update()
    assert len(reductions) == 1 + len(before)
    assert all(torch.count_nonzero(tensor) == 0 for tensor in reductions)
    assert all(not torch.equal(a, b) for a, b in zip(before, alg.actor.estimator_parameters(), strict=True))
    assert all(p.grad is None for p in alg.actor.estimator_parameters())


def test_checkpoint_restores_both_optimizers():
    alg = algorithm()
    fill_rollout(alg)
    alg.update()
    state = copy.deepcopy(alg.save())
    restored = algorithm()
    assert restored.load(state, load_cfg=None, strict=True)
    for name, value in alg.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[name])
    for optimizer_name in ("optimizer", "estimator_optimizer"):
        expected = getattr(alg, optimizer_name).state_dict()
        actual = getattr(restored, optimizer_name).state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        for index, entries in expected["state"].items():
            for name, value in entries.items():
                torch.testing.assert_close(value, actual["state"][index][name])


def test_history_layout_and_jit_export():
    model = algorithm().actor.eval()
    flat = torch.tensor([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    expected = torch.tensor([[[1.0, 2, 3, 7, 8], [4, 5, 6, 9, 10]]])
    torch.testing.assert_close(MODEL.to_frames(flat, [3, 2], 2), expected)
    obs = observations()
    exported = torch.jit.script(model.as_jit())
    torch.testing.assert_close(exported(obs["policy"]), model(obs))


def test_construct_rejects_time_major_history():
    with pytest.raises(ValueError, match="term"):
        algorithm(history_order="time")
