"""Exercise DreamWaQ against installed RSL-RL without starting Isaac Sim."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

PACKAGE = Path(__file__).parents[1] / "src/Qmini/tasks/humanoid/gr1t/algo/dream_waq"
SPEC = importlib.util.spec_from_file_location("_test_dreamwaq", PACKAGE / "__init__.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MODEL = importlib.import_module("_test_dreamwaq.model")
PPO = importlib.import_module("_test_dreamwaq.ppo")


def observations(n=4):
    return TensorDict({"policy": torch.randn(n, 10), "critic": torch.randn(n, 6)}, batch_size=[n])


def actor(obs):
    return MODEL.DreamWaQActor(
        obs,
        {"actor": ["policy"]},
        "actor",
        2,
        hidden_dims=[8],
        history_length=2,
        term_widths=[3, 2],
        latent_dim=2,
        encoder_hidden_dims=[8, 4],
        decoder_hidden_dims=[8],
        distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 1.0},
    )


def augment(env, obs=None, actions=None):
    return obs.repeat(2), actions.repeat(2, 1)


def algorithm(obs, symmetry=False):
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=[8], obs_normalization=True)
    storage = RolloutStorage("rl", len(obs), 3, obs, [2])
    return PPO.DreamWaQPPO(
        actor(obs),
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        symmetry_cfg={"env": None, "use_data_augmentation": True, "data_augmentation_func": augment}
        if symmetry
        else None,
    )


def test_auxiliary_ignores_terminal_inputs_and_all_targets():
    obs = observations()
    model = actor(obs)
    next_frame, velocity = torch.randn(4, 5), torch.randn(4, 3)
    valid = torch.tensor([True, False, True, False])
    # Invalid rows must never enter the estimator, even if reset targets are non-finite.
    obs["policy"][~valid] = float("nan")
    next_frame[~valid] = float("nan")
    velocity[~valid] = float("nan")
    torch.manual_seed(12)
    actual = model.auxiliary_loss(obs, next_frame, velocity, valid[:, None])
    torch.manual_seed(12)
    expected = model.auxiliary_loss(obs[valid], next_frame[valid], velocity[valid], torch.ones(2, 1))
    for loss, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(loss, reference)
    sum(actual).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_all_terminal_auxiliary_batch_has_zero_gradients():
    obs = observations()
    model = actor(obs)
    obs["policy"].fill_(float("nan"))
    losses = model.auxiliary_loss(
        obs, torch.full((4, 5), float("nan")), torch.full((4, 3), float("nan")), torch.zeros(4, 1)
    )
    assert all(loss.item() == 0 for loss in losses)
    sum(losses).backward()
    # All ranks need the same parameter gradient structure for distributed reduction.
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.parameters())


@pytest.mark.parametrize("symmetry", [False, True])
@pytest.mark.parametrize("all_done", [False, True])
def test_rollout_reset_boundaries_and_update(symmetry, all_done):
    obs = observations()
    alg = algorithm(obs, symmetry)
    done_sequence = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0]])
    for step in range(3):
        previous_velocity = obs["critic"][:, :3].clone()
        next_obs = observations()
        dones = torch.ones(4, dtype=torch.long) if all_done else done_sequence[step]
        with torch.inference_mode():
            alg.act(obs)
            time_outs = dones * torch.tensor([0, 0, 1, 0])
            alg.process_env_step(next_obs, torch.ones(4), dones, {"time_outs": time_outs})
        stored = alg.storage.observations[step]
        torch.testing.assert_close(stored["dreamwaq_velocity"], previous_velocity)
        torch.testing.assert_close(stored["dreamwaq_next"], MODEL.to_frames(next_obs["policy"], [3, 2], 2)[:, -1])
        assert torch.equal(stored["dreamwaq_valid"].flatten().bool(), ~dones.bool())
        obs = next_obs
    expected_valid = torch.zeros_like(done_sequence, dtype=torch.bool) if all_done else ~done_sequence.bool()
    assert torch.equal(alg.storage.observations["dreamwaq_valid"].squeeze(-1), expected_valid)
    with torch.inference_mode():
        alg.compute_returns(obs)
    before = [p.detach().clone() for p in alg.actor.mlp.parameters()]
    losses = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    if all_done:
        assert all(losses[key] == 0 for key in ("velocity", "reconstruction", "kl"))
    assert any(not torch.equal(old, new) for old, new in zip(before, alg.actor.mlp.parameters(), strict=True))
    assert alg.critic.obs_normalizer.count == 12
    assert alg.storage.step == 0


def test_history_layout_and_jit_export():
    flat = torch.tensor([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    expected = torch.tensor([[[1.0, 2, 3, 7, 8], [4, 5, 6, 9, 10]]])
    torch.testing.assert_close(MODEL.to_frames(flat, [3, 2], 2), expected)
    obs = observations()
    model = actor(obs).eval()
    exported = torch.jit.script(model.as_jit())
    torch.testing.assert_close(exported(obs["policy"]), model(obs))


def test_construct_rejects_incompatible_history_layout():
    obs = observations()
    policy = SimpleNamespace(history_length=2, history_order="time", flatten_history_dim=True)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            observation_manager=SimpleNamespace(
                active_terms={"policy": ["a", "b"]}, group_obs_term_dim={"policy": [(6,), (4,)]}
            ),
            cfg=SimpleNamespace(observations=SimpleNamespace(policy=policy)),
        )
    )
    with pytest.raises(ValueError, match="term"):
        PPO.DreamWaQPPO.construct_algorithm(obs, env, {}, "cpu")
