"""Command-range curricula evaluated before completed episode rewards are reset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["command_levels_lin_vel", "command_levels_ang_vel"]


def _command_levels(env, env_ids, reward_term_name, range_multiplier, axes):
    if len(range_multiplier) != 2 or not all(math.isfinite(v) for v in range_multiplier):
        raise ValueError("range_multiplier must contain two finite values")
    if not 0 < range_multiplier[0] <= range_multiplier[1]:
        raise ValueError("Require 0 < initial multiplier <= final multiplier")

    # MultiCriticRewardManager qualifies term names as task/<term>.
    manager = env.reward_manager
    name = reward_term_name
    if name not in manager._episode_sums:
        matches = [key for key in manager._episode_sums if key.rsplit("/", 1)[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one tracking reward matching {name!r}, got {matches}")
        name = matches[0]
    weight = manager.get_term_cfg(name).weight
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Command curriculum requires a positive tracking reward, not an error penalty")

    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    if not hasattr(env, "_as2w_command_curricula"):
        env._as2w_command_curricula = {}
    states = env._as2w_command_curricula
    if axes not in states:
        original = torch.tensor([getattr(ranges, axis) for axis in axes], device=env.device)
        if not torch.isfinite(original).all() or not (original[:, 0] <= original[:, 1]).all():
            raise ValueError("Command ranges must have finite ordered lower/upper bounds")
        states[axes] = {
            "original": original,
            "final": original * range_multiplier[1],
            "last_check": env.common_step_counter,
            "last_seen": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            "score_sum": torch.zeros((), device=env.device),
            "count": 0,
        }
        for axis, bounds in zip(axes, original * range_multiplier[0], strict=True):
            setattr(ranges, axis, tuple(bounds.tolist()))

    state = states[axes]
    step = env.common_step_counter
    ids = torch.arange(env.num_envs, device=env.device)
    ids = ids[env_ids] if env_ids is not None else ids
    ids = ids.unique()
    # Initial and repeated resets must not count empty or duplicate episodes.
    ids = ids[(env.episode_length_buf[ids] > 0) & (state["last_seen"][ids] != step)]
    if ids.numel():
        scores = manager._episode_sums[name][ids] / (env.max_episode_length_s * weight)
        state["score_sum"] += scores.sum()
        state["count"] += ids.numel()
        state["last_seen"][ids] = step

    # Accumulate all reset episodes, including early failures. Check at most once
    # per episode-length interval, even when resets never land on exact multiples.
    if step - state["last_check"] >= env.max_episode_length and state["count"]:
        if state["score_sum"] / state["count"] > 0.8:
            current = torch.tensor([getattr(ranges, axis) for axis in axes], device=env.device)
            expanded = current + current.new_tensor((-0.1, 0.1))
            expanded = expanded.clamp(min=state["final"][:, :1], max=state["final"][:, 1:])
            for axis, bounds in zip(axes, expanded, strict=True):
                setattr(ranges, axis, tuple(bounds.tolist()))
        state["last_check"] = step
        state["score_sum"].zero_()
        state["count"] = 0

    return torch.tensor(getattr(ranges, axes[0])[1], device=env.device)


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """Expand X/Y bounds by 0.1 m/s when mean normalized tracking exceeds 0.8.

    Initialize once from configured ranges. Episode scores use the full episode
    duration, so early terminations reduce success. Return the current X upper
    bound for Curriculum logs. Uses the positive XY exponential tracking reward.
    """
    return _command_levels(env, env_ids, reward_term_name, range_multiplier, ("lin_vel_x", "lin_vel_y"))


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """Expand yaw bounds by 0.1 rad/s using the positive yaw tracking reward."""
    return _command_levels(env, env_ids, reward_term_name, range_multiplier, ("ang_vel_z",))
