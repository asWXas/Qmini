# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import importlib

import gymnasium as gym

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import Qmini.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected_ids = {
        "GR1T-Flat-MultiCritic",
        "GR1T-Flat-MultiCritic-Play",
        "Qmini-Balance-Cartpole",
        "Qmini-Flat",
        "Qmini-Flat-DreamWaQ",
        "Qmini-Flat-DreamWaQ-Play",
        "Qmini-Flat-HIMLoco",
        "Qmini-Flat-HIMLoco-Play",
        "Qmini-Flat-Play",
        "Qmini-Flat-Play-TCN",
        "Qmini-Flat-TCN",
        "Qmini-Rough",
        "Template-Qmini",
        "Template-Qmini-MultiCritic",
        "UMV-Flat",
        "UMV-Flat-Play",
    }

    for task_id in expected_ids:
        spec = gym.spec(task_id)
        assert spec.kwargs["default_agent"] == "rsl_rl"
        assert "-v0" not in task_id


def test_task_configs_use_supported_physics():
    """Legacy tasks use PhysX; Q1 rough locomotion also exposes a Newton preset."""
    prefixes = ("GR1T-", "Qmini-", "Template-", "UMV-")
    for spec in gym.registry.values():
        if not spec.id.startswith(prefixes):
            continue
        module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
        cfg_class = getattr(importlib.import_module(module_name), class_name)
        physics = cfg_class().sim.physics
        if spec.id == "Qmini-Rough":
            assert isinstance(physics.physx, PhysxCfg)
            assert isinstance(physics.newton, NewtonCfg)
            assert isinstance(physics.default, NewtonCfg)
        else:
            assert isinstance(physics, PhysxCfg), spec.id
