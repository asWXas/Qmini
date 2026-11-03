# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T reward exports; existing ``mdp.<term>`` and ``mdp.rewards.<term>`` remain available."""

from .common import command_smooth_scale
from .contacts import feet_slide, feet_stumble_penalty, flight_phase_penalty, foothold_flatness
from .legacy import foot_clearance_scan_reward, gait_phase
from .regularization import action_smoothness_l2, joint_mechanical_power_l1
from .stability import base_lateral_sway_penalty, base_motion_penalty, body_motion_penalty
from .task import base_height_penalty, base_mothion_up_exp, stand_still

__all__ = [
    "command_smooth_scale",
    "base_mothion_up_exp",
    "stand_still",
    "base_height_penalty",
    "base_motion_penalty",
    "body_motion_penalty",
    "base_lateral_sway_penalty",
    "feet_slide",
    "feet_stumble_penalty",
    "flight_phase_penalty",
    "foothold_flatness",
    "action_smoothness_l2",
    "joint_mechanical_power_l1",
    "foot_clearance_scan_reward",
    "gait_phase",
]
