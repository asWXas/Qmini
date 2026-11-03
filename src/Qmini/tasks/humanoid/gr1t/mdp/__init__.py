# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T MDP API.

commands: velocity command sampling; observations: actor/critic inputs;
rewards: task, stability, contacts, and regularization; gait: shared step clock;
style: whole-body coordination; symmetry: policy/action reflection.
Local curriculums are optional and imported explicitly, so the existing Isaac
Lab terrain curriculum remains the default exported implementation.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from isaaclab_tasks.core.velocity.mdp import *

from .commands import *  # noqa: F401, F403
from .gait import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .style import *  # noqa: F401, F403
