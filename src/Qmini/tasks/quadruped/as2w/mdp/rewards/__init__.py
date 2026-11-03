"""当前 RewardsCfg 使用的自定义奖励及其步态辅助接口。

速度跟踪、速度正则等原生奖励由配置直接引用 Isaac Lab。
"""

from .stability import (
    base_height_l2,
    diagonal_joint_symmetry_l2,
    gravity_z_alignment,
    stand_still_joint_deviation,
)

from .diagnostics import GaitScanTest
from .tracking import ang_vel_z_tracking_error_l2, lin_vel_xy_tracking_error_l2

from .gait import (
    DiagonalGaitCfg,
    diagonal_gait_targets,
    standing_wheel_contact_penalty,
    diagonal_gait_contact_penalty,
    pure_x_terrain_gait_penalty,
)

from .wheels import (
    diagonal_wheel_distance_difference,
)

from .wheel_slip import (
    WheelContactSlip,
)

from .regularization import (
    action_smoothness_l2,
)

from .safety import (
    wheel_lateral_contact_force,
    joint_torque_limits,
    joint_pos_limits,
)

__all__ = [
    "ang_vel_z_tracking_error_l2",
    "lin_vel_xy_tracking_error_l2",
    "GaitScanTest",
    "base_height_l2",
    "diagonal_joint_symmetry_l2",
    "gravity_z_alignment",
    "stand_still_joint_deviation",
    "DiagonalGaitCfg",
    "diagonal_gait_targets",
    "standing_wheel_contact_penalty",
    "diagonal_gait_contact_penalty",
    "pure_x_terrain_gait_penalty",
    "diagonal_wheel_distance_difference",
    "WheelContactSlip",
    "action_smoothness_l2",
    "wheel_lateral_contact_force",
    "joint_torque_limits",
    "joint_pos_limits",
]
