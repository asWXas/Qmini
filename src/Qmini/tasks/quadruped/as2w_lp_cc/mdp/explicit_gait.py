"""兼容旧步态导入路径；实现位于 rewards/gait.py。"""

from .rewards.gait import (
    DiagonalGaitCfg,
    diagonal_gait_targets,
    standing_wheel_contact_penalty,
    diagonal_gait_contact_penalty,
    _gait_tracking_penalties as _gait_tracking_penalties,
    pure_x_terrain_gait_penalty,
)

__all__ = [
    "DiagonalGaitCfg",
    "diagonal_gait_targets",
    "standing_wheel_contact_penalty",
    "diagonal_gait_contact_penalty",
    "pure_x_terrain_gait_penalty",
]
