"""轮子碰撞力与关节力矩限位。"""

import math
import re

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

__all__ = ["wheel_lateral_contact_force", "joint_torque_limits", "joint_pos_limits"]


def wheel_lateral_contact_force(env, sensor_cfg: SceneEntityCfg, force_ratio: float = 3.0, force_limit: float = 180.0):
    """接触合力超限与水平碰撞的连续惩罚，逐轮取较大值后平均，范围 [0, 1]。

    水平力允许值为 max(force_ratio * |Fz|, 0.2 * force_limit)。
    两项超限量均除以 force_limit 并裁剪到 [0, 1]，限制冲击峰值的影响。
    """
    if not math.isfinite(force_limit) or force_limit <= 0:
        raise ValueError("force_limit must be finite and positive")
    if not math.isfinite(force_ratio) or force_ratio <= 0:
        raise ValueError("force_ratio must be finite and positive")
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    magnitude = torch.linalg.vector_norm(forces, dim=-1)
    overload = ((magnitude - force_limit) / force_limit).clamp(0.0, 1.0)
    horizontal_force = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    allowed = (force_ratio * forces[..., 2].abs()).clamp_min(0.2 * force_limit)
    stumble = ((horizontal_force - allowed) / force_limit).clamp(0.0, 1.0)
    return torch.maximum(overload, stumble).mean(dim=1)


def joint_torque_limits(env, asset_cfg: SceneEntityCfg):
    """请求力矩超过关节上限的惩罚；隐式 PD 下使用估算力矩。"""
    asset = env.scene[asset_cfg.name]
    effort = asset.actuators.computed_effort.torch[:, asset_cfg.joint_ids]
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    return torch.clamp(effort.abs() - limits, min=0.0).sum(dim=1)


class joint_pos_limits(ManagerTermBase):
    """按关节名正则指定位置范围（弧度），None 或未匹配时使用原软限位。

    自定义范围直接作为奖励边界，不修改物理关节限位，也不再次乘软限位比例。
    分组在初始化时解析；重叠分组、无匹配分组和非法范围会报错。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        selected = list(range(len(asset.joint_names)))[ids] if isinstance(ids, slice) else list(ids)
        names = [asset.joint_names[i] for i in selected]
        lower = [0.0] * len(names)
        upper = [0.0] * len(names)
        custom = [False] * len(names)
        assigned = set()
        for pattern, bounds in (cfg.params.get("joint_ranges") or {}).items():
            matched = [i for i, name in enumerate(names) if re.fullmatch(pattern, name)]
            if not matched:
                raise ValueError(f"Joint range pattern {pattern!r} matches no selected joints")
            if assigned.intersection(matched):
                raise ValueError(f"Overlapping joint range pattern: {pattern!r}")
            assigned.update(matched)
            if bounds is None:
                continue
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Expected finite (lower, upper) with lower <= upper for {pattern!r}")
            for i in matched:
                lower[i], upper[i] = bounds
                custom[i] = True
        position = asset.data.joint_pos.torch
        self._lower = position.new_tensor(lower)
        self._upper = position.new_tensor(upper)
        self._custom = torch.tensor(custom, device=position.device, dtype=torch.bool)

    def __call__(self, env, asset_cfg: SceneEntityCfg, joint_ranges: dict | None = None):
        asset = env.scene[asset_cfg.name]
        position = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
        lower = torch.where(self._custom, self._lower, limits[..., 0])
        upper = torch.where(self._custom, self._upper, limits[..., 1])
        return ((lower - position).clamp_min(0.0) + (position - upper).clamp_min(0.0)).sum(dim=1)
