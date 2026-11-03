"""世界坐标下的对角轮距约束。"""

import math

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..terrain_geometry import GridScanEdgeDetector as _GridScanEdgeDetector
from ..terrain_geometry import terrain_step_gate as _terrain_step_gate

__all__ = ["diagonal_wheel_distance_difference"]


class diagonal_wheel_distance_difference(ManagerTermBase):
    """世界坐标 XYZ 下对角轮心距离差惩罚，随地形突变平滑关闭。

    轮心顺序 FL、FR、RL、RR。门控与速度指令无关，使用与纯 vx 步态
    相同的地形检测。未提供扫描时保持原惩罚；无有效扫描三连点时跳过约束。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        height_sensor_cfg = cfg.params.get("height_sensor_cfg")
        self._terrain_edges = None
        if height_sensor_cfg is not None:
            sensor = env.scene.sensors[height_sensor_cfg.name]
            self._terrain_edges = _GridScanEdgeDetector(sensor.cfg.pattern_cfg, env.device)

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        tolerance: float = 0.03,
        height_sensor_cfg: SceneEntityCfg | None = None,
        step_threshold: float = 0.03,
        step_full_height: float = 0.06,
    ):
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        positions = env.scene[asset_cfg.name].data.body_link_pos_w.torch[:, asset_cfg.body_ids]
        if positions.shape[1] != 4:
            raise ValueError("Expected four wheel bodies ordered FL, FR, RL, RR")
        fr_rl = torch.linalg.vector_norm(positions[:, 1] - positions[:, 2], dim=-1)
        fl_rr = torch.linalg.vector_norm(positions[:, 0] - positions[:, 3], dim=-1)
        penalty = ((fr_rl - fl_rr).abs() - tolerance).clamp(min=0.0)
        if height_sensor_cfg is not None:
            hits = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w.torch
            edge, valid = self._terrain_edges(hits)
            gate = _terrain_step_gate(edge, step_threshold, step_full_height)
            penalty = torch.where(valid & (gate < 1.0), penalty * (1.0 - gate), 0.0)
        return penalty
