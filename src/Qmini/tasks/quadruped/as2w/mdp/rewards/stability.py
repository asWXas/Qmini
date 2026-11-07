"""机体高度、重力对齐与关节姿态。"""

import math

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

__all__ = ["base_height_l2", "gravity_z_alignment", "stand_still_joint_deviation", "diagonal_joint_symmetry_l2"]


class base_height_l2(ManagerTermBase):
    """中心九点平均地面上的机身高度误差平方，按 height_std 归一化。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        pattern = sensor.cfg.pattern_cfg
        # 使用实际射线排列生成索引，兼容 xy/yx ordering；只在初始化时计算。
        starts, _ = pattern.func(pattern, device="cpu")
        targets = torch.tensor([-0.2, 0.0, 0.2], dtype=starts.dtype)
        selected = torch.ones(starts.shape[0], dtype=torch.bool)
        for axis in (0, 1):
            selected &= torch.isclose(starts[:, axis, None], targets[None, :], atol=1e-6, rtol=0.0).any(dim=1)
        self._scan_ids = selected.nonzero(as_tuple=True)[0].to(env.device)
        if self._scan_ids.numel() != 9:
            raise ValueError("base_height_l2 requires nine scan points at x/y = {-0.2, 0, 0.2} m")

    def __call__(
        self,
        env,
        target_height: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        height_std: float = 1.0,
    ):
        if not math.isfinite(height_std) or height_std <= 0.0:
            raise ValueError("height_std must be finite and positive")
        heights = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[:, self._scan_ids, 2]
        finite = torch.isfinite(heights)
        # 与原生奖励一致：忽略未命中点，全部未命中时地面高度回退为 0。
        ground_height = heights.masked_fill(~finite, 0.0).sum(dim=1) / finite.sum(dim=1).clamp(min=1)
        base_height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2]
        return ((base_height - ground_height - target_height) / height_std).square()


def gravity_z_alignment(env, std: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):  # noqa: B008
    """重力 Z 分量接近 -1 的指数奖励。"""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("std must be finite and positive")
    gravity_b = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return torch.exp(-((gravity_b[:, 2] + 1.0) / std).square())


def stand_still_joint_deviation(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    stand_still_scale: float = 1.5,
    moving_hip_scale: float = 0.1,
    moving_thigh_scale: float = 0.1,
    moving_calf_scale: float = 0.1,
    command_transition_width: float = 0.1,
):
    """默认姿态平方偏差，按 vx、vy、yaw 指令的最大绝对值平滑混合静止与运动系数。

    过渡区以 command_threshold 为中心，总宽度为 command_transition_width。
    默认 [0.05, 0.15]；两端使用 smoothstep，端点斜率为零。
    """
    if not math.isfinite(command_threshold) or command_threshold <= 0:
        raise ValueError("command_threshold must be finite and positive")
    if not math.isfinite(command_transition_width) or not 0 < command_transition_width <= 2 * command_threshold:
        raise ValueError("command_transition_width must be finite and in (0, 2 * command_threshold]")
    scales = (stand_still_scale, moving_hip_scale, moving_thigh_scale, moving_calf_scale)
    if any(not math.isfinite(scale) or scale < 0 for scale in scales):
        raise ValueError("Joint deviation scales must be finite and non-negative")
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    selected_ids = list(range(len(asset.joint_names)))[joint_ids] if isinstance(joint_ids, slice) else joint_ids
    groups = [asset.joint_names[index].rsplit("_", 1)[-1] for index in selected_ids]
    moving_scales = {"hip": moving_hip_scale, "thigh": moving_thigh_scale, "calf": moving_calf_scale}
    if any(group not in moving_scales for group in groups):
        raise ValueError("Expected only hip, thigh, and calf joints")
    deviation = (asset.data.joint_pos.torch[:, joint_ids] - asset.data.default_joint_pos.torch[:, joint_ids]).square()
    command = env.command_manager.get_command(command_name)
    magnitude = command[:, :3].abs().amax(dim=-1)
    start = command_threshold - 0.5 * command_transition_width
    t = ((magnitude - start) / command_transition_width).clamp(0.0, 1.0)
    blend = t.square() * (3.0 - 2.0 * t)
    standing_penalty = deviation.sum(dim=-1) * stand_still_scale
    moving_penalty = (deviation * deviation.new_tensor([moving_scales[group] for group in groups])).sum(dim=-1)
    return (1.0 - blend) * standing_penalty + blend * moving_penalty


def diagonal_joint_symmetry_l2(
    env,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    invert_hip: bool = True,
    invert_thigh: bool = False,
):
    """对角腿姿态误差平方和；每对只计一次，不包含轮关节。

    两侧顺序均为 hip/thigh/calf：FL、FR 对应 RR、RL。
    invert_hip/invert_thigh 控制对应右侧关节是否反号；calf 始终同号。
    默认 hip 反号、thigh 同号，使用实际关节角度而非默认姿态偏移。
    """
    left = env.scene[left_cfg.name].data.joint_pos.torch[:, left_cfg.joint_ids]
    right = env.scene[right_cfg.name].data.joint_pos.torch[:, right_cfg.joint_ids]
    if left.shape[-1] != 6 or right.shape[-1] != 6:
        raise ValueError("Diagonal symmetry requires two ordered hip/thigh/calf triplets per side")
    signs = left.new_tensor([-1.0 if invert_hip else 1.0, -1.0 if invert_thigh else 1.0, 1.0] * 2)
    return (left - right * signs).square().sum(dim=-1)
