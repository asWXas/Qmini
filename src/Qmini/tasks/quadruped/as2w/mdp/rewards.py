"""AS2W 任务的自定义奖励项。"""

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def flat_terrain_four_wheel_contact(
    env,
    height_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    height_threshold: float = 0.03,
    contact_threshold: float = 1.0,
    command_name: str = "base_velocity",
    standing_threshold: float = 0.1,
    pure_vx_tolerance: float = 1e-3,
    vx_threshold: float = 0.5,
    std: float = 0.2,
):
    """站立或平坦低速纯 VX 时按接触比例奖励，其他情况返回 0。"""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and positive")
    heights = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    valid = torch.isfinite(heights)
    highest = heights.masked_fill(~valid, -torch.inf).amax(dim=-1)
    lowest = heights.masked_fill(~valid, torch.inf).amin(dim=-1)
    flat = valid.all(dim=-1) & ((highest - lowest) < height_threshold)
    forces = env.scene.sensors[contact_sensor_cfg.name].data.net_forces_w.torch[:, contact_sensor_cfg.body_ids]
    if forces.shape[1] != 4:
        raise ValueError("Expected exactly four wheel contact bodies")
    contact = torch.linalg.vector_norm(forces, dim=-1) > contact_threshold
    contact_ratio = contact.to(dtype=heights.dtype).sum(dim=-1) / 4.0
    reward = torch.exp(-(1.0 - contact_ratio) / std)
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.vector_norm(command[:, :3], dim=-1) < standing_threshold
    pure_vx = (command[:, 0].abs() < vx_threshold) & (
        command[:, 1:3].abs() <= pure_vx_tolerance
    ).all(dim=-1)
    return torch.where(standing | (flat & pure_vx), reward, 0.0)


def wheel_obstacle_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_names: list[str],
    wheel_radius: float = 0.086,
    clearance: float = 0.02,
    obstacle_threshold: float = 0.03,
):
    """局部地形极差触发，惩罚轮底低于最高地面加安全余量的部分。"""
    wheel_z = env.scene[asset_cfg.name].data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    if wheel_z.shape[1] != len(sensor_names):
        raise ValueError("Wheel bodies and scanners must have matching order and count")
    penalty = torch.zeros_like(wheel_z[:, 0])
    for index, name in enumerate(sensor_names):
        heights = env.scene.sensors[name].data.ray_hits_w.torch[..., 2]
        valid = torch.isfinite(heights)
        highest = heights.masked_fill(~valid, -torch.inf).amax(dim=-1)
        lowest = heights.masked_fill(~valid, torch.inf).amin(dim=-1)
        obstacle = (valid.sum(dim=-1) >= 2) & ((highest - lowest) > obstacle_threshold)
        deficit = (highest + clearance + wheel_radius - wheel_z[:, index]).clamp(min=0.0)
        penalty += torch.where(obstacle, deficit, 0.0)
    return penalty


def gait_air_time_limits(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    air_time_range: tuple[float, float] = (0.15, 0.35),
    command_threshold: float = 0.1,
):
    """横移/转向时，过早落地扣分；悬空超时持续扣分，不在落地时重复扣超时分。"""
    lower, upper = air_time_range
    if not 0.0 <= lower < upper:
        raise ValueError("Expected 0 <= minimum air time < maximum air time")
    sensor = env.scene.sensors[sensor_cfg.name]
    ids = sensor_cfg.body_ids
    last_air = sensor.data.last_air_time.torch[:, ids]
    current_air = sensor.data.current_air_time.torch[:, ids]
    landed = sensor.compute_first_contact(env.step_dt).torch[:, ids]
    completed = (lower - last_air).clamp(min=0.0)
    completed = completed * landed * (last_air > 0.0)
    ongoing = (current_air - upper).clamp(min=0.0)
    command = env.command_manager.get_command(command_name)
    active = torch.linalg.vector_norm(command[:, 1:3], dim=1) > command_threshold
    return torch.where(active, (completed + ongoing).sum(dim=1), 0.0)


def joint_pose_limits(env, hip: tuple[float, float], thigh: tuple[float, float], calf: tuple[float, float]):
    """三组绝对角度范围（下限, 上限），单位 rad。"""
    asset = env.scene["robot"]
    position = asset.data.joint_pos.torch
    penalty = torch.zeros_like(position[:, 0])
    for group, (lower, upper) in (("hip", hip), ("thigh", thigh), ("calf", calf)):
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
            raise ValueError(f"Invalid {group} limits: {(lower, upper)}")
        ids = [i for i, name in enumerate(asset.joint_names) if name.endswith(f"_{group}")]
        if not ids:
            raise ValueError(f"No joints found for {group}")
        q = position[:, ids]
        penalty += ((lower - q).clamp(min=0.0) + (q - upper).clamp(min=0.0)).sum(dim=1)
    return penalty


def stand_still_joint_deviation(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    stand_still_scale: float = 5.0,
    moving_scale: float = 1.0,
):
    """腿部默认姿态 L1 偏差，按完整速度命令模长调整惩罚。"""
    asset = env.scene[asset_cfg.name]
    deviation = (
        (
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        )
        .abs()
        .sum(dim=1)
    )
    command = env.command_manager.get_command(command_name)
    standing = torch.linalg.vector_norm(command, dim=-1) < command_threshold
    scale = torch.where(standing, stand_still_scale, moving_scale)
    return deviation * scale


def diagonal_wheel_distance_difference(env, asset_cfg: SceneEntityCfg, tolerance: float = 0.03):
    """轮心顺序 FL、FR、RL、RR；惩罚两条三维对角线长度差超出容差的部分。"""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    positions = env.scene[asset_cfg.name].data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    if positions.shape[1] != 4:
        raise ValueError("Expected four wheel bodies ordered FL, FR, RL, RR")
    fr_rl = torch.linalg.vector_norm(positions[:, 1] - positions[:, 2], dim=-1)
    fl_rr = torch.linalg.vector_norm(positions[:, 0] - positions[:, 3], dim=-1)
    return ((fr_rl - fl_rr).abs() - tolerance).clamp(min=0.0)


def gravity_z_alignment(env, std: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):  # noqa: B008
    """重力 Z 分量接近 -1 的指数奖励。"""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("std must be finite and positive")
    gravity_b = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return torch.exp(-((gravity_b[:, 2] + 1.0) / std).square())


def joint_torque_limits(env, asset_cfg: SceneEntityCfg):
    """请求力矩超过关节上限的惩罚；隐式 PD 下使用估算力矩。"""
    asset = env.scene[asset_cfg.name]
    effort = asset.actuators.computed_effort.torch[:, asset_cfg.joint_ids]
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    return torch.clamp(effort.abs() - limits, min=0.0).sum(dim=1)


class action_rate_l2_second_order(ManagerTermBase):
    """原始动作二阶差分平方和；每回合前两步不惩罚。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._older_action = torch.zeros_like(env.action_manager.action)
        self._value = torch.zeros_like(env.action_manager.action[:, 0])
        self._last_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._older_action[ids] = 0.0
        self._value[ids] = 0.0

    def __call__(self, env):
        if self._last_step != env.common_step_counter:
            action = env.action_manager.action
            previous = env.action_manager.prev_action
            difference = action - 2.0 * previous + self._older_action
            self._value = difference.square().sum(dim=1)
            self._value.masked_fill_(env.episode_length_buf < 3, 0.0)
            self._older_action.copy_(previous)
            self._last_step = env.common_step_counter
        return self._value.clone()


class GaitReward(ManagerTermBase):
    """对角轮同步步态，仅横移或转向命令启用。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.std = cfg.params["std"]
        self.max_err = cfg.params["max_err"]
        if self.std <= 0 or self.max_err <= 0:
            raise ValueError("std and max_err must be positive")
        self.contact_sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        if not self.contact_sensor.cfg.track_air_time:
            raise ValueError("GaitReward requires track_air_time=True")
        pairs = cfg.params["synced_feet_pair_names"]
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError("Expected two synchronized wheel pairs")
        self.synced_feet_pairs = [self.contact_sensor.find_sensors(pair)[0] for pair in pairs]
        ids = [idx for pair in self.synced_feet_pairs for idx in pair]
        if any(len(pair) != 2 for pair in self.synced_feet_pairs) or len(set(ids)) != 4:
            raise ValueError("Expected four distinct wheel sensors")

    def __call__(
        self,
        env,
        std,
        max_err,
        synced_feet_pair_names,
        sensor_cfg,
        command_name="base_velocity",
        command_threshold=0.1,
    ):
        first, second = self.synced_feet_pairs
        reward = self._sync_reward_func(*first) * self._sync_reward_func(*second)
        for left in first:
            for right in second:
                reward = reward * self._async_reward_func(left, right)
        command = env.command_manager.get_command(command_name)
        active = torch.linalg.vector_norm(command[:, 1:3], dim=1) > command_threshold
        return torch.where(active, reward, 0.0)

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time.torch
        contact_time = self.contact_sensor.data.current_contact_time.torch
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time.torch
        contact_time = self.contact_sensor.data.current_contact_time.torch
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)
