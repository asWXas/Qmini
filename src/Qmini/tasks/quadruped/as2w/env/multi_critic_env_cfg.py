"""AS2W environments with three semantic reward groups for multi-head PPO."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .as2w_env_cfg import AS2WFlatEnvCfg, AS2WRoughEnvCfg


REWARD_GROUPS = {
    # 任务：速度跟踪与机体重力对齐。
    "task": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "gravity_z_alignment"),
    # 风格：机体运动、动作平滑、步态、轮子布局及能耗。
    "style": (
        "lin_vel_z_l2", "ang_vel_xy_l2", "action_rate_l2", "action_rate_l2_second_order",
        "feet_air_time", "air_time_variance_penalty", "stand_still_joint_deviation", "wheel_obstacle_clearance",
        "diagonal_wheel_distance_difference", "dof_torques_l2", "leg_acc_l2", "wheel_acc_l2",
        "flat_four_wheel_contact", "turning_lateral_gait", "wheel_contact_slip",
    ),
    # 安全：异常接触及关节位置、力矩限位；姿态限位重新启用时也归入此组。
    "safety": ("undesired_contacts", "dof_pos_limits", "joint_pose_limits", "dof_torque_limits", "terminated"),
}


def group_rewards(rewards):
    """Preserve tuned terms and reject unassigned terms instead of silently dropping them."""
    terms = {name: term for name, term in vars(rewards).items() if isinstance(term, RewardTermCfg)}
    assigned = [name for names in REWARD_GROUPS.values() for name in names]
    if len(assigned) != len(set(assigned)) or set(terms) - set(assigned):
        raise ValueError(f"Reward grouping contains duplicates or unassigned terms: {set(terms) - set(assigned)}")
    return {group: {name: terms[name] for name in names if name in terms}
            for group, names in REWARD_GROUPS.items()}


@configclass
class AS2WRoughMultiCriticEnvCfg(AS2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = group_rewards(self.rewards)


@configclass
class AS2WFlatMultiCriticEnvCfg(AS2WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = group_rewards(self.rewards)
