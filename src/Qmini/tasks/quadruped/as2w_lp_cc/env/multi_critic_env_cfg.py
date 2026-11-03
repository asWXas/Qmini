"""AS2W environments with three semantic reward groups for multi-head PPO."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .as2w_env_cfg import AS2WFlatEnvCfg, AS2WRoughEnvCfg


REWARD_GROUPS = {
    # 任务：速度跟踪与机体重力对齐。
    "task": (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "gravity_alignment",
    ),
    # 风格：机体运动、步态、轮子布局、动作平滑及能耗。
    "style": (
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "base_height_l2",
        "action_rate_l2",
        "action_smoothness_l2",
        "explicit_diagonal_gait",
        "pure_x_terrain_gait",
        "joint_pose_deviation",
        "diagonal_distance_difference",
        "joint_torques_l2",
        "leg_joint_acc_l2",
        "wheel_joint_acc_l2",
        "wheel_slip",
        "standing_wheel_contact",
    ),
    # 安全：异常接触及关节位置、力矩限位。
    "safety": (
        "undesired_contacts",
        "wheel_stumble",
        "joint_pos_limits",
        "joint_torque_limits",
        "termination_penalty",
    ),
}


def group_rewards(rewards):
    """Preserve tuned terms and reject unassigned terms instead of silently dropping them."""
    terms = {name: term for name, term in vars(rewards).items() if isinstance(term, RewardTermCfg)}
    assigned = [name for names in REWARD_GROUPS.values() for name in names]
    if len(assigned) != len(set(assigned)) or set(terms) - set(assigned):
        raise ValueError(f"Reward grouping contains duplicates or unassigned terms: {set(terms) - set(assigned)}")
    return {group: {name: terms[name] for name in names if name in terms} for group, names in REWARD_GROUPS.items()}


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
