import math

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from .env.gr1t_env_cfg import LocomotionVelocityRoughEnvCfg

# =============================================================================
# Flat Environment
# =============================================================================

@configclass
class QminiFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """GR1T2 locomotion on its own mixed terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()
        # Straight legs put the GR1T2 sole about 0.953 m below the base.
        # Start just above the floor so reset does not begin with bent knees.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.96)
        # GR1T 独立地形入口：mdp/terrains/rough.py，由基础场景加载并启用课程。
        self.scene.terrain.terrain_type = "generator"

        # ==================================================================
        # Commands
        # ==================================================================

        # Grouped velocity sampling and heading settings inherit CommandsCfg.

        # Reward tuning lives here; body/joint selections inherit RewardsCfg.
        # Multi-critic reuses these tuned terms through from_flat().
        # ----------------------------------------------------------
        # Task：速度跟踪
        # ----------------------------------------------------------
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.25)
        self.rewards.track_lin_vel_xy.params["command_name"] = "base_velocity"

        self.rewards.track_ang_vel_z.weight = 1.5
        self.rewards.track_ang_vel_z.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z.params["command_name"] = "base_velocity"

        # ----------------------------------------------------------
        # Task：基座/躯干直立与离地高度
        # ----------------------------------------------------------
        self.rewards.base_upright.weight = 0.25
        self.rewards.base_upright.params["std"] = math.sqrt(0.25)

        self.rewards.torso_upright.weight = 1.0
        self.rewards.torso_upright.params["std"] = math.sqrt(0.25)

        self.rewards.base_height.weight = -8.0
        self.rewards.base_height.params["scan_half_extent"] = (0.2, 0.2)
        self.rewards.base_height.params["target_height"] = 0.95

        # ----------------------------------------------------------
        # Task：低速站姿
        # ----------------------------------------------------------
        self.rewards.stand_still.weight = -0.5
        self.rewards.stand_still.params["command_name"] = "base_velocity"
        self.rewards.stand_still.params["gait_reward_name"] = "gait"
        self.rewards.stand_still.params["legs_weight"] = 0.75
        self.rewards.stand_still.params["waist_weight"] = 0.5
        self.rewards.stand_still.params["arms_weight"] = 0.5
        self.rewards.stand_still.params["stand_speed"] = 0.05
        self.rewards.stand_still.params["move_speed"] = 0.20
        self.rewards.stand_still.params["posture_softness"] = 0.05

        # ----------------------------------------------------------
        # Coordination：基座/躯干运动稳定性
        # ----------------------------------------------------------
        self.rewards.base_motion_penalty.weight = -0.10
        self.rewards.base_motion_penalty.params["lin_vel_b"] = 0.8
        self.rewards.base_motion_penalty.params["ang_vel_b"] = 0.2
        self.rewards.torso_motion_penalty.weight = -0.15
        self.rewards.torso_motion_penalty.params["lin_vel_b"] = 0.8
        self.rewards.torso_motion_penalty.params["ang_vel_b"] = 0.2

        # 抑制左右摇摆 / "鸭子步": 横向速度 + roll 角速度 + 侧倾角
        self.rewards.base_lateral_sway.weight = -0.1
        self.rewards.base_lateral_sway.params["command_name"] = "base_velocity"
        self.rewards.base_lateral_sway.params["lin_vel_weight"] = 1.0
        self.rewards.base_lateral_sway.params["roll_rate_weight"] = 0.5
        self.rewards.base_lateral_sway.params["roll_angle_weight"] = 0.5
        self.rewards.base_lateral_sway.params["max_speed"] = 0.10

        # ----------------------------------------------------------
        # Coordination：步态、落脚质量和全身风格
        # ----------------------------------------------------------
        self.rewards.gait.weight = 1.0
        self.rewards.gait.params["command_name"] = "base_velocity"
        self.rewards.gait.params["period"] = 0.4
        self.rewards.gait.params["duty_factor"] = 0.6
        self.rewards.gait.params["target_height"] = 0.06
        self.rewards.gait.params["height_std"] = 0.03
        self.rewards.gait.params["sole_offset"] = 0.054909
        self.rewards.gait.params["command_threshold"] = 0.1
        self.rewards.gait.params["command_transition_width"] = 0.05
        self.rewards.gait.params["phase_transition"] = 0.1
        self.rewards.gait.params["contact_threshold"] = 1.0
        self.rewards.gait.params["sensor_names"] = ["height_scanner_left", "height_scanner_right"]
        self.rewards.foothold_flatness.weight = 1.0
        self.rewards.foothold_flatness.params["sensor_names"] = ["height_scanner_left", "height_scanner_right"]
        self.rewards.foothold_flatness.params["command_name"] = "base_velocity"
        self.rewards.foothold_flatness.params["height_std"] = 0.02
        self.rewards.foothold_flatness.params["min_air_time"] = 0.08
        self.rewards.foothold_flatness.params["contact_threshold"] = 1.0
        self.rewards.foothold_flatness.params["command_threshold"] = 0.1
        self.rewards.foothold_flatness.params["command_transition_width"] = 0.05

        # Independent style weights; timing/clearance remain in gait.
        self.rewards.leg_style.weight = -4.0
        self.rewards.leg_style.params["gait_reward_name"] = "gait"
        self.rewards.leg_style.params["stance_knee_limit"] = 0.12
        self.rewards.leg_style.params["fast_stance_knee_limit"] = 0.45
        self.rewards.leg_style.params["swing_knee_min"] = 0.35
        self.rewards.leg_style.params["fast_swing_knee_min"] = 0.55
        self.rewards.leg_style.params["swing_weight"] = 0.25
        self.rewards.leg_style.params["hip_weight"] = 0.1
        # 髋带动大腿、膝屈曲、踝补偿脚掌俯仰；仅在腾空摆腿中段柔性约束。
        self.rewards.leg_style.params["pitch_weight"] = 0.5
        self.rewards.leg_style.params["pitch_tolerance"] = 0.15
        self.rewards.arm_body_coordination.weight = -1.0
        self.rewards.arm_body_coordination.params["gait_reward_name"] = "gait"
        self.rewards.arm_body_coordination.params["min_arm_scale"] = 0.4
        self.rewards.arm_body_coordination.params["max_arm_scale"] = 1.0
        self.rewards.arm_body_coordination.params["position_tolerance"] = 0.05
        self.rewards.arm_body_coordination.params["velocity_weight"] = 0.15
        self.rewards.arm_body_coordination.params["waist_weight"] = 0.5
        self.rewards.arm_body_coordination.params["waist_yaw_gain"] = 0.2
        self.rewards.arm_body_coordination.params["max_waist_yaw"] = 0.15
        self.rewards.upper_body_style.weight = -2.0
        self.rewards.upper_body_style.params["gait_reward_name"] = "gait"
        self.rewards.upper_body_style.params["min_elbow_offset"] = -0.1
        self.rewards.upper_body_style.params["max_elbow_offset"] = -0.6
        self.rewards.upper_body_style.params["arm_tolerance"] = 0.1
        self.rewards.upper_body_style.params["elbow_tolerance"] = 0.15
        self.rewards.upper_body_style.params["waist_tolerance"] = 0.08
        self.rewards.cycle_symmetry.weight = -1.0
        self.rewards.cycle_symmetry.params["sensor_names"] = ["height_scanner_left", "height_scanner_right"]
        self.rewards.cycle_symmetry.params["gait_reward_name"] = "gait"
        self.rewards.cycle_symmetry.params["posture_tolerance"] = 0.05
        self.rewards.cycle_symmetry.params["amplitude_tolerance"] = 0.05
        self.rewards.cycle_symmetry.params["command_tolerance"] = 0.15
        self.rewards.cycle_symmetry.params["terrain_tolerance"] = 0.05

        # Constraint：支撑与足部接触
        self.rewards.flight_phase.weight = -0.5
        self.rewards.flight_phase.params["command_name"] = "base_velocity"
        self.rewards.flight_phase.params["speed_threshold"] = 0.1
        self.rewards.flight_phase.params["running_speed_range"] = (1.0, 2.0)
        self.rewards.flight_phase.params["air_time_tolerance"] = 0.02
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_stumble.params["threshold"] = 4.0

        # ----------------------------------------------------------
        # Constraint：动作平滑与能耗
        # ----------------------------------------------------------
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness_l2.weight = -0.01
        self.rewards.joint_acc.weight = -2.5e-7
        self.rewards.joint_torques.weight = -1.0e-7
        self.rewards.joint_mechanical_power.weight = -2.0e-4

        # ----------------------------------------------------------
        # Constraint：限位、碰撞与终止
        # ----------------------------------------------------------
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_torque_limits.weight = 0.0

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0

        self.rewards.terminated.weight = -100.0

        # Isaac Lab skips computing zero-weight terms, but still registers and
        # displays them. None keeps them out of RewardManager and its logs.
        for name, term in list(vars(self.rewards).items()):
            if isinstance(term, RewardTermCfg) and term.weight == 0.0:
                setattr(self.rewards, name, None)


# =============================================================================
# Play / Evaluation Environment
# =============================================================================


@configclass
class QminiFlatEnvCfg_PLAY(QminiFlatEnvCfg):
    """Flat-terrain configuration for visualization and evaluation."""

    def __post_init__(self):
        """Post initialization."""

        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.events.external_wrench = None # type: ignore
        self.events.push_robot = None # type: ignore


@configclass
class QminiFlatTCNEnvCfg(QminiFlatEnvCfg):
    """TCN variant using Isaac Lab 3.0's native time-major observation history."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_order = "time"


@configclass
class QminiFlatTCNEnvCfg_PLAY(QminiFlatEnvCfg_PLAY):
    """TCN playback variant using time-major observation history."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_order = "time"
