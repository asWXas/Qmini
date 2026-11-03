"""GR1T environment settings shared with the grouped-reward specialization."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from Qmini.tasks.humanoid.gr1t.env.gr1t_env_cfg import RewardsCfg as FlatRewardsCfg
from Qmini.tasks.humanoid.gr1t.flat_env_cfg import QminiFlatEnvCfg, QminiFlatEnvCfg_PLAY


@configclass
class TaskRewardsCfg:
    """Task: command tracking, upright posture, terrain-relative height, and standing."""

    track_lin_vel_xy: RewardTermCfg | None = None
    track_ang_vel_z: RewardTermCfg | None = None
    alive: RewardTermCfg | None = None
    base_upright: RewardTermCfg | None = None
    torso_upright: RewardTermCfg | None = None
    base_height: RewardTermCfg | None = None
    stand_still: RewardTermCfg | None = None


@configclass
class CoordinationRewardsCfg:
    """Coordination: body-motion damping, lateral sway, gait, footholds, and limb style."""

    base_motion_penalty: RewardTermCfg | None = None
    torso_motion_penalty: RewardTermCfg | None = None
    base_lateral_sway: RewardTermCfg | None = None
    gait: RewardTermCfg | None = None
    foothold_flatness: RewardTermCfg | None = None
    leg_style: RewardTermCfg | None = None
    arm_body_coordination: RewardTermCfg | None = None
    upper_body_style: RewardTermCfg | None = None
    cycle_symmetry: RewardTermCfg | None = None


@configclass
class ConstraintRewardsCfg:
    """Contact, smoothness, energy, actuator, and termination constraints."""

    feet_slide: RewardTermCfg | None = None
    feet_stumble: RewardTermCfg | None = None
    flight_phase: RewardTermCfg | None = None
    action_rate_l2: RewardTermCfg | None = None
    action_smoothness_l2: RewardTermCfg | None = None
    joint_acc: RewardTermCfg | None = None
    joint_torques: RewardTermCfg | None = None
    joint_mechanical_power: RewardTermCfg | None = None
    joint_pos_limits: RewardTermCfg | None = None
    joint_torque_limits: RewardTermCfg | None = None
    undesired_contacts: RewardTermCfg | None = None
    terminated: RewardTermCfg | None = None


@configclass
class GR1TMultiCriticRewardsCfg:
    """Semantic GR1T reward groups in critic-head order."""

    task: TaskRewardsCfg = TaskRewardsCfg()
    coordination: CoordinationRewardsCfg = CoordinationRewardsCfg()
    constraint: ConstraintRewardsCfg = ConstraintRewardsCfg()

    @classmethod
    def from_flat(cls, flat: FlatRewardsCfg) -> GR1TMultiCriticRewardsCfg:
        """Move the already tuned flat reward terms into semantic groups."""
        grouped = cls()
        assigned = set()
        for group_cfg in grouped.__dict__.values():
            for term_name in group_cfg.__dict__:
                if hasattr(flat, term_name):
                    if term_name in assigned:
                        raise ValueError(f"GR1T reward assigned to multiple groups: {term_name}")
                    setattr(group_cfg, term_name, getattr(flat, term_name))
                    assigned.add(term_name)

        active = {name for name, term in vars(flat).items() if isinstance(term, RewardTermCfg) and term.weight != 0.0}
        if active - assigned:
            raise ValueError(f"Unassigned GR1T rewards: {sorted(active - assigned)}")

        return grouped


@configclass
class GR1TMultiCriticFlatEnvCfg(QminiFlatEnvCfg):
    """GR1T flat task with unchanged reward terms reorganized into three groups."""

    def __post_init__(self) -> None:
        # The source GR1T flat config tunes a flat RewardsCfg. Let it finish first,
        # then reorganize those exact term objects without changing functions,
        # parameters, or RewTerm weights.
        self.rewards = FlatRewardsCfg()
        super().__post_init__()
        self.rewards = GR1TMultiCriticRewardsCfg.from_flat(self.rewards)


@configclass
class GR1TMultiCriticFlatEnvCfg_PLAY(QminiFlatEnvCfg_PLAY):
    """Share GR1T playback settings, then group the resulting reward terms."""

    def __post_init__(self) -> None:
        self.rewards = FlatRewardsCfg()
        super().__post_init__()
        self.rewards = GR1TMultiCriticRewardsCfg.from_flat(self.rewards)


__all__ = [
    "CoordinationRewardsCfg",
    "ConstraintRewardsCfg",
    "GR1TMultiCriticFlatEnvCfg",
    "GR1TMultiCriticFlatEnvCfg_PLAY",
    "GR1TMultiCriticRewardsCfg",
    "TaskRewardsCfg",
]
