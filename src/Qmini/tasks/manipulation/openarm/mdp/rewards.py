# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp, carry, optionally set down and reorient/regrasp, then reach the object goal."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude

from .contacts import finger_object_contact, finger_object_forces, object_table_support
from .state import tcp_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ee_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    position = env.scene["object"].data.root_pos_w.torch
    tcp, _ = tcp_pose_w(env)
    return torch.linalg.vector_norm(position - tcp, dim=-1)


def _lift_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rise above the configured spawn center, accounting for environment origins.

    The reset event does not randomize z. Using the spawn center is conservative
    across size variants and gives zero lift reward at reset, before settling.
    """
    obj = env.scene["object"]
    initial_z = obj.data.default_root_pose.torch[:, 2] + env.scene.env_origins[:, 2]
    return (obj.data.root_pos_w.torch[:, 2] - initial_z).clamp_min(0.0)


def _grasped(env: ManagerBasedRLEnv, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """Require measured contact on BOTH fingers, closing intent and a centered object."""
    fingers = env.scene[gripper_cfg.name].data.joint_pos.torch[:, gripper_cfg.joint_ids]
    closing = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0
    aperture = (fingers.amin(dim=-1) > 0.003) & (fingers.amax(dim=-1) < 0.04)
    return closing & aperture & (_ee_distance(env) < 0.07) & finger_object_contact(env).all(dim=-1)


def _goal_distance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    target, _ = combine_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        env.command_manager.get_command(command_name)[:, :3],
    )
    return torch.linalg.vector_norm(target - env.scene["object"].data.root_pos_w.torch, dim=-1)


def _goal_angle(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    _, target = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3], command[:, 3:7]
    )
    return quat_error_magnitude(env.scene["object"].data.root_quat_w.torch, target)


def object_pose_error(env: ManagerBasedRLEnv, command_name: str, orientation: bool = False) -> torch.Tensor:
    """Negative object error: meters for position, shortest rotation angle for orientation."""
    return -(_goal_angle(env, command_name) if orientation else _goal_distance(env, command_name))


def object_pose_reached(
    env: ManagerBasedRLEnv, command_name: str, position_tolerance: float, orientation_tolerance: float
) -> torch.Tensor:
    """Success depends only on the object's position AND orientation, never the TCP."""
    return (_goal_distance(env, command_name) < position_tolerance) & (
        _goal_angle(env, command_name) < orientation_tolerance
    )


def _command_changed(command: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    # q and -q describe the same goal; changing only the sign must not re-arm rewards.
    return (command[:, :3] != previous[:, :3]).any(dim=-1) | (
        quat_error_magnitude(command[:, 3:7], previous[:, 3:7]) > 1e-6
    )


class grasp_transport_success(ManagerTermBase):
    """Shared per-environment state for grasping, transport rewards and success.

    Isaac Lab evaluates terminations before rewards. This term advances the state
    once per control step; reward terms only read its buffers. Grasp-stage score
    differences telescope across retries because their baselines are frozen during
    transport, not reset. Controlled set-downs preserve transport accounting.
    Regressions always count, including while unheld; positive progress requires
    a confirmed grasp (or supported tabletop orientation adjustment). Thus gating
    and regrasping cannot turn a closed motion cycle into positive progress.
    Uncontrolled losses additionally repay positive net transport credit.
    """

    ACQUIRE_COMPONENTS = ("reaching_object", "approach_open_gripper", "closing_object", "grasping_object")
    TRANSPORT_COMPONENTS = ("lifting_object", "object_goal_tracking", "object_orientation_tracking")

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.transporting = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros_like(self.transporting)
        self._succeeded = torch.zeros_like(self.transporting)
        self._ever_grasped = torch.zeros_like(self.transporting)
        self._grasp_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._stable_steps = torch.zeros_like(self._grasp_steps)
        self._drop_count = torch.zeros_like(self._grasp_steps)
        self._placement_count = torch.zeros_like(self._grasp_steps)
        self._support_steps = torch.zeros_like(self._grasp_steps)
        self._table_adjusting = torch.zeros_like(self.transporting)
        self._placed = torch.zeros_like(self.transporting)
        self._acquire_previous = torch.zeros(env.num_envs, len(self.ACQUIRE_COMPONENTS), device=env.device)
        self._transport_previous = torch.zeros(env.num_envs, len(self.TRANSPORT_COMPONENTS), device=env.device)
        self._transport_credit = torch.zeros_like(self._transport_previous)
        self._last_command = env.command_manager.get_command(cfg.params["command_name"]).clone()
        self.rewards = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in (
                *self.ACQUIRE_COMPONENTS,
                *self.TRANSPORT_COMPONENTS,
                "grasp_confirmed",
                "object_pose_success",
                "grasp_lost",
            )
        }

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        if self._succeeded[env_ids].numel():
            log = self._env.extras.setdefault("log", {})
            log["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
            log["Metrics/grasp_rate"] = self._ever_grasped[env_ids].float().mean().item()
            log["Metrics/drop_count"] = self._drop_count[env_ids].float().mean().item()
            log["Metrics/controlled_setdowns"] = self._placement_count[env_ids].float().mean().item()
        for buffer in (
            self.transporting,
            self._initialized,
            self._succeeded,
            self._ever_grasped,
            self._grasp_steps,
            self._stable_steps,
            self._drop_count,
            self._placement_count,
            self._support_steps,
            self._table_adjusting,
            self._placed,
            self._acquire_previous,
            self._transport_previous,
            self._transport_credit,
            *self.rewards.values(),
        ):
            buffer[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gripper_cfg: SceneEntityCfg,
        command_name: str,
        grasp_time: float,
        lift_height: float,
        position_tolerance: float,
        orientation_tolerance: float,
        linear_speed_tolerance: float,
        angular_speed_tolerance: float,
        settle_time: float,
        force_limit: float,
        support_time: float,
        table_linear_speed_limit: float,
        table_angular_speed_limit: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        changed = _command_changed(command, self._last_command)
        valid = self._initialized & ~changed
        held = _grasped(env, gripper_cfg)
        obj = env.scene["object"].data
        linear_speed = obj.root_lin_vel_w.torch.norm(dim=-1)
        angular_speed = obj.root_ang_vel_w.torch.norm(dim=-1)
        # Positive upward force from the actual tabletop, never just low height
        # or total contact force (which could come from the gripper).
        supported = object_table_support(env)
        table_slow = (linear_speed < table_linear_speed_limit) & (angular_speed < table_angular_speed_limit)
        quiet = (linear_speed < linear_speed_tolerance) & (angular_speed < angular_speed_tolerance)
        support_steps = max(1, math.ceil(support_time / env.step_dt))
        previously_supported = self._support_steps >= support_steps
        self._support_steps = torch.where(supported & table_slow, (self._support_steps + 1).clamp(max=support_steps), 0)
        table_ready = self._support_steps >= support_steps
        table_adjusting = table_ready & finger_object_contact(env).any(dim=-1)
        was_transporting = self.transporting.clone()
        grasp_steps = max(1, math.ceil(grasp_time / env.step_dt))
        self._grasp_steps = torch.where(held, (self._grasp_steps + 1).clamp(max=grasp_steps), 0)
        self.transporting = self._grasp_steps >= grasp_steps
        lost = was_transporting & ~held
        entering = self.transporting & ~was_transporting
        opening = env.action_manager.get_term("gripper_action").raw_actions[:, 0] >= 0
        set_down = lost & opening & previously_supported & table_ready & quiet
        dropped = lost & ~set_down
        self._placed |= set_down
        self._placed &= supported & ~entering & ~dropped
        self._placement_count += set_down.long()
        # An actual confirmed grasp earns a milestone once per episode. Opening
        # and regrasping cannot repeatedly collect it.
        self.rewards["grasp_confirmed"].copy_(entering & ~self._ever_grasped)
        self._ever_grasped |= entering

        # On entering transport, account for the final grasp improvement, then
        # freeze these scores. On a drop, resume against the same saved scores.
        acquiring = ~was_transporting | lost
        acquire_value = torch.stack(
            (
                object_ee_distance(env, 0.2),
                approach_with_open_gripper(env, gripper_cfg),
                gripper_closing_progress(env, gripper_cfg),
                object_grasp_quality(env, gripper_cfg),
            ),
            dim=-1,
        )
        acquire_delta = torch.where((valid & acquiring)[:, None], acquire_value - self._acquire_previous, 0.0)
        self._acquire_previous = torch.where((acquiring | ~valid)[:, None], acquire_value, self._acquire_previous)
        for i, name in enumerate(self.ACQUIRE_COMPONENTS):
            self.rewards[name].copy_(acquire_delta[:, i])

        transport_value = torch.stack(
            (
                (_lift_height(env) / lift_height).clamp(max=1.0),
                -_goal_distance(env, command_name),
                -_goal_angle(env, command_name),
            ),
            dim=-1,
        )
        continuing = was_transporting & self.transporting
        can_improve = continuing[:, None].expand_as(transport_value).clone()
        # Optional preorientation/reorientation on the table. Position/lift
        # progress still requires carrying, so pushing alone cannot solve A->B.
        can_improve[:, 2] |= self._table_adjusting & table_adjusting
        change = transport_value - self._transport_previous
        transport_delta = torch.where(valid[:, None], torch.where(can_improve, change, change.clamp_max(0.0)), 0.0)
        # A changed goal starts a new error baseline, including when grasp loss
        # happens on the same step. Never refund credit measured against old B.
        self._transport_credit[~valid] = 0.0
        self._transport_credit += transport_delta
        # Set-downs/regrasps never reset either ledger. Lowering/reorienting away
        # pays back progress even when the grasp gate is off; repeating the same
        # lift/put-down cycle has no positive net shaping reward.
        transport_delta -= torch.where(dropped[:, None], self._transport_credit.clamp_min(0.0), 0.0)
        self._transport_credit[dropped | ~valid] = 0.0
        self._transport_previous.copy_(transport_value)
        self._table_adjusting.copy_(table_adjusting)
        for i, name in enumerate(self.TRANSPORT_COMPONENTS):
            self.rewards[name].copy_(transport_delta[:, i])
        self.rewards["grasp_lost"].copy_(dropped)
        self._drop_count += dropped.long()

        quiet_at_goal = (
            (self.transporting | (self._placed & table_ready))
            & object_pose_reached(env, command_name, position_tolerance, orientation_tolerance)
            & quiet
            & (finger_object_forces(env).norm(dim=-1).amax(dim=-1) <= force_limit)
        )
        self._stable_steps[changed] = 0
        settle_steps = max(1, math.ceil(settle_time / env.step_dt))
        self._stable_steps = torch.where(quiet_at_goal, (self._stable_steps + 1).clamp(max=settle_steps), 0)
        success = self._stable_steps >= settle_steps
        self.rewards["object_pose_success"].copy_(success & ~self._succeeded)
        self._succeeded |= success
        self._last_command.copy_(command)
        self._initialized[:] = True
        return success


def staged_manipulation_reward(
    env: ManagerBasedRLEnv, component: str, state_term_name: str = "object_pose_success"
) -> torch.Tensor:
    """Read a stage reward after termination computation; cancel the manager's dt scaling."""
    state = env.termination_manager.get_term_cfg(state_term_name).func
    return state.rewards[component] / env.step_dt


def elapsed_time_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A small running cost discourages waiting; its weight is negative."""
    return torch.ones(env.num_envs, device=env.device)


def object_ee_distance(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Dense reaching signal before the first successful grasp."""
    return 1.0 / (1.0 + (_ee_distance(env) / std).square())


def _close_readiness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Close once the object center is within 2 cm of the TCP; fade out by 5 cm."""
    return ((0.05 - _ee_distance(env)) / 0.03).clamp(0.0, 1.0)


def approach_with_open_gripper(env: ManagerBasedRLEnv, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """Open for approach, but remove the cost of closing BEFORE first contact.

    Once aligned, all apertures receive the same preparation score. Previously
    every pre-contact closing step lost open-gripper score while grasp score
    stayed zero, encouraging the policy to hover with its fingers open.
    """
    fingers = env.scene[gripper_cfg.name].data.joint_pos.torch[:, gripper_cfg.joint_ids]
    open_fraction = (fingers.amin(dim=-1) / 0.044).clamp(0.0, 1.0)
    preparation = open_fraction + (1.0 - open_fraction) * _close_readiness(env)
    return preparation * object_ee_distance(env, 0.15)


def gripper_closing_progress(env: ManagerBasedRLEnv, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward physical finger closure near the object, including before contact.

    Used only as a signed difference. A raw close command alone pays nothing;
    empty closure far from the object pays nothing; reopening repays progress.
    """
    fingers = env.scene[gripper_cfg.name].data.joint_pos.torch[:, gripper_cfg.joint_ids]
    closed_fraction = (1.0 - fingers.amax(dim=-1) / 0.044).clamp(0.0, 1.0)
    return _close_readiness(env) * closed_fraction


def object_grasp_quality(env: ManagerBasedRLEnv, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """Shape first contact, then give the larger reward for a two-finger grasp."""
    partial = finger_object_contact(env).float().mean(dim=-1)
    closing = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0
    return (0.25 * partial * closing + 0.75 * _grasped(env, gripper_cfg)) * object_ee_distance(env, 0.05)


def object_lift_progress(env: ManagerBasedRLEnv, gripper_cfg: SceneEntityCfg, lift_height: float) -> torch.Tensor:
    """Smooth lift progress, capped at the requested rise above the reset center."""
    return _grasped(env, gripper_cfg).float() * (_lift_height(env) / lift_height).clamp(max=1.0)


def excessive_grip_force(env: ManagerBasedRLEnv, force_limit: float) -> torch.Tensor:
    """Penalize excessive squeezing, with dimensionless normalization."""
    return ((finger_object_forces(env).norm(dim=-1) / force_limit - 1.0).clamp_min(0.0).square()).sum(dim=-1)


def object_motion_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Discourage fast translation and spinning during transport, including release."""
    obj = env.scene["object"]
    linear = obj.data.root_lin_vel_w.torch.square().sum(dim=-1)
    angular = obj.data.root_ang_vel_w.torch.square().sum(dim=-1)
    return (_lift_height(env) > 0.02).float() * (linear + 0.05 * angular)
