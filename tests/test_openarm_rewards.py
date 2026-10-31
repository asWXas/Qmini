"""Behavioral checks for staged OpenArm rewards without a physics simulator."""

import math
import re
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.managers import RewardManager, TerminationManager

from isaaclab_tasks.core.lift.mdp.commands.pose_commands import ObjectUniformPoseCommand
from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import ObjectUniformPoseCommandCfg

from Qmini.tasks.manipulation.openarm import mdp
from Qmini.tasks.manipulation.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg
from Qmini.tasks.manipulation.openarm.mdp.contacts import object_table_support


class Scene(dict):
    """Minimal scene with per-environment origins."""


def proxy(value):
    return SimpleNamespace(torch=value)


@pytest.fixture
def reward_env():
    count = 4
    origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 2.0, 3.0], [-4.0, 0.0, -2.0], [1.0, 1.0, 0.0]])
    pose = torch.tensor([[0.3, 0.0, 0.055, 0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)
    position = origins + pose[:, :3]
    joints = torch.zeros(count, 9)
    joints[:, 7:] = 0.02
    joint_names = [*(f"openarm_joint{i}" for i in range(1, 8)), "openarm_finger_joint1", "openarm_finger_joint2"]

    def find_joints(patterns, preserve_order=False):
        if isinstance(patterns, str):
            patterns = [patterns]
        indices = [i for i, name in enumerate(joint_names) if any(re.fullmatch(pattern, name) for pattern in patterns)]
        return indices, [joint_names[i] for i in indices]

    robot = SimpleNamespace(
        joint_names=joint_names,
        num_joints=9,
        find_joints=find_joints,
        find_bodies=lambda name: ([0], ["openarm_ee_tcp"]),
        data=SimpleNamespace(
            root_pos_w=proxy(origins.clone()),
            root_quat_w=proxy(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)),
            joint_pos=proxy(joints),
            joint_vel=proxy(torch.zeros_like(joints)),
            joint_acc=proxy(torch.zeros_like(joints)),
            soft_joint_pos_limits=proxy(torch.tensor([-2.0, 2.0]).expand(count, 9, 2).clone()),
            body_link_pos_w=proxy(position[:, None, :].clone()),
            body_link_quat_w=proxy(pose[:, None, 3:7].clone()),
        ),
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=proxy(position.clone()),
            root_quat_w=proxy(pose[:, 3:7].clone()),
            default_root_pose=proxy(pose),
            root_lin_vel_w=proxy(torch.zeros(count, 3)),
            root_ang_vel_w=proxy(torch.zeros(count, 3)),
        )
    )
    scene = Scene(robot=robot, object=obj)
    scene["object_contact"] = SimpleNamespace(
        data=SimpleNamespace(
            normal_force_matrix_w=proxy(torch.zeros(count, 1, 11, 3)),
            net_normal_forces_w=proxy(torch.zeros(count, 1, 3)),
        )
    )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        scene[f"{side}_finger_contact"] = SimpleNamespace(
            data=SimpleNamespace(
                normal_force_matrix_w=proxy(torch.tensor([0.0, sign, 0.0]).expand(count, 1, 1, 3).clone()),
            )
        )
    scene.env_origins = origins
    gripper = SimpleNamespace(raw_actions=-torch.ones(count, 1))
    command = torch.tensor([[0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        scene=scene,
        extras={},
        action_manager=SimpleNamespace(
            get_term=lambda name: gripper,
            action=torch.zeros(count, 8),
            prev_action=torch.zeros(count, 8),
        ),
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    cfg = OpenArmCubeLiftEnvCfg()

    env.sim = SimpleNamespace(is_playing=lambda: True)
    env.termination_manager = TerminationManager({"object_pose_success": cfg.terminations.object_pose_success}, env)
    return env, cfg


def move_object(env, local_position):
    position = env.scene.env_origins + torch.tensor(local_position)
    env.scene["object"].data.root_pos_w.torch[:] = position
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, :] = position


def evaluate(term, env):
    return term.func(env, **term.params)


def test_pose_command_measures_object_instead_of_hand_in_rotated_robot_frame(reward_env):
    env, cfg = reward_env
    command_cfg = cfg.commands.object_pose
    assert isinstance(command_cfg, ObjectUniformPoseCommandCfg)
    assert not command_cfg.position_only
    command_cfg.debug_vis = False
    env.sim = SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None))
    command = ObjectUniformPoseCommand(command_cfg, env)
    command.pose_command_b[:] = env.command_manager.get_command("object_pose")
    s = 2**-0.5
    env.scene["robot"].data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, s, s])
    move_object(env, [0.0, 0.3, 0.2])
    # Deliberately keep the hand away from the goal: only the object should count.
    env.scene["robot"].data.body_link_pos_w.torch[:] += 0.4
    obj = env.scene["object"]
    obj.data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, s, s])
    obj.data.root_pos_w.torch[1, 0] += 0.1
    obj.data.root_quat_w.torch[2] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    obj.data.root_quat_w.torch[3] *= -1
    obj.data.root_link_pose_w = proxy(torch.cat((obj.data.root_pos_w.torch, obj.data.root_quat_w.torch), dim=-1))
    command._update_metrics()
    torch.testing.assert_close(command.metrics["position_error"], torch.tensor([0.0, 0.1, 0.0, 0.0]))
    torch.testing.assert_close(command.metrics["orientation_error"], torch.tensor([0.0, 0.0, math.pi / 2, 0.0]))
    # Both policy and critic receive all seven target pose components.
    env.command_manager.get_command = lambda name: command.command
    for group in (cfg.observations.policy, cfg.observations.critic):
        observation = evaluate(group.target_object_pose, env)
        assert observation.shape == (4, 7)
        torch.testing.assert_close(observation, command.pose_command_b)


def state(env):
    return env.termination_manager.get_term_cfg("object_pose_success").func


def step(env, cfg):
    env.termination_manager.compute()
    return {name: evaluate(term, env) * term.weight * env.step_dt for name, term in vars(cfg.rewards).items()}


def confirm_grasp(env, cfg):
    for _ in range(math.ceil(cfg.terminations.object_pose_success.params["grasp_time"] / env.step_dt)):
        step(env, cfg)
    assert state(env).transporting.all()


def settle(env, cfg):
    for _ in range(math.ceil(cfg.terminations.object_pose_success.params["settle_time"] / env.step_dt)):
        result = step(env, cfg)
    return result


def set_grasp(env, held):
    env.action_manager.get_term("gripper_action").raw_actions[:] = -1 if held else 1


def set_table_support(env, supported=True):
    contact = env.scene["object_contact"].data
    contact.net_normal_forces_w.torch[:, 0] = contact.normal_force_matrix_w.torch[:, 0].sum(dim=1)
    if supported:
        contact.net_normal_forces_w.torch[:, 0, 2] += 1.0


def prepare_table_support(env, cfg):
    set_table_support(env)
    for _ in range(math.ceil(cfg.terminations.object_pose_success.params["support_time"] / env.step_dt) + 1):
        step(env, cfg)


def test_support_requires_external_upward_force_near_table(reward_env):
    env, _ = reward_env
    contact = env.scene["object_contact"].data
    contact.normal_force_matrix_w.torch[:, 0, 0, 2] = 5.0  # Robot supports the cube.
    set_table_support(env, False)
    assert not object_table_support(env).any()
    set_table_support(env)
    assert object_table_support(env).all()
    # Reject the ground, airborne contact, and a purely lateral external force.
    env.scene["object"].data.root_pos_w.torch[1, 2] = env.scene.env_origins[1, 2] - 1.0
    env.scene["object"].data.root_pos_w.torch[2, 2] = env.scene.env_origins[2, 2] + 0.3
    contact.net_normal_forces_w.torch[3, 0] = torch.tensor([2.0, 0.0, 5.0])
    assert object_table_support(env).tolist() == [True, False, False, False]


def test_table_preorientation_is_optional_and_only_pays_signed_progress(reward_env):
    env, cfg = reward_env
    set_grasp(env, False)
    quat = env.scene["object"].data.root_quat_w.torch
    quat[:] = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    prepare_table_support(env, cfg)
    quat[:] = torch.tensor([0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)])
    result = step(env, cfg)
    earned = result["object_orientation_tracking"]
    torch.testing.assert_close(earned, torch.full((4,), math.pi / 2))
    assert not state(env).transporting.any()
    assert result["object_goal_tracking"].count_nonzero() == 0
    assert step(env, cfg)["object_orientation_tracking"].count_nonzero() == 0
    # Turning away with the fingers removed still repays the improvement.
    for side in ("left", "right"):
        env.scene[f"{side}_finger_contact"].data.normal_force_matrix_w.torch.zero_()
    quat[:] = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    torch.testing.assert_close(step(env, cfg)["object_orientation_tracking"], -earned)
    quat[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert step(env, cfg)["object_orientation_tracking"].count_nonzero() == 0  # No remote/passive adjustment income.


def test_supported_setdown_preserves_progress_and_allows_regrasp(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.055])
    prepare_table_support(env, cfg)
    confirm_grasp(env, cfg)
    move_object(env, [0.4, 0.0, 0.055])
    assert (step(env, cfg)["object_goal_tracking"] > 0).all()
    credit = state(env)._transport_credit.clone()
    set_grasp(env, False)
    result = step(env, cfg)
    assert result["grasp_lost"].count_nonzero() == 0
    assert state(env)._placed.all() and not state(env).transporting.any()
    torch.testing.assert_close(state(env)._transport_credit, credit)
    assert not env.termination_manager.dones.any()
    set_grasp(env, True)
    for _ in range(6):
        result = step(env, cfg)
        assert result["grasp_confirmed"].count_nonzero() == 0
        for name in state(env).TRANSPORT_COMPONENTS:
            assert result[name].count_nonzero() == 0
    assert state(env).transporting.all()
    move_object(env, [0.35, 0.0, 0.055])
    assert (step(env, cfg)["object_goal_tracking"] > 0).all()
    assert state(env)._drop_count.count_nonzero() == 0
    assert (state(env)._placement_count == 1).all()
    state(env).reset(torch.tensor([1, 3]))
    assert state(env)._support_steps.tolist() == [5, 0, 5, 0]
    assert state(env)._placement_count.tolist() == [1, 0, 1, 0]


def test_uncontrolled_release_still_counts_as_drop(reward_env):
    env, cfg = reward_env
    prepare_table_support(env, cfg)
    confirm_grasp(env, cfg)
    set_grasp(env, False)
    # No support; excessive speed; lost contacts despite closing; first impact.
    env.scene["object_contact"].data.net_normal_forces_w.torch[0] = 0
    env.scene["object"].data.root_lin_vel_w.torch[1, 0] = 0.2
    env.action_manager.get_term("gripper_action").raw_actions[2] = -1
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[2] = 0
    state(env)._support_steps[3] = 0
    result = step(env, cfg)
    torch.testing.assert_close(result["grasp_lost"], torch.full((4,), -2.0))
    assert not state(env)._placed.any()
    assert not env.termination_manager.dones.any()


def test_repeated_supported_lift_setdown_regrasp_cycles_cannot_farm(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.055])
    prepare_table_support(env, cfg)
    confirm_grasp(env, cfg)
    names = (*state(env).ACQUIRE_COMPONENTS, *state(env).TRANSPORT_COMPONENTS, "grasp_confirmed")
    total = torch.zeros(4)

    def account():
        nonlocal total
        result = step(env, cfg)
        total += sum(result[name] for name in names)
        assert result["grasp_lost"].count_nonzero() == 0

    for _ in range(8):
        set_table_support(env, False)
        move_object(env, [0.4, 0.0, 0.13])
        account()
        move_object(env, [0.5, 0.0, 0.055])
        set_table_support(env)
        for _ in range(6):
            account()
        set_grasp(env, False)
        account()
        for _ in range(3):
            account()  # Waiting at the table cannot earn anything.
        set_grasp(env, True)
        for _ in range(6):
            account()
        torch.testing.assert_close(total, torch.zeros(4), atol=1e-5, rtol=0)


def test_supported_final_placement_needs_prior_grasp_and_exact_object_goal(reward_env):
    env, cfg = reward_env
    env.command_manager.get_command("object_pose")[:, 2] = 0.055
    set_grasp(env, False)
    prepare_table_support(env, cfg)
    assert settle(env, cfg)["object_pose_success"].count_nonzero() == 0
    move_object(env, [0.5, 0.0, 0.055])
    set_grasp(env, True)
    confirm_grasp(env, cfg)
    move_object(env, [0.3, 0.0, 0.055])
    env.scene["object"].data.root_quat_w.torch[1] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    env.scene["object"].data.root_pos_w.torch[2, 0] += 0.05
    set_grasp(env, False)
    total = torch.zeros(4)
    for _ in range(15):
        total += step(env, cfg)["object_pose_success"]
    torch.testing.assert_close(total, torch.tensor([100.0, 0.0, 0.0, 100.0]))
    assert env.termination_manager.dones.tolist() == [True, False, False, True]


def test_aligned_closure_is_rewarded_before_contact_without_open_gripper_barrier(reward_env):
    env, cfg = reward_env
    for side in ("left", "right"):
        env.scene[f"{side}_finger_contact"].data.normal_force_matrix_w.torch[:] = 0
    fingers = env.scene["robot"].data.joint_pos.torch[:, 7:]
    fingers[:] = 0.044
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] += torch.tensor([0.0, 0.01, 0.04, 0.1])
    set_grasp(env, False)
    step(env, cfg)
    # Merely changing the commanded action, without finger motion, pays nothing.
    set_grasp(env, True)
    assert step(env, cfg)["closing_object"].count_nonzero() == 0
    fingers[:] = 0.035
    closed = step(env, cfg)
    assert torch.all(closed["closing_object"][:3] > 0)
    assert closed["closing_object"][3] == 0  # No credit for empty closure far away.
    torch.testing.assert_close(closed["approach_open_gripper"][:2], torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.all(sum(closed[name] for name in state(env).ACQUIRE_COMPONENTS)[:3] > 0)
    assert not state(env).transporting.any()  # Closure is preparation, not a confirmed grasp.
    for name in state(env).TRANSPORT_COMPONENTS:
        assert closed[name].count_nonzero() == 0
    for _ in range(10):
        stationary = step(env, cfg)
        assert all(torch.all(value <= 0) for value in stationary.values())
    # Reopening returns exactly the scored progress, so open/close cycles cannot farm it.
    set_grasp(env, False)
    fingers[:] = 0.044
    reopened = step(env, cfg)
    for name in state(env).ACQUIRE_COMPONENTS:
        torch.testing.assert_close(closed[name] + reopened[name], torch.zeros(4), atol=1e-6, rtol=0)


def test_first_contact_does_not_remove_preparation_score(reward_env):
    env, cfg = reward_env
    set_grasp(env, True)
    for side in ("left", "right"):
        env.scene[f"{side}_finger_contact"].data.normal_force_matrix_w.torch[:] = 0
    step(env, cfg)
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[:, 0, 0, 1] = 1
    result = step(env, cfg)
    torch.testing.assert_close(result["approach_open_gripper"], torch.zeros(4))
    assert torch.all(result["grasping_object"] > 0)
    assert not state(env).transporting.any()


def test_confirmed_grasp_milestone_is_once_per_episode_and_reset_is_local(reward_env):
    env, cfg = reward_env
    count = math.ceil(cfg.terminations.object_pose_success.params["grasp_time"] / env.step_dt)
    total = torch.zeros(4)
    for _ in range(3):
        set_grasp(env, True)
        for _ in range(count):
            total += step(env, cfg)["grasp_confirmed"]
        set_grasp(env, False)
        total += step(env, cfg)["grasp_confirmed"]
    torch.testing.assert_close(total, torch.full((4,), 5.0))
    state(env).reset(torch.tensor([0, 2]))
    assert env.extras["log"]["Metrics/grasp_rate"] == 1.0
    set_grasp(env, True)
    for _ in range(count):
        total += step(env, cfg)["grasp_confirmed"]
    torch.testing.assert_close(total, torch.tensor([10.0, 5.0, 10.0, 5.0]))


def test_acquire_first_and_confirm_continuous_two_finger_grasp(reward_env):
    env, cfg = reward_env
    set_grasp(env, False)
    step(env, cfg)
    move_object(env, [0.4, 0.0, 0.2])
    result = step(env, cfg)
    assert not state(env).transporting.any()
    for name in state(env).TRANSPORT_COMPONENTS:
        assert result[name].count_nonzero() == 0
    set_grasp(env, True)
    count = math.ceil(cfg.terminations.object_pose_success.params["grasp_time"] / env.step_dt)
    for _ in range(count - 1):
        step(env, cfg)
    assert not state(env).transporting.any()
    # Losing one finger resets confirmation instead of accepting intermittent contact.
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[:] = 0
    step(env, cfg)
    assert not state(env).transporting.any()
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[:, 0, 0, 1] = 1
    confirm_grasp(env, cfg)


def test_position_progress_is_signed_and_stationary_or_round_trip_cannot_earn(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.2])
    confirm_grasp(env, cfg)
    move_object(env, [0.4, 0.0, 0.2])
    closer = step(env, cfg)["object_goal_tracking"]
    torch.testing.assert_close(closer, torch.full((4,), 0.1 * cfg.rewards.object_goal_tracking.weight))
    for _ in range(20):
        rewards = step(env, cfg)
        assert all(torch.all(value <= 0) for value in rewards.values())
        assert torch.all(sum(rewards.values()) < 0)
    move_object(env, [0.5, 0.0, 0.2])
    torch.testing.assert_close(step(env, cfg)["object_goal_tracking"], -closer)
    total = torch.zeros(4)
    for _ in range(10):
        move_object(env, [0.4, 0.0, 0.2])
        total += step(env, cfg)["object_goal_tracking"]
        move_object(env, [0.5, 0.0, 0.2])
        total += step(env, cfg)["object_goal_tracking"]
    torch.testing.assert_close(total, torch.zeros(4))


def test_orientation_progress_and_quaternion_sign_invariance(reward_env):
    env, cfg = reward_env
    quats = env.scene["object"].data.root_quat_w.torch

    def rotate(angle):
        quats[:] = torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])

    rotate(math.pi / 2)
    confirm_grasp(env, cfg)
    rotate(math.pi / 4)
    closer = step(env, cfg)["object_orientation_tracking"]
    torch.testing.assert_close(closer, torch.full((4,), math.pi / 2))
    quats *= -1
    torch.testing.assert_close(step(env, cfg)["object_orientation_tracking"], torch.zeros(4))
    rotate(math.pi / 2)
    torch.testing.assert_close(step(env, cfg)["object_orientation_tracking"], -closer)


def test_transport_disables_grasp_shaping_and_drop_returns_to_acquiring(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.2])
    confirm_grasp(env, cfg)
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] += 0.02
    result = step(env, cfg)
    for name in state(env).ACQUIRE_COMPONENTS:
        assert result[name].count_nonzero() == 0
    set_grasp(env, False)
    result = step(env, cfg)
    assert not state(env).transporting.any()
    assert not env.termination_manager.dones.any()  # Dropping on the table is recoverable.
    torch.testing.assert_close(result["grasp_lost"], torch.full((4,), -2.0))
    assert step(env, cfg)["grasp_lost"].count_nonzero() == 0  # One penalty per loss.
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] += 0.1
    step(env, cfg)
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] -= 0.05
    assert torch.all(step(env, cfg)["reaching_object"] > 0)


def test_drop_repays_transport_gains_and_regrasp_does_not_reissue_them(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.1])
    confirm_grasp(env, cfg)
    total = torch.zeros(4)
    for cycle in range(5):
        move_object(env, [0.4, 0.0, 0.2])
        earned = step(env, cfg)
        total += sum(earned[name] for name in state(env).TRANSPORT_COMPONENTS)
        assert torch.all(earned["object_goal_tracking"] > 0)
        assert torch.all(earned["lifting_object"] > 0)
        set_grasp(env, False)
        lost = step(env, cfg)
        total += sum(lost[name] for name in state(env).TRANSPORT_COMPONENTS)
        if cycle == 0:
            for name in state(env).TRANSPORT_COMPONENTS:
                torch.testing.assert_close(earned[name] + lost[name], torch.zeros(4))
        # Free flight or pushing closer to B gives neither progress nor success.
        move_object(env, [0.3, 0.0, 0.2])
        result = settle(env, cfg)
        assert result["object_pose_success"].count_nonzero() == 0
        for name in state(env).TRANSPORT_COMPONENTS:
            assert result[name].count_nonzero() == 0
        move_object(env, [0.5, 0.0, 0.1])
        retreat = step(env, cfg)
        total += sum(retreat[name] for name in state(env).TRANSPORT_COMPONENTS)
        assert (total <= 1e-5).all()
        set_grasp(env, True)
        confirm_grasp(env, cfg)
        result = step(env, cfg)
        for name in state(env).TRANSPORT_COMPONENTS:
            assert result[name].count_nonzero() == 0


def test_repeated_grasp_release_cannot_accumulate_acquisition_reward(reward_env):
    env, cfg = reward_env
    set_grasp(env, False)
    step(env, cfg)
    total = torch.zeros(4)
    count = math.ceil(cfg.terminations.object_pose_success.params["grasp_time"] / env.step_dt)
    for _ in range(10):
        set_grasp(env, True)
        for _ in range(count):
            rewards = step(env, cfg)
            total += sum(rewards[name] for name in state(env).ACQUIRE_COMPONENTS)
        set_grasp(env, False)
        rewards = step(env, cfg)
        total += sum(rewards[name] for name in state(env).ACQUIRE_COMPONENTS)
    torch.testing.assert_close(total, torch.zeros(4), atol=1e-5, rtol=0)


def test_success_checks_held_object_pose_not_tcp_pose(reward_env):
    env, cfg = reward_env
    move_object(env, [0.35, 0.0, 0.2])
    # TCP at B, object still 5 cm away. This can be a valid grasp, but not success.
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] -= 0.05
    confirm_grasp(env, cfg)
    assert settle(env, cfg)["object_pose_success"].count_nonzero() == 0
    move_object(env, [0.3, 0.0, 0.2])
    env.scene["robot"].data.body_link_pos_w.torch[:, 0, 0] += 0.04
    obj = env.scene["object"].data
    obj.root_quat_w.torch[1] *= -1
    obj.root_quat_w.torch[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    obj.root_pos_w.torch[3, 0] -= 0.04  # Also too far from the hand to count as held.
    result = settle(env, cfg)
    torch.testing.assert_close(result["object_pose_success"], torch.tensor([100.0, 100.0, 0.0, 0.0]))
    assert env.termination_manager.dones.tolist() == [True, True, False, False]
    assert not cfg.terminations.object_pose_success.time_out
    assert step(env, cfg)["object_pose_success"].count_nonzero() == 0


def test_success_requires_quiet_safe_settling_and_loss_clears_dwell(reward_env):
    env, cfg = reward_env
    confirm_grasp(env, cfg)
    move_object(env, [0.3, 0.0, 0.2])
    obj = env.scene["object"].data
    obj.root_lin_vel_w.torch[0, 0] = 0.1
    obj.root_ang_vel_w.torch[1, 0] = 1.0
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[2, 0, 0, 1] = 50
    env.action_manager.get_term("gripper_action").raw_actions[3] = 1
    assert settle(env, cfg)["object_pose_success"].count_nonzero() == 0
    assert not env.termination_manager.dones.any()
    obj.root_lin_vel_w.torch[:] = 0
    obj.root_ang_vel_w.torch[:] = 0
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[2, 0, 0, 1] = 1
    set_grasp(env, True)
    # No immediate bonus when speed/force recover or contact returns.
    assert step(env, cfg)["object_pose_success"].count_nonzero() == 0
    for _ in range(30):
        result = step(env, cfg)
    assert state(env)._succeeded.all()
    assert result["object_pose_success"].count_nonzero() == 0


def test_reset_is_per_environment_and_target_change_has_no_windfall(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.2])
    confirm_grasp(env, cfg)
    state(env).reset(torch.tensor([0, 2]))
    move_object(env, [0.4, 0.0, 0.2])
    reward = step(env, cfg)["object_goal_tracking"]
    expected = 0.1 * cfg.rewards.object_goal_tracking.weight
    torch.testing.assert_close(reward, torch.tensor([0.0, expected, 0.0, expected]))
    env.command_manager.get_command("object_pose")[1, 0] += 0.05
    assert step(env, cfg)["object_goal_tracking"].count_nonzero() == 0
    assert not state(env).transporting[0] and state(env).transporting[1]
    state(env).reset()
    assert not state(env).transporting.any()
    assert state(env)._transport_credit.count_nonzero() == 0


def test_goal_change_during_loss_does_not_refund_old_goal_credit(reward_env):
    env, cfg = reward_env
    move_object(env, [0.5, 0.0, 0.2])
    confirm_grasp(env, cfg)
    move_object(env, [0.4, 0.0, 0.2])
    assert (step(env, cfg)["object_goal_tracking"] > 0).all()
    env.command_manager.get_command("object_pose")[:, 0] = 0.1
    set_grasp(env, False)
    result = step(env, cfg)
    assert result["object_goal_tracking"].count_nonzero() == 0
    torch.testing.assert_close(result["grasp_lost"], torch.full((4,), -2.0))


def test_rotated_robot_frame_and_partial_settle_reset_after_drop(reward_env):
    env, cfg = reward_env
    s = 2**-0.5
    env.scene["robot"].data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, s, s])
    confirm_grasp(env, cfg)
    move_object(env, [0.0, 0.3, 0.2])
    env.scene["object"].data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, s, s])
    torch.testing.assert_close(mdp.object_pose_error(env, "object_pose"), torch.zeros(4))
    torch.testing.assert_close(mdp.object_pose_error(env, "object_pose", orientation=True), torch.zeros(4))
    for _ in range(5):
        assert step(env, cfg)["object_pose_success"].count_nonzero() == 0
    env.action_manager.get_term("gripper_action").raw_actions[0] = 1
    step(env, cfg)
    assert state(env)._stable_steps[0] == 0 and state(env)._stable_steps[1] == 6
    env.action_manager.get_term("gripper_action").raw_actions[0] = -1
    for _ in range(5):
        step(env, cfg)
    assert state(env)._succeeded.tolist() == [False, True, True, True]
    assert state(env)._stable_steps[0] == 1
    result = None
    for _ in range(9):
        result = step(env, cfg)
    torch.testing.assert_close(result["object_pose_success"], torch.tensor([100.0, 0.0, 0.0, 0.0]))


def test_table_drop_retries_but_below_workspace_ends_episode(reward_env):
    env, cfg = reward_env
    # The task uses a flat scene; the inherited fall threshold is in world z.
    env.scene.env_origins[:, 2] = 0
    env.scene["robot"].data.root_pos_w.torch[:, 2] = 0
    move_object(env, [0.3, 0.0, 0.055])
    env.max_episode_length = 400
    env.episode_length_buf = torch.zeros(4, dtype=torch.long)
    env.termination_manager = TerminationManager(cfg.terminations, env)
    confirm_grasp(env, cfg)
    set_grasp(env, False)
    move_object(env, [0.3, 0.0, 0.055])
    step(env, cfg)
    assert not env.termination_manager.dones.any()
    assert not state(env).transporting.any()
    env.scene["object"].data.root_pos_w.torch[:, 2] = -0.1
    step(env, cfg)
    assert env.termination_manager.get_term("object_out_of_workspace").all()
    assert env.termination_manager.terminated.all()


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.04])
def test_real_managers_deliver_exactly_one_100_bonus_independent_of_dt(reward_env, step_dt):
    env, cfg = reward_env
    env.step_dt = step_dt
    manager = RewardManager({"success": cfg.rewards.object_pose_success}, env)
    move_object(env, [0.3, 0.0, 0.2])
    total = torch.zeros(4)
    for _ in range(math.ceil(0.5 / step_dt)):
        env.termination_manager.compute()
        total += manager.compute(step_dt)
    torch.testing.assert_close(total, torch.full((4,), 100.0))
    # Neither opposite quaternion signs nor leaving/reentering re-arm success.
    env.command_manager.get_command("object_pose")[:, 3:7] *= -1
    set_grasp(env, False)
    env.termination_manager.compute()
    set_grasp(env, True)
    for _ in range(math.ceil(0.5 / step_dt)):
        env.termination_manager.compute()
        total += manager.compute(step_dt)
    torch.testing.assert_close(total, torch.full((4,), 100.0))
    env.termination_manager.reset(torch.tensor([0, 2]))
    assert env.extras["log"]["Metrics/success_rate"] == 1.0
    for _ in range(math.ceil(0.5 / step_dt)):
        env.termination_manager.compute()
        total += manager.compute(step_dt)
    torch.testing.assert_close(total, torch.tensor([200.0, 100.0, 200.0, 100.0]))


def test_safety_penalties_and_full_reward_manager_are_wired(reward_env):
    env, cfg = reward_env
    manager = RewardManager(cfg.rewards, env)
    env.termination_manager.compute()
    baseline = manager.compute(env.step_dt).clone()
    env.action_manager.action[:] = 1
    robot = env.scene["robot"].data
    robot.joint_vel.torch[:] = 1
    robot.joint_acc.torch[:] = 100
    robot.joint_pos.torch[:, 0] = 2.1
    env.scene["left_finger_contact"].data.normal_force_matrix_w.torch[:, 0, 0, 1] = 50
    env.termination_manager.compute()
    assert torch.all(manager.compute(env.step_dt) < baseline)
    for name in ("action_rate", "joint_vel", "joint_acc", "joint_limits", "excessive_grip_force"):
        assert torch.all(evaluate(getattr(cfg.rewards, name), env) > 0)
        assert getattr(cfg.rewards, name).weight < 0


def test_training_config_uses_fixed_goal_drop_recovery_and_no_continuous_income():
    cfg = OpenArmCubeLiftEnvCfg()
    cfg.validate()
    assert min(cfg.commands.object_pose.resampling_time_range) > cfg.episode_length_s
    assert cfg.terminations.object_pose_success.func is mdp.grasp_transport_success
    assert cfg.terminations.object_out_of_workspace.params["minimum_height"] < 0
    assert cfg.events.reset_object_position.params["pose_range"]["z"] == (0.0, 0.0)
    for term in vars(cfg.rewards).values():
        if term.weight > 0:
            assert term.func is mdp.staged_manipulation_reward
    spec = gym.spec("Qmini-Lift-Cube-OpenArm")
    assert spec.kwargs["env_cfg_entry_point"].startswith("Qmini.tasks.manipulation.openarm.")
