"""Geometry and contact checks for rolling-to-climbing reward transitions."""

import math
from types import SimpleNamespace as N

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from Qmini.tasks.quadruped.as2w.mdp.rewards import (
    action_rate_l2_second_order,
    flat_terrain_four_wheel_contact,
    obstacle_stall,
    stand_still_joint_deviation,
    terrain_aware_action_rate_l2,
    wheel_obstacle_clearance,
)
from Qmini.tasks.quadruped.as2w.mdp.terrain_geometry import scan_step_height, terrain_motion_scale

SENSOR_NAMES = ["fl", "fr", "rl", "rr"]
WHEELS = SceneEntityCfg("robot", body_ids=[0, 1, 2, 3])
CONTACT = SceneEntityCfg("contact", body_ids=[0, 1, 2, 3])


class Scene(dict):
    pass


def make_env(heights, commands=None, yaw=0.0):
    """Four identical local 3 by 3 scans with physical world XY coordinates."""
    heights = torch.as_tensor(heights, dtype=torch.float).reshape(-1, 9)
    count = heights.shape[0]
    axis = torch.tensor([-0.1, 0.0, 0.1])
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    hits = torch.stack((x.flatten().expand(count, -1), y.flatten().expand(count, -1), heights), dim=-1)
    positions = torch.zeros(count, 4, 3)
    positions[:, :, 2] = 0.086
    # This Isaac Lab version stores quaternions as (x, y, z, w).
    quaternion = torch.tensor([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    robot = N(
        joint_names=["FL_hip", "FL_thigh", "FL_calf"],
        data=N(
            body_link_pos_w=N(torch=positions),
            root_quat_w=N(torch=quaternion.expand(count, -1).clone()),
            root_lin_vel_w=N(torch=torch.zeros(count, 3)),
            joint_pos=N(torch=torch.ones(count, 3)),
            default_joint_pos=N(torch=torch.zeros(count, 3)),
        ),
    )
    scene = Scene(robot=robot)
    scene.sensors = {name: N(data=N(ray_hits_w=N(torch=hits.clone()))) for name in SENSOR_NAMES}
    scene.sensors["contact"] = N(data=N(net_forces_w=N(torch=torch.zeros(count, 4, 3))))
    commands = torch.tensor([[0.4, 0.0, 0.0]]).expand(count, -1).clone() if commands is None else commands
    return N(
        scene=scene,
        command_manager=N(get_command=lambda name: commands),
        action_manager=N(action=torch.ones(count, 3), prev_action=torch.zeros(count, 3)),
        common_step_counter=3,
        episode_length_buf=torch.full((count,), 3, dtype=torch.long),
    )


def edge_heights(height=0.12, axis=0, sign=1):
    grid = torch.zeros(3, 3)
    index = 2 if sign > 0 else 0
    if axis == 0:
        grid[index, :] = height
    else:
        grid[:, index] = height
    return grid


def clearance(env):
    return wheel_obstacle_clearance(env, WHEELS, SENSOR_NAMES, contact_sensor_cfg=CONTACT)


def stall(env):
    return obstacle_stall(env, WHEELS, SENSOR_NAMES)


def test_planes_preserve_rolling_even_with_large_world_height_range():
    coordinates = torch.tensor([-0.1, 0.0, 0.1])
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    # Includes steep planes, both slope signs, and a nonzero world origin.
    heights = torch.stack((torch.zeros_like(x), x, -x + 0.7 * y, 5.0 + 0.8 * x + 0.4 * y))
    env = make_env(heights)
    for sensor in SENSOR_NAMES:
        step, valid = scan_step_height(env.scene.sensors[sensor].data.ray_hits_w.torch)
        assert valid.all()
        torch.testing.assert_close(step, torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(clearance(env), torch.zeros(4))
    torch.testing.assert_close(stall(env), torch.zeros(4))
    torch.testing.assert_close(terrain_motion_scale(env, SENSOR_NAMES, 0.2), torch.ones(4))


@pytest.mark.parametrize("yaw,command_x,axis,sign", [
    (0.0, 0.4, 0, 1),
    (0.0, -0.4, 0, -1),
    (math.pi / 2.0, 0.4, 1, 1),
    (math.pi / 2.0, -0.4, 1, -1),
])
def test_command_direction_and_heading_select_upcoming_riser(yaw, command_x, axis, sign):
    commands = torch.tensor([[command_x, 0.0, 0.0]])
    env = make_env(edge_heights(axis=axis, sign=sign), commands, yaw)
    torch.testing.assert_close(clearance(env), torch.tensor([4.0 * 0.14]))
    torch.testing.assert_close(stall(env), torch.ones(1))
    # The same visible riser must not punish retreating from it.
    commands[:, 0] *= -1.0
    torch.testing.assert_close(clearance(env), torch.zeros(1))
    torch.testing.assert_close(stall(env), torch.zeros(1))


def test_supports_are_not_lifted_unless_they_push_against_the_riser():
    env = make_env(edge_heights().expand(5, -1, -1))
    forces = env.scene.sensors["contact"].data.net_forces_w.torch
    forces[1:, :, 2] = 100.0
    forces[2, :, 0] = -50.0
    forces[3, :, 0] = -200.0
    forces[4, :, 0] = 200.0
    # Airborne, normal support, partial resistance, blocked, assisting force.
    expected = torch.tensor([1.0, 0.0, 50.0 / 101.0 - 0.3, 1.0, 0.0]) * (4.0 * 0.14)
    torch.testing.assert_close(clearance(env), expected)
    # Clearance is satisfied once the wheel bottoms clear the top plus margin.
    env.scene["robot"].data.body_link_pos_w.torch[:, :, 2] = 0.23
    torch.testing.assert_close(clearance(env), torch.zeros(5))


def test_obstacle_terms_are_inactive_without_translation_command():
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, 0.0]])
    env = make_env(edge_heights().expand(3, -1, -1), commands)
    torch.testing.assert_close(clearance(env), torch.zeros(3))
    torch.testing.assert_close(stall(env), torch.zeros(3))


def test_invalid_scans_never_create_obstacles_or_nan_rewards():
    env = make_env(edge_heights().expand(3, -1, -1))
    for name in SENSOR_NAMES:
        hits = env.scene.sensors[name].data.ray_hits_w.torch
        hits[0, 8, 2] = torch.inf
        hits[1, 4, 2] = torch.nan
        hits[2, :, :] = torch.inf
        step, valid = scan_step_height(hits)
        assert not valid.any()
        torch.testing.assert_close(step, torch.zeros(3))
    torch.testing.assert_close(clearance(env), torch.zeros(3))
    torch.testing.assert_close(stall(env), torch.zeros(3))
    torch.testing.assert_close(terrain_motion_scale(env, SENSOR_NAMES, 0.2), torch.ones(3))


def test_stall_penalty_rewards_progress_at_different_command_speeds():
    commands = torch.tensor([[0.2, 0.0, 0.0]] * 5 + [[0.8, 0.0, 0.0]])
    env = make_env(edge_heights().expand(6, -1, -1), commands)
    env.scene["robot"].data.root_lin_vel_w.torch[:, 0] = torch.tensor([0.0, 0.1, 0.2, 0.4, -0.1, 0.4])
    torch.testing.assert_close(stall(env), torch.tensor([1.0, 0.5, 0.0, 0.0, 1.0, 0.5]))


def test_local_steps_relax_style_costs_continuously():
    heights = torch.stack([edge_heights(height) for height in (0.0, 0.03, 0.055, 0.08, 0.12)])
    env = make_env(heights)
    scale = torch.tensor([1.0, 1.0, 0.6, 0.2, 0.2])
    torch.testing.assert_close(terrain_motion_scale(env, SENSOR_NAMES, 0.2), scale)
    torch.testing.assert_close(terrain_aware_action_rate_l2(env, SENSOR_NAMES), 3.0 * scale)
    term = action_rate_l2_second_order(RewardTermCfg(func=action_rate_l2_second_order, weight=-0.01), env)
    torch.testing.assert_close(term(env, terrain_sensor_names=SENSOR_NAMES), 3.0 * scale)
    deviation = stand_still_joint_deviation(
        env,
        SceneEntityCfg("robot", joint_ids=[0, 1, 2]),
        "base_velocity",
        terrain_sensor_names=SENSOR_NAMES,
        rough_scale=0.2,
    )
    torch.testing.assert_close(deviation, 0.75 * scale)


def test_a_single_blocked_wheel_does_not_require_all_supports_to_lift():
    env = make_env(torch.zeros(3, 3))
    env.scene.sensors["fl"].data.ray_hits_w.torch[0, :, 2] = edge_heights().flatten()
    forces = env.scene.sensors["contact"].data.net_forces_w.torch
    forces[:, :, 2] = 100.0
    forces[:, 0, 0] = -200.0
    torch.testing.assert_close(clearance(env), torch.tensor([0.14]))
    torch.testing.assert_close(stall(env), torch.ones(1))


def test_smooth_slope_contact_preference_releases_at_steps_and_missing_rays():
    coordinates = torch.tensor([-0.1, 0.0, 0.1])
    x, _ = torch.meshgrid(coordinates, coordinates, indexing="ij")
    env = make_env(torch.stack((x, edge_heights(), x)))
    # The broad scan has a large world-Z span in all three environments.
    broad_hits = torch.zeros(3, 2, 3)
    broad_hits[:, 1, 2] = 0.3
    env.scene.sensors["broad"] = N(data=N(ray_hits_w=N(torch=broad_hits)))
    forces = env.scene.sensors["contact"].data.net_forces_w.torch
    forces[:, 1:, 2] = 100.0
    # One incomplete local scan must not masquerade as a smooth patch.
    env.scene.sensors["fl"].data.ray_hits_w.torch[2, 8, 2] = torch.inf
    result = flat_terrain_four_wheel_contact(
        env, SceneEntityCfg("broad"), CONTACT, wheel_sensor_names=SENSOR_NAMES
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))
