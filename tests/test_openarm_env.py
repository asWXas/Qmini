"""OpenArm joint execution feedback checks without launching the simulator."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import MultiAssetSpawnerCfg

from Qmini.tasks.manipulation.openarm import mdp
from Qmini.tasks.manipulation.openarm.agents.rsl_rl_ppo_cfg import OpenArmLiftCubePPORunnerCfg
from Qmini.tasks.manipulation.openarm.joint_pos_env_cfg import (
    OBJECT_BASE_SCALE,
    OBJECT_SIZE_MULTIPLIERS,
    OPENARM_JOINTS,
    OpenArmCubeLiftEnvCfg,
)


@pytest.mark.parametrize("joint_ids", [slice(None), [2, 0]])
def test_execution_error_uses_applied_targets_and_clears_reset_rows(joint_ids):
    # Already processed arm/hand targets, not raw policy outputs. The last
    # column represents a finger commanded closed while blocked by an object.
    targets = torch.tensor([[2.0, -0.5, 0.0], [3.0, 2.0, 0.044]])
    positions = torch.tensor([[1.0, -0.2, 0.02], [1.5, 1.0, 0.03]])
    original_targets, original_positions = targets.clone(), positions.clone()
    robot = SimpleNamespace(
        actuators=SimpleNamespace(target_command=SimpleNamespace(position=SimpleNamespace(torch=targets))),
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=positions)),
    )
    env = SimpleNamespace(scene={"robot": robot}, episode_length_buf=torch.tensor([4, 0]))
    selector = SceneEntityCfg("robot", joint_ids=joint_ids)

    error = mdp.joint_position_execution_error(env, selector)
    expected = torch.tensor([[1.0, -0.3, -0.02], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(error, expected[:, joint_ids])
    torch.testing.assert_close(targets, original_targets)
    torch.testing.assert_close(positions, original_positions)

    # A newly executed action becomes visible immediately after reset.
    targets[1] = torch.tensor([2.0, 0.5, 0.044])
    env.episode_length_buf[1] = 1
    error = mdp.joint_position_execution_error(env, selector)
    torch.testing.assert_close(error[1], torch.tensor([0.5, -0.5, 0.014])[joint_ids])


def test_execution_feedback_is_wired_to_policy_and_critic():
    cfg = OpenArmCubeLiftEnvCfg()
    runner = OpenArmLiftCubePPORunnerCfg()
    cfg.validate()
    runner.validate()
    term = cfg.observations.policy.joint_pos_error
    assert term.func is mdp.joint_position_execution_error
    assert term.params["asset_cfg"].name == "robot"
    assert term.params["asset_cfg"].joint_names == OPENARM_JOINTS
    assert term.params["asset_cfg"].joint_names == cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert cfg.observations.critic.joint_pos_error.func is mdp.joint_position_execution_error


def test_privileged_observations_are_isolated_from_policy():
    cfg = OpenArmCubeLiftEnvCfg()
    policy, critic = cfg.observations.policy, cfg.observations.critic
    policy_terms = {name for name, value in vars(policy).items() if isinstance(value, ObservationTermCfg)}
    critic_terms = {name for name, value in vars(critic).items() if isinstance(value, ObservationTermCfg)}
    assert policy_terms == {
        "joint_pos",
        "joint_vel",
        "joint_pos_error",
        "object_position",
        "object_orientation",
        "target_object_pose",
        "actions",
    }
    assert critic_terms == policy_terms | {
        "object_lin_vel",
        "object_ang_vel",
        "contact_state",
        "contact_forces",
        "end_effector_pose",
        "end_effector_velocity",
    }
    assert policy.enable_corruption and not critic.enable_corruption
    assert policy.concatenate_terms and critic.concatenate_terms
    for name in policy_terms:
        assert getattr(policy, name) is not getattr(critic, name)
        assert getattr(policy, name).func is getattr(critic, name).func
    for name in ("joint_pos", "joint_vel", "joint_pos_error"):
        assert getattr(critic, name).params["asset_cfg"].joint_names == OPENARM_JOINTS
    critic.joint_pos.params["asset_cfg"].joint_names = ["changed"]
    assert policy.joint_pos.params["asset_cfg"].joint_names == OPENARM_JOINTS


def test_object_privileged_state_uses_robot_root_axes():
    def proxy(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))

    s = 2**-0.5
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=proxy([[0, 0, 0, 1], [0, 0, s, s]])))
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=proxy([[0, 0, 0, 1], [0, 0, 0, -1]]),
            root_lin_vel_w=proxy([[1, 0, 0], [1, 0, 0]]),
            root_ang_vel_w=proxy([[0, 1, 0], [0, 1, 0]]),
        )
    )
    env = SimpleNamespace(scene={"robot": robot, "object": obj})
    params = {"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")}
    # Equivalent quaternion signs are canonicalized; a 90-degree base rotation
    # must also rotate both velocity vectors into the robot's axes.
    torch.testing.assert_close(
        mdp.object_orientation_in_robot_root_frame(env, **params),
        torch.tensor([[0, 0, 0, 1], [0, 0, -s, s]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        mdp.object_linear_velocity_in_robot_root_frame(env, **params),
        torch.tensor([[1, 0, 0], [0, -1, 0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        mdp.object_angular_velocity_in_robot_root_frame(env, **params),
        torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32),
    )
    torch.testing.assert_close(obj.data.root_lin_vel_w.torch, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_rl_games_has_privileged_value_network():
    path = Path(__file__).parents[1] / "src/Qmini/tasks/manipulation/openarm/agents/rl_games_ppo_cfg.yaml"
    cfg = yaml.safe_load(path.read_text())["params"]["config"]
    assert cfg["central_value_config"]["network"]["central_value"]
    assert cfg["central_value_config"]["minibatch_size"] == cfg["minibatch_size"]


def test_size_variants_expand_to_randomized_scene_combinations():
    cfg = OpenArmCubeLiftEnvCfg()
    cfg.validate()
    # Inverted collision groups currently suppress cloned cube/table contacts.
    # The task uses spatially separated workspaces and normal collision detection.
    assert not cfg.scene.filter_collisions
    assert cfg.scene.env_spacing >= 2.5
    spawn = cfg.scene.object.spawn
    assert isinstance(spawn, MultiAssetSpawnerCfg)
    assert len(spawn.assets_cfg) == len(OBJECT_SIZE_MULTIPLIERS)
    for variant, factor in zip(spawn.assets_cfg, OBJECT_SIZE_MULTIPLIERS, strict=True):
        assert variant.scale == pytest.approx((OBJECT_BASE_SCALE * factor,) * 3)
        assert variant.rigid_props.disable_gravity is False

    # Exercise the actual scene's clone-combination expansion without a simulator.
    scene = object.__new__(InteractiveScene)
    scene.cfg = cfg.scene
    scene.cloner_cfg = cfg.scene.clone_cfg
    scene._env_fmt = scene.cloner_cfg.clone_template
    _, worlds, weights = scene._collect_asset_cfgs()
    assert len(worlds) == len(OBJECT_SIZE_MULTIPLIERS)
    np.testing.assert_allclose(weights, np.full(len(worlds), 1 / len(worlds)))
    # All worlds share the robot/table and differ by exactly one object prototype.
    shared = set.intersection(*(set(world) for world in worlds))
    assert all(len(set(world) - shared) == 1 for world in worlds)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        choices = scene.cloner_cfg.clone_strategy(weights, 256)
        np.random.seed(42)
        np.testing.assert_array_equal(choices, scene.cloner_cfg.clone_strategy(weights, 256))
    finally:
        np.random.set_state(state)
    assert set(choices) == set(range(len(worlds)))


def test_object_mass_randomization_updates_inertia_without_compounding():
    cfg = OpenArmCubeLiftEnvCfg()
    event = cfg.events.object_mass
    assert event.mode == "startup"
    assert event.func is mdp.randomize_rigid_body_mass
    mass = torch.tensor([[0.1], [0.2]])
    inertia = torch.eye(3).reshape(1, 1, 9).repeat(2, 1, 1) * 0.001
    initial_mass, initial_inertia = mass.clone(), inertia.clone()

    def set_masses(*, masses, body_ids, env_ids):
        mass[env_ids, body_ids] = masses

    def set_inertias(*, inertias, body_ids, env_ids):
        inertia[env_ids, body_ids] = inertias

    obj = SimpleNamespace(
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=mass), body_inertia=SimpleNamespace(torch=inertia)),
        set_masses_index=set_masses,
        set_inertias_index=set_inertias,
    )
    env = SimpleNamespace(scene={"object": obj})
    term = event.func(event, env)
    for _ in range(2):
        params = {**event.params, "mass_distribution_params": (1.5, 1.5)}
        term(env, None, **params)
        torch.testing.assert_close(mass, initial_mass * 1.5)
        torch.testing.assert_close(inertia, initial_inertia * 1.5)

    material = cfg.events.object_physics_material
    assert material.func is mdp.randomize_rigid_body_material
    assert material.mode == "startup"
    assert material.params["asset_cfg"].name == "object"
    assert material.params["make_consistent"]
    assert material.params["num_buckets"] == 64


@pytest.mark.parametrize("env_ids", [None, slice(1, 3), torch.tensor([2, 0])])
def test_arm_reset_preserves_other_joints_and_synchronizes_targets(env_ids):
    defaults = torch.zeros(3, 9)
    defaults[:, 7:] = 0.044
    positions = defaults.clone()
    velocities = torch.ones_like(defaults)
    targets = torch.ones_like(defaults)
    velocity_targets = torch.ones_like(defaults)
    limits = torch.tensor([-0.02, 0.02]).expand(3, 9, 2).clone()

    def writer(buffer, key):
        def write(**kwargs):
            rows = kwargs["env_ids"]
            if not isinstance(rows, slice):
                rows = rows[:, None]
            buffer[rows, kwargs["joint_ids"]] = kwargs[key]

        return write

    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=defaults),
            soft_joint_pos_limits=SimpleNamespace(torch=limits),
        ),
        write_joint_position_to_sim_index=writer(positions, "position"),
        write_joint_velocity_to_sim_index=writer(velocities, "velocity"),
        set_joint_position_target_index=writer(targets, "target"),
        set_joint_velocity_target_index=writer(velocity_targets, "target"),
    )
    selector = SceneEntityCfg("robot", joint_ids=list(range(7)))
    env = SimpleNamespace(scene={"robot": robot})
    mdp.reset_arm_joints_near_default(env, env_ids, (-0.05, 0.05), selector)
    rows = slice(None) if env_ids is None else env_ids
    assert (positions[rows, :7].abs() <= 0.02).all()
    assert positions[rows, :7].abs().sum() > 0
    torch.testing.assert_close(targets[rows, :7], positions[rows, :7])
    assert (velocities[rows, :7] == 0).all()
    assert (velocity_targets[rows, :7] == 0).all()
    torch.testing.assert_close(positions[:, 7:], defaults[:, 7:])
    assert (defaults[:, :7] == 0).all()
    if env_ids is not None:
        untouched = 0 if isinstance(env_ids, slice) else 1
        torch.testing.assert_close(positions[untouched], defaults[untouched])
        assert (velocities[untouched] == 1).all()


def test_arm_reset_runs_after_default_reset_and_selects_only_arm():
    cfg = OpenArmCubeLiftEnvCfg()
    event = cfg.events.reset_arm_joints
    assert list(vars(cfg.events)).index("reset_all") < list(vars(cfg.events)).index("reset_arm_joints")
    assert cfg.events.reset_all.params["reset_joint_targets"]
    assert event.func is mdp.reset_arm_joints_near_default
    assert event.mode == "reset"
    assert event.params["asset_cfg"].joint_names == ["openarm_joint[1-7]"]


def test_critic_tcp_pose_and_velocity_use_base_axes_and_tcp_origin():
    def proxy(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))

    s = 2**-0.5
    robot = SimpleNamespace(
        find_bodies=lambda name: ([0], ["openarm_ee_tcp"]),
        data=SimpleNamespace(
            root_pos_w=proxy([[0, 0, 0], [4, 5, 0]]),
            root_quat_w=proxy([[0, 0, 0, 1], [0, 0, s, s]]),
            body_link_lin_vel_w=proxy([[[1, 0, 0]], [[1, 0, 0]]]),
            body_link_ang_vel_w=proxy([[[0, 1, 0]], [[0, 1, 0]]]),
            body_link_pos_w=proxy([[[1, 0, 0]], [[5, 5, 0]]]),
            body_link_quat_w=proxy([[[0, 0, 0, 1]], [[0, 0, 0, 1]]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})
    torch.testing.assert_close(
        mdp.end_effector_pose_in_robot_root_frame(env),
        torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 1],
                [0, -1, 0, 0, 0, -s, s],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        mdp.end_effector_velocity_in_robot_root_frame(env, SceneEntityCfg("robot", body_ids=[0])),
        torch.tensor([[1, 0, 0, 0, 1, 0], [0, -1, 0, 1, 0, 0]], dtype=torch.float32),
    )


def test_contact_flags_detect_support_and_opposing_finger_forces():
    def proxy(value):
        return SimpleNamespace(torch=value)

    points = torch.full((3, 1, 1, 3), float("nan"))
    points[1] = 0  # Opposing contacts can cancel net force but still have a point.
    object_force = torch.zeros(3, 1, 3)
    object_force[0, 0, 2] = 1  # Table support without a robot contact point.
    finger_force = torch.zeros(3, 1, 1, 3)
    finger_force[1, 0, 0, 0] = 2
    scene = {
        "object_contact": SimpleNamespace(
            data=SimpleNamespace(contact_pos_w=proxy(points), net_forces_w=proxy(object_force))
        ),
        "left_finger_contact": SimpleNamespace(
            data=SimpleNamespace(normal_force_matrix_w=proxy(finger_force), net_forces_w=proxy(finger_force[:, 0]))
        ),
        "right_finger_contact": SimpleNamespace(
            data=SimpleNamespace(normal_force_matrix_w=proxy(-finger_force), net_forces_w=proxy(-finger_force[:, 0]))
        ),
        "robot": SimpleNamespace(
            data=SimpleNamespace(root_quat_w=proxy(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)))
        ),
    }
    env = SimpleNamespace(scene=scene, num_envs=3)
    torch.testing.assert_close(
        mdp.object_and_gripper_contacts(env),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    forces = mdp.object_and_gripper_contact_forces(env)
    assert forces.shape == (3, 15)
    torch.testing.assert_close(
        forces[1], torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0])
    )
