"""Q1 asymmetric observations and Isaac Lab 3.0 disturbance lifecycle checks."""

import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
import warp as wp

from isaaclab.envs import mdp as native_mdp
from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.wrench_composer import WrenchComposer
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from Qmini.assets.qmini import JOINT_NAMES
from Qmini.tasks.humanoid.qmini.agents.rsl_rl_ppo_cfg import QminiRoughPPORunnerCfg
from Qmini.tasks.humanoid.qmini.env.env import LocomotionVelocityRoughEnvCfg, RoughPhysicsCfg
from Qmini.tasks.humanoid.qmini.mdp.disturbances import (
    base_com_offset_observation,
    base_height_above_ground,
    base_mass_observation,
    base_wrench_observation,
    body_com_observation,
    body_mass_observation,
    feet_air_time,
    feet_contact_state,
    feet_friction,
)
from Qmini.tasks.humanoid.qmini.mdp.observations import joint_position_execution_error


def proxy(value):
    return SimpleNamespace(torch=value)


def test_config_registration_and_observation_isolation():
    cfg = LocomotionVelocityRoughEnvCfg()
    cfg.validate()
    runner = QminiRoughPPORunnerCfg()
    runner.validate()
    assert gym.spec("Qmini-Rough").entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    policy_terms = {
        key for key, value in vars(cfg.observations.policy).items() if isinstance(value, ObservationTermCfg)
    }
    critic_terms = {
        key for key, value in vars(cfg.observations.critic).items() if isinstance(value, ObservationTermCfg)
    }
    assert policy_terms == {
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    }
    assert {
        "joint_pos_error",
        "base_wrench",
        "base_mass_scale",
        "base_com_offset",
        "feet_contact",
        "feet_air_time",
        "feet_static_friction",
        "feet_dynamic_friction",
        "height_scan",
        "base_lin_vel",
    } <= critic_terms
    assert "push" not in critic_terms
    assert "pd_gains" not in critic_terms
    for event in vars(cfg.events).values():
        if isinstance(event, EventTermCfg):
            assert event.func is getattr(native_mdp, event.func.__name__)
    assert cfg.events.pd_gains.func is native_mdp.randomize_actuator_gains
    assert cfg.events.pd_gains.params["operation"] == "scale"
    assert cfg.observations.policy.history_length == 5
    assert cfg.observations.policy.enable_corruption
    assert cfg.observations.critic.history_length == 0
    assert not cfg.observations.critic.enable_corruption
    assert cfg.actions.joint_pos.joint_names == JOINT_NAMES
    assert cfg.actions.joint_pos.preserve_order
    assert isinstance(RoughPhysicsCfg().physx, PhysxCfg)
    assert isinstance(RoughPhysicsCfg().newton, NewtonCfg)


def test_position_execution_error_uses_command_order_and_masks_reset_rows():
    # The command buffer already contains scaled/offset/clipped position targets.
    targets = torch.tensor([[0.4, -0.2, 0.8], [1.0, 2.0, 3.0]])
    positions = torch.tensor([[0.1, -0.1, 0.5], [-0.5, 0.4, 0.2]])
    asset = SimpleNamespace(
        actuators=SimpleNamespace(target_command=SimpleNamespace(position=proxy(targets))),
        data=SimpleNamespace(joint_pos=proxy(positions)),
    )
    env = SimpleNamespace(scene={"robot": asset}, episode_length_buf=torch.tensor([3, 0]))
    selector = SceneEntityCfg("robot", joint_ids=[2, 0, 1])
    torch.testing.assert_close(
        joint_position_execution_error(env, selector), torch.tensor([[0.3, 0.3, -0.1], [0.0, 0.0, 0.0]])
    )
    # First executed action after reset must become observable immediately.
    targets[1] = torch.tensor([-0.3, 0.5, 0.6])
    env.episode_length_buf[1] = 1
    torch.testing.assert_close(joint_position_execution_error(env, selector)[1], torch.tensor([0.4, 0.2, 0.1]))
    torch.testing.assert_close(targets[0], torch.tensor([0.4, -0.2, 0.8]))
    torch.testing.assert_close(asset.data.joint_pos.torch, positions)


def test_scene_names_match_urdf():
    cfg = LocomotionVelocityRoughEnvCfg()
    root = ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
    body_names = {link.get("name") for link in root.findall("link")}
    for group in (cfg.observations.policy, cfg.observations.critic, cfg.events, cfg.rewards, cfg.terminations):
        for term in vars(group).values():
            for selector in getattr(term, "params", {}).values():
                if not isinstance(selector, SceneEntityCfg) or not selector.body_names:
                    continue
                patterns = [selector.body_names] if isinstance(selector.body_names, str) else selector.body_names
                assert all(any(re.fullmatch(pattern, name) for name in body_names) for pattern in patterns)
    assert cfg.scene.height_scanner.prim_path.endswith("/base_link")
    contact_pattern = cfg.scene.contact_forces.prim_path.rsplit("/", 1)[1]
    for link in root.findall("link"):
        if re.fullmatch(contact_pattern, link.get("name")):
            assert link.find("collision") is not None


def test_play_disables_pushes_and_curriculum_without_mutating_training():
    train = LocomotionVelocityRoughEnvCfg()
    play = LocomotionVelocityRoughEnvCfg()
    play.play_mode()
    assert play.events.base_external_force_torque is None
    assert play.events.push_robot is None
    assert play.curriculum.terrain_levels is None
    assert not play.scene.terrain.terrain_generator.curriculum
    assert not play.observations.policy.enable_corruption
    assert train.events.base_external_force_torque is not None
    assert train.curriculum.terrain_levels is not None


@pytest.fixture
def wrench_env(tmp_path):
    wp.config.kernel_cache_dir = str(tmp_path / "warp")
    asset = SimpleNamespace(num_instances=3, num_bodies=2, device="cpu")
    asset.permanent_wrench_composer = WrenchComposer(asset)
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset})
    cfg = LocomotionVelocityRoughEnvCfg().events.base_external_force_torque
    cfg.params["asset_cfg"].body_ids = [1]
    cfg.params["force_range"] = (3.0, 3.0)
    cfg.params["torque_range"] = (0.5, 0.5)
    return env, cfg


def test_native_wrench_observation_reads_persistent_buffer(wrench_env):
    env, cfg = wrench_env
    selector = cfg.params["asset_cfg"]
    assert torch.count_nonzero(base_wrench_observation(env, selector)) == 0
    cfg.func(env, torch.tensor([0, 1]), **cfg.params)
    expected = torch.tensor([[3.0, 3.0, 3.0, 0.5, 0.5, 0.5]] * 2 + [[0.0] * 6])
    torch.testing.assert_close(base_wrench_observation(env, selector), expected)
    composer = env.scene["robot"].permanent_wrench_composer
    assert torch.count_nonzero(wp.to_torch(composer.local_force_b)[:, 0]) == 0
    # Native events persist; there is no synthetic pulse timer in the observation.
    torch.testing.assert_close(base_wrench_observation(env, selector), expected)
    composer.reset()
    assert torch.count_nonzero(base_wrench_observation(env, selector)) == 0


@pytest.mark.parametrize("ids", [torch.tensor([0], dtype=torch.int32), slice(0, 1)])
def test_native_composer_partial_reset_clears_only_reset_envs(wrench_env, ids):
    env, cfg = wrench_env
    cfg.func(env, None, **cfg.params)
    env.scene["robot"].permanent_wrench_composer.reset(ids)
    observed = base_wrench_observation(env, cfg.params["asset_cfg"])
    assert torch.count_nonzero(observed[0]) == 0
    torch.testing.assert_close(observed[1:, :3], torch.full((2, 3), 3.0))


def test_privileged_mass_com_and_terrain_height():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=proxy(torch.tensor([[4.0, 1.0], [5.0, 1.0]])),
            body_com_pos_b=proxy(torch.zeros(2, 2, 3)),
            root_pos_w=proxy(torch.tensor([[0.0, 0.0, 1.45], [0.0, 0.0, 2.45]])),
        )
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    hits = torch.zeros(2, 3, 3)
    hits[0, :, 2] = torch.tensor([1.0, 1.0, float("inf")])
    hits[1, :, 2] = float("inf")
    scene.sensors = {"height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=proxy(hits)))}
    env = SimpleNamespace(scene=scene, num_envs=2)
    selector = SceneEntityCfg("robot", body_ids=[0])
    torch.testing.assert_close(body_mass_observation(env, selector), torch.tensor([[4.0], [5.0]]))
    assert body_com_observation(env, selector).shape == (2, 3)
    torch.testing.assert_close(
        base_height_above_ground(env, SceneEntityCfg("height_scanner")), torch.full((2, 1), 0.45)
    )


@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_material_observation_reads_applied_values(backend):
    materials = torch.tensor([[[0.7, 0.6, 0.1]], [[1.1, 0.9, 0.0]]])
    if backend == "physx":
        view = SimpleNamespace(get_material_properties=lambda: wp.from_torch(materials))
        physics = PhysxCfg()
        expected = materials.flatten(1)
    else:

        def attribute(name, model):
            index = 0 if name == "shape_material_mu" else 2
            return wp.from_torch(materials[:, None, :, index].contiguous())

        view = SimpleNamespace(get_attribute=attribute)
        physics = NewtonCfg()
        expected = materials[:, :, [0, 0, 2]].flatten(1)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=physics)),
        sim=SimpleNamespace(backend_manager=SimpleNamespace(get_model=lambda: None)),
    )
    term = feet_friction.__new__(feet_friction)
    term.asset = SimpleNamespace(root_view=view)
    torch.testing.assert_close(term._read_materials(env), expected)


def test_feet_contact_and_air_time_preserve_left_right_order():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w=proxy(torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.5]]])),
            current_air_time=proxy(torch.tensor([[0.3, 0.0, 0.1]])),
        )
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    # Sensor order is right, left, other; selected output must be left, right.
    selected = SceneEntityCfg("contact_forces", body_ids=[1, 0])
    torch.testing.assert_close(feet_contact_state(env, selected), torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(feet_air_time(env, selected), torch.tensor([[0.0, 0.3]]))


@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_foot_friction_maps_public_bodies_to_backend_shapes(backend):
    term = feet_friction.__new__(feet_friction)
    materials = torch.tensor([[[0.7, 0.6, 0.0], [0.9, 0.8, 0.0], [1.1, 1.0, 0.0]]])
    term._body_shapes = None
    if backend == "physx":
        view = SimpleNamespace(
            link_paths=[["/right", "/left"]], get_material_properties=lambda: wp.from_torch(materials)
        )
        shapes = SimpleNamespace(
            create_rigid_body_view=lambda path: SimpleNamespace(max_shapes=1 if path == "/right" else 2)
        )
        physics = PhysxCfg()
        expected = torch.tensor([[[1.0, 0.9], [0.7, 0.6]]])
    else:
        # Newton shapes need not be contiguous or ordered by body.
        def attribute(name, model):
            index = 0 if name == "shape_material_mu" else 2
            return wp.from_torch(materials[:, None, :, index].contiguous())

        view = SimpleNamespace(body_shapes=[[2], [0, 1]], get_attribute=attribute)
        shapes = None
        physics = NewtonCfg()
        expected = torch.tensor([[[0.8, 0.8], [1.1, 1.1]]])
    term.asset = SimpleNamespace(
        root_view=view,
        _physics_sim_view=shapes,
        map_body_ids_to_backend=lambda ids: [1, 0],
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(backend_manager=SimpleNamespace(get_model=lambda: None)),
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=physics)),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term)),
    )
    selected = SceneEntityCfg("robot", body_ids=[0, 1])
    torch.testing.assert_close(term(env, selected, 0), expected[:, :, 0])
    torch.testing.assert_close(term(env, selected, 1), expected[:, :, 1])


def test_mass_scaling_uses_original_mass_without_compounding():
    cfg = LocomotionVelocityRoughEnvCfg().events.base_mass
    cfg.params["asset_cfg"].body_ids = [0]
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=proxy(torch.tensor([[4.0, 1.0], [5.0, 1.0]])),
            body_inertia=proxy(torch.ones(2, 2, 9)),
        )
    )

    def set_masses(masses, body_ids, env_ids):
        asset.data.body_mass.torch[env_ids, body_ids] = masses

    def set_inertias(inertias, body_ids, env_ids):
        asset.data.body_inertia.torch[env_ids, body_ids] = inertias

    asset.set_masses_index = set_masses
    asset.set_inertias_index = set_inertias
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": asset})
    term = cfg.func(cfg, env)
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    torch.testing.assert_close(base_mass_observation(env, "scale"), torch.ones(2, 1))
    for ratio in (1.25, 0.8, 1.25):
        cfg.params["mass_distribution_params"] = (ratio, ratio)
        term(env, None, **cfg.params)
        torch.testing.assert_close(base_mass_observation(env, "mass"), torch.tensor([[4.0], [5.0]]) * ratio)
        torch.testing.assert_close(base_mass_observation(env, "scale"), torch.full((2, 1), ratio))
        torch.testing.assert_close(base_mass_observation(env, "delta"), torch.tensor([[4.0], [5.0]]) * (ratio - 1))
        torch.testing.assert_close(asset.data.body_inertia.torch[:, 0], torch.full((2, 9), ratio))
    torch.testing.assert_close(asset.data.body_mass.torch[:, 1], torch.ones(2))


def test_native_push_is_additive_and_only_affects_selected_envs():
    cfg = LocomotionVelocityRoughEnvCfg().events.push_robot
    cfg.params["velocity_range"] = {"x": (0.2, 0.2), "yaw": (-0.1, -0.1)}
    asset = SimpleNamespace(device="cpu", data=SimpleNamespace(root_vel_w=proxy(torch.ones(3, 6))))

    def write(root_velocity, env_ids):
        asset.data.root_vel_w.torch[env_ids] = root_velocity

    asset.write_root_velocity_to_sim_index = write
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset}, common_step_counter=1)
    cfg.func(env, torch.tensor([0, 2]), **cfg.params)
    delta = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, -0.1], [0.0] * 6, [0.2, 0.0, 0.0, 0.0, 0.0, -0.1]])
    torch.testing.assert_close(asset.data.root_vel_w.torch, torch.ones(3, 6) + delta)


@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_com_offsets_use_original_pose_and_actual_written_values(backend):
    cfg = LocomotionVelocityRoughEnvCfg().events.base_com
    assert cfg.mode == "startup"
    cfg.params["asset_cfg"].body_ids = [0]
    poses = torch.zeros(3, 2, 7)
    poses[:, 0, :3] = torch.tensor([0.02, 0.000056, -0.041502])
    poses[..., 6] = 1.0
    original = poses.clone()
    asset = SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(
            body_com_pose_b=proxy(poses),
            body_com_pos_b=proxy(poses[..., :3]),
        ),
    )

    def set_coms(coms, body_ids, env_ids):
        rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]
        poses[rows, body_ids] = coms

    asset.set_coms_index = set_coms
    physics = PhysxCfg() if backend == "physx" else NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env = SimpleNamespace(
        num_envs=3, device="cpu", scene={"robot": asset}, cfg=SimpleNamespace(sim=SimpleNamespace(physics=physics))
    )
    term = cfg.func(cfg, env)
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    torch.testing.assert_close(base_com_offset_observation(env), torch.zeros(3, 3))
    cfg.params["com_range"] = {"x": (0.01, 0.01), "y": (-0.01, -0.01), "z": (0.005, 0.005)}
    term(env, None, **cfg.params)
    delta = torch.tensor([[0.01, -0.01, 0.005]]).repeat(3, 1)
    torch.testing.assert_close(base_com_offset_observation(env), delta)
    # Reapplying the same sample cannot accumulate CoM drift.
    term(env, None, **cfg.params)
    torch.testing.assert_close(base_com_offset_observation(env), delta)
    cfg.params["com_range"] = {"x": (-0.005, -0.005)}
    term(env, torch.tensor([1]), **cfg.params)
    delta[1] = torch.tensor([-0.005, 0.0, 0.0])
    torch.testing.assert_close(base_com_offset_observation(env), delta)
    torch.testing.assert_close(poses[:, 1], original[:, 1])
    torch.testing.assert_close(poses[..., 3:], original[..., 3:])
    # The observation reads back the actual data, not a cached random sample.
    poses[0, 0, 0] += 0.001
    delta[0, 0] += 0.001
    torch.testing.assert_close(base_com_offset_observation(env), delta)
