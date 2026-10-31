"""Rigid contact kinematics, raw PhysX adapters, and AS2W slip wiring."""

from types import SimpleNamespace as N

import pytest
import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

from Qmini.tasks.quadruped.as2w.mdp.wheel_slip import WheelContactSlip


@pytest.mark.parametrize("modern_api", [True, False])
@pytest.mark.parametrize("warp_buffers", [True, False])
def test_contact_slip(modern_api, warp_buffers):
    # Rolling, locked sliding, spinning, sideways, airborne, slope rolling,
    # normal approach, below tolerance, unloaded contact, two distinct surfaces.
    n = 10
    positions = torch.zeros(n, 4, 3)
    velocities = torch.zeros_like(positions)
    angular = torch.zeros_like(positions)
    points = torch.zeros(n * 2, 3)
    normals = torch.zeros_like(points)
    forces = torch.full((n * 2, 1), 10.0)
    counts = torch.ones(n, 1, dtype=torch.int32)
    starts = torch.arange(0, n * 2, 2).reshape(n, 1).to(torch.int32)
    points[::2, 2] = -1.0
    normals[::2, 2] = 1.0
    velocities[0, :, 0] = 1.0
    angular[0, :, 1] = 1.0
    velocities[1, :, 0] = 1.0
    angular[2, :, 1] = 1.0
    velocities[3, :, 1] = 1.0
    velocities[4, :, :] = 50.0
    counts[4] = 0
    # 45-degree uphill pure rolling, with reversed normal sign and non-unit length.
    c = 2**-0.5
    points[10] = torch.tensor([c, 0.0, -c])
    normals[10] = torch.tensor([2 * c, 0.0, -2 * c])
    velocities[5] = torch.tensor([c, 0.0, c])
    angular[5, :, 1] = 1.0
    velocities[6, :, 2] = -2.0  # Normal impact is not tangential slip.
    velocities[7, :, 0] = 0.02
    velocities[8, :, 0] = 5.0
    forces[16] = 0.0
    # Pure rolling at bottom, but sliding on a second side contact: must not average them.
    velocities[9, :, 0] = 1.0
    angular[9, :, 1] = 1.0
    counts[9] = 2
    points[19] = torch.tensor([1.0, 0.0, 0.0])
    normals[19] = torch.tensor([-1.0, 0.0, 0.0])
    buffers = (forces, points, normals, torch.zeros(n * 2, 1), counts, starts)
    if warp_buffers:
        buffers = tuple(
            wp.from_torch(value, dtype=wp.uint32 if value.dtype == torch.int32 else wp.float32) for value in buffers
        )
    calls = []

    def read(*, dt):
        calls.append(dt)
        return buffers

    names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    sensors = {}
    for name in names:
        view = N(filter_count=1, max_contact_data_count=n * 2)
        setattr(view, "get_normal_contact_data" if modern_api else "get_contact_data", read)
        sensors[name] = N(body_names=[name], contact_view=view)
    asset = N(
        body_names=names,
        data=N(
            body_link_pos_w=N(torch=positions),
            body_link_lin_vel_w=N(torch=velocities),
            body_link_ang_vel_w=N(torch=angular),
        ),
    )

    class Scene(dict):
        pass

    scene = Scene(robot=asset)
    scene.sensors = sensors
    env = N(scene=scene, num_envs=n, device="cpu", physics_dt=0.005)
    params = dict(
        asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1, 2, 3]),
        sensor_names=names,
        slip_threshold=0.05,
        contact_threshold=1.0,
    )
    term = WheelContactSlip(N(params=params), env)
    result = term(env, **params)
    expected_speed = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.02, 0.0, 1.0])
    torch.testing.assert_close(term.slip_speed, expected_speed[:, None].expand(-1, 4), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result, (expected_speed - 0.05).clamp(min=0).square() * 4)
    assert torch.equal(term.is_slipping, (expected_speed > 0.05)[:, None].expand(-1, 4))
    assert calls == [0.005] * 4
    # Changing the link reference origin (including off-center wheel meshes) changes neither slip nor penalty.
    offset = torch.tensor([0.2, 0.05, 0.1]).expand_as(positions)
    velocities += torch.linalg.cross(angular, offset)
    positions += offset
    torch.testing.assert_close(term(env, **params), result, atol=1e-6, rtol=1e-5)
    # Removing contacts overwrites old slip diagnostics; airborne spin is not penalized.
    counts.zero_()
    torch.testing.assert_close(term(env, **params), torch.zeros(n))
    term.reset()
    assert not term.is_slipping.any()


def test_slip_sensor_and_reward_configuration():
    from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import AS2WFlatEnvCfg, AS2WRoughEnvCfg

    for cfg_type, path in (
        (AS2WRoughEnvCfg, "/World/ground/terrain/mesh"),
        (AS2WFlatEnvCfg, "/World/ground/terrain/GroundPlane/CollisionPlane"),
    ):
        cfg = cfg_type()
        term = cfg.rewards.wheel_contact_slip
        assert term.weight < 0
        assert term.func is WheelContactSlip
        for body, sensor_name in zip(term.params["asset_cfg"].body_names, term.params["sensor_names"], strict=True):
            sensor = getattr(cfg.scene, sensor_name)
            assert sensor.prim_path.endswith("/" + body)
            assert sensor.filter_prim_paths_expr == [path]
            assert sensor.track_contact_points
            assert sensor.max_contact_data_count_per_prim == 64
