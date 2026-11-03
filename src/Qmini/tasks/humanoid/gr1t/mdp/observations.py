"""GR1T observations: joint/body state, feet/terrain, and privileged dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensor


__all__ = [
    "joint_pos_err",
    "body_ang_vel",
    "body_projected_gravity",
    "air_time",
    "contact_force_violation",
    "feet_contact_state",
    "feet_friction",
    "base_mass_ratio",
    "base_com_offset_observation",
    "base_wrench_observation",
    "feet_height",
    "feet_scan",
]


def joint_pos_err(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current joint angle minus the last applied position target, in radians.

    Observations are computed after physics and before the next policy action,
    so the actuator command still holds the preceding control step's target,
    including action scaling and default offsets. Newly reset environments have
    no preceding target; return zero there to avoid leaking the previous episode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    position = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    target = asset.actuators.target_command.position.torch[:, asset_cfg.joint_ids]
    error = position - target
    return torch.where(env.episode_length_buf[:, None] > 0, error, 0.0)


def body_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity of selected bodies expressed in their local frames. Returns: [num_envs, 3 * num_bodies]."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]

    body_ang_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)
    return body_ang_vel_b.reshape(env.num_envs, -1)


def body_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity direction projected into selected body frames. Returns: [num_envs, 3 * num_bodies]."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity_dir_w = asset.data.GRAVITY_VEC_W.torch.unsqueeze(1)

    # IMPORTANT: quat_apply_inverse does not automatically expand the vector across multiple bodies.
    gravity_dir_w = gravity_dir_w.expand(-1, body_quat_w.shape[1], -1)
    projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_dir_w)

    return projected_gravity.reshape(env.num_envs, -1)


def air_time(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current air time of selected feet. Returns: [num_envs, num_feet]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch

    if current_air_time is None:
        raise RuntimeError("Contact sensor air-time tracking is disabled. Set track_air_time=True in ContactSensorCfg.")

    air_time_val = current_air_time[:, sensor_cfg.body_ids]
    return air_time_val.reshape(env.num_envs, -1)


def contact_force_violation(
    env: ManagerBasedEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """输出接触力超过 threshold 的幅值。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.linalg.norm(net_contact_forces, dim=-1)
    max_force = torch.max(force_norm, dim=1).values

    return (max_force - threshold).clamp_min(0.0)


def feet_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Current foot contact flags (0/1), ordered by body_ids. Shape: [num_envs, num_feet]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).float()


class feet_friction(ManagerTermBase):
    """Actual PhysX foot friction, averaged over each body's collision shapes.

    Cache only the body-to-shape mapping; read material values on every call so
    startup/reset randomization is reflected in subsequent observations.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._body_shape_slices = None

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, component: int) -> torch.Tensor:
        """Return [num_envs, num_feet] in selected body order: static (0) or dynamic (1)."""
        if component not in (0, 1):
            raise ValueError("Friction component must be 0 (static) or 1 (dynamic)")
        materials = wp.to_torch(self.asset.root_view.get_material_properties())
        if self._body_shape_slices is None:
            shape_slices = []
            offset = 0
            for path in self.asset.root_view.link_paths[0]:
                count = self.asset._physics_sim_view.create_rigid_body_view(path).max_shapes
                shape_slices.append(slice(offset, offset + count))
                offset += count
            if offset != materials.shape[1]:
                raise ValueError("PhysX body-shape mapping does not match the material buffer")
            self._body_shape_slices = shape_slices

        body_ids = self.asset.map_body_ids_to_backend(asset_cfg.body_ids)
        if isinstance(body_ids, slice):
            body_ids = range(len(self._body_shape_slices))[body_ids]
        values = []
        for body_id in body_ids:
            shapes = self._body_shape_slices[body_id]
            if shapes.start == shapes.stop:
                raise ValueError("Foot friction requires a body with collision shapes")
            values.append(materials[:, shapes, component].mean(dim=1))
        return torch.stack(values, dim=-1).to(env.device)


def base_mass_ratio(env: ManagerBasedEnv, event_name: str = "base_mass") -> torch.Tensor:
    """Current/default mass for the bodies selected by the mass randomization event."""
    term = env.event_manager.get_term_cfg(event_name).func
    mass = term.asset.data.body_mass.torch[:, term.asset_cfg.body_ids]
    # Observation dimensions are inferred before startup events capture defaults.
    if term.default_mass is None:
        return torch.ones_like(mass)
    return mass / term.default_mass[:, term.asset_cfg.body_ids]


def base_com_offset_observation(env: ManagerBasedEnv, event_name: str = "base_com") -> torch.Tensor:
    """Actual CoM displacement [dx, dy, dz] from the default, in the base frame (m)."""
    term = env.event_manager.get_term_cfg(event_name).func
    current = term.asset.data.body_com_pos_b.torch[:, term.asset_cfg.body_ids]
    original = current if term.default_com is None else term.default_com[:, term.asset_cfg.body_ids, :3]
    return (current - original).reshape(env.num_envs, -1)


def base_wrench_observation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Persistent local [Fx, Fy, Fz, Tx, Ty, Tz] applied by the external-wrench event.

    Force is in N and torque in N*m. Scene reset clears these buffers; velocity
    pushes do not write to them and are not represented as an external force.
    """
    composer = env.scene[asset_cfg.name].permanent_wrench_composer
    forces = wp.to_torch(composer.local_force_b)[:, asset_cfg.body_ids]
    torques = wp.to_torch(composer.local_torque_b)[:, asset_cfg.body_ids]
    return torch.cat((forces, torques), dim=-1).reshape(env.num_envs, -1)


def feet_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, sensor_names: list[str]) -> torch.Tensor:
    """Foot clearance relative to local terrain. Returns: [num_envs, num_feet]."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    terrain_heights = []

    for foot_id, sensor_name in enumerate(sensor_names):
        scanner = env.scene.sensors[sensor_name]
        ray_hit_z = scanner.data.ray_hits_w.torch[..., 2]

        # Ray caster may return inf for missed rays.
        valid = torch.isfinite(ray_hit_z)
        ray_hit_z = torch.where(valid, ray_hit_z, torch.full_like(ray_hit_z, torch.nan))

        terrain_z = torch.nanmedian(ray_hit_z, dim=1).values

        # If all rays missed, make clearance 0 instead of NaN.
        terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, feet_z[:, foot_id])
        terrain_heights.append(terrain_z)

    terrain_z = torch.stack(terrain_heights, dim=-1)
    feet_clearance = feet_z - terrain_z

    # Numerical penetration -> 0
    feet_clearance = torch.clamp(feet_clearance, min=0.0)

    return feet_clearance.reshape(env.num_envs, -1)


def feet_scan(env: ManagerBasedEnv, sensor_names: list[str]) -> torch.Tensor:
    """Foot-frame origin Z minus terrain Z, in meters; returns [num_envs, 9 * 2].

    Concatenate left then right 3x3 scans, preserving each sensor's ray order.
    The ray-start offset is not included in data.pos_w, so do not subtract it.
    Missed rays return 1 m, matching the observation's upper clipping bound.
    """
    if len(sensor_names) != 2 or len(set(sensor_names)) != 2:
        raise ValueError("Expected two distinct foot scanners ordered left, right")
    scans = []
    for name in sensor_names:
        data = env.scene.sensors[name].data
        hits = data.ray_hits_w.torch
        if hits.shape[1] != 9:
            raise ValueError(f"Foot scanner {name} must have 9 rays, got {hits.shape[1]}")
        heights = data.pos_w.torch[:, 2, None] - hits[..., 2]
        valid = torch.isfinite(hits).all(dim=-1) & torch.isfinite(heights)
        scans.append(torch.where(valid, heights, 1.0))
    return torch.cat(scans, dim=-1)
