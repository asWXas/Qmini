"""Read-only privileged observations for native Isaac Lab disturbance events."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab_newton.physics import NewtonCfg


class feet_friction(ManagerTermBase):
    """Read actual per-shape friction without wrapping or modifying material events.

    Newton has one friction coefficient; it occupies both friction columns.
    Shape ordering is backend-specific and fixed for the lifetime of the scene.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._body_shapes = None

    def body_friction(self, env, body_ids):
        """Static/dynamic friction in selected public body order, averaged over each body's shapes."""
        materials = self._read_materials(env).reshape(env.num_envs, -1, 3)
        if self._body_shapes is None:
            view = self.asset.root_view
            if isinstance(env.cfg.sim.physics, NewtonCfg):
                self._body_shapes = view.body_shapes
            else:
                # PhysX articulation materials concatenate shapes in backend link order.
                self._body_shapes = []
                offset = 0
                for path in view.link_paths[0]:
                    count = self.asset._physics_sim_view.create_rigid_body_view(path).max_shapes
                    self._body_shapes.append(list(range(offset, offset + count)))
                    offset += count
                if offset != materials.shape[1]:
                    raise ValueError("PhysX body-shape mapping does not match the material buffer")
        values = []
        for body_id in self.asset.map_body_ids_to_backend(body_ids):
            shape_ids = self._body_shapes[body_id]
            if not len(shape_ids):
                raise ValueError("Foot friction requires a body with collision shapes")
            values.append(materials[:, shape_ids, :2].mean(dim=1))
        return torch.stack(values, dim=1)

    def _read_materials(self, env):
        view = self.asset.root_view
        if isinstance(env.cfg.sim.physics, NewtonCfg):
            model = env.sim.backend_manager.get_model()
            mu = wp.to_torch(view.get_attribute("shape_material_mu", model)[:, 0])
            restitution = wp.to_torch(view.get_attribute("shape_material_restitution", model)[:, 0])
            values = torch.stack((mu, mu, restitution), dim=-1)
        else:
            values = wp.to_torch(view.get_material_properties())
        return values.reshape(env.num_envs, -1).to(env.device).clone()

    def __call__(self, env, asset_cfg: SceneEntityCfg, component: int):
        """Per-foot static (0) or dynamic (1) friction in the selected body order."""
        return self.body_friction(env, asset_cfg.body_ids)[:, :, component]


def base_wrench_observation(env, asset_cfg: SceneEntityCfg):
    """Actual local force [N] and torque [Nm] written by apply_external_force_torque.

    Scene reset clears these buffers, including when the event is disabled in play.
    This observation assumes the configured native event is the only wrench writer.
    """
    composer = env.scene[asset_cfg.name].permanent_wrench_composer
    forces = wp.to_torch(composer.local_force_b)[:, asset_cfg.body_ids]
    torques = wp.to_torch(composer.local_torque_b)[:, asset_cfg.body_ids]
    return torch.cat((forces, torques), dim=-1).reshape(env.num_envs, -1)


def feet_contact_state(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).float()


def feet_air_time(env, sensor_cfg: SceneEntityCfg):
    """Current airborne duration in seconds, zero while the foot is in contact."""
    return env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]


def base_mass_observation(env, quantity: str, event_name="base_mass"):
    term = env.event_manager.get_term_cfg(event_name).func
    mass = term.asset.data.body_mass.torch[:, term.asset_cfg.body_ids]
    # ObservationManager infers shapes before startup events capture their defaults.
    original = mass if term.default_mass is None else term.default_mass[:, term.asset_cfg.body_ids]
    if quantity == "mass":
        return mass
    if quantity == "scale":
        return mass / original
    if quantity == "delta":
        return mass - original
    raise ValueError(f"Unknown mass observation quantity: {quantity}")


def base_com_offset_observation(env, event_name="base_com"):
    """Actual CoM offset [dx, dy, dz] in meters, expressed in the base link frame."""
    term = env.event_manager.get_term_cfg(event_name).func
    current = term.asset.data.body_com_pos_b.torch[:, term.asset_cfg.body_ids]
    original = current if term.default_com is None else term.default_com[:, term.asset_cfg.body_ids, :3]
    return (current - original).reshape(env.num_envs, -1)


def body_mass_observation(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.body_mass.torch[:, asset_cfg.body_ids]


def body_com_observation(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.body_com_pos_b.torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def feet_contact_forces(env, sensor_cfg: SceneEntityCfg):
    return env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids].reshape(env.num_envs, -1)


def base_height_above_ground(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg | None = None):
    """Height over the median finite terrain ray hit, robust to missed rays."""
    hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    ground = torch.where(torch.isfinite(hits), hits, torch.nan).nanmedian(dim=-1).values
    ground = torch.where(torch.isfinite(ground), ground, env.scene.env_origins[:, 2])
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    return (env.scene[asset_name].data.root_pos_w.torch[:, 2] - ground).unsqueeze(-1)


def base_too_low(env, minimum_height, sensor_cfg: SceneEntityCfg):
    return base_height_above_ground(env, sensor_cfg).squeeze(-1) < minimum_height
