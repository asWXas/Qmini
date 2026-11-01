"""AS2W privileged observations, read from current simulator state.

Wheel friction is the mean over each wheel's collision shapes. Newton exposes
one friction coefficient, used for both static and dynamic observations.
"""

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab_newton.physics import NewtonCfg



def wheel_scan_height_range(env, sensor_names: list[str]):
    """每个轮子九个扫描点的 Z 极差，按传入顺序返回 (num_envs, 4)，单位米。

    扫描缺失时返回 1.5 m，避免将未知地形当作平地或输出 NaN。
    """
    ranges = []
    for name in sensor_names:
        heights = env.scene.sensors[name].data.ray_hits_w.torch[..., 2]
        height_range = heights.amax(dim=-1) - heights.amin(dim=-1)
        ranges.append(torch.where(torch.isfinite(heights).all(dim=-1), height_range, 1.5))
    return torch.stack(ranges, dim=-1)


def action_history(env):
    """Last two issued actions, ordered [a(t-1), a(t-2)]; reset rows are zeroed."""
    return torch.cat((env.action_manager.action, env.action_manager.prev_action), dim=-1)


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


def base_com_offset_observation(env, event_name="base_com"):
    """Actual CoM offset [dx, dy, dz] in meters, expressed in the base link frame."""
    term = env.event_manager.get_term_cfg(event_name).func
    current = term.asset.data.body_com_pos_b.torch[:, term.asset_cfg.body_ids]
    original = current if term.default_com is None else term.default_com[:, term.asset_cfg.body_ids, :3]
    return (current - original).reshape(env.num_envs, -1)


def joint_position_execution_error(env, asset_cfg: SceneEntityCfg):
    """Last applied position target minus current joint position [rad].

    Read the actuator target after action scaling/offset and physics, rather
    than the raw policy output. Reset episodes have no previous command.
    """
    asset = env.scene[asset_cfg.name]
    target = asset.actuators.target_command.position.torch[:, asset_cfg.joint_ids]
    actual = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    return (target - actual).masked_fill((env.episode_length_buf == 0).unsqueeze(-1), 0.0)


def joint_velocity_execution_error(env, asset_cfg: SceneEntityCfg):
    """Last applied velocity target minus current joint velocity [rad/s]."""
    asset = env.scene[asset_cfg.name]
    target = asset.actuators.target_command.velocity.torch[:, asset_cfg.joint_ids]
    actual = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return (target - actual).masked_fill((env.episode_length_buf == 0).unsqueeze(-1), 0.0)
