"""Per-contact wheel slip on static terrain using PhysX contact points and normals.

Positions and velocities use the same link origin, so the off-center wheel mesh
needs no radius or center correction. All angular motion is included.
"""

import math
from contextlib import nullcontext

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

__all__ = ["WheelContactSlip"]


@wp.kernel
def _contact_slip_kernel(
    forces: wp.array(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    counts: wp.array2d(dtype=wp.uint32),
    starts: wp.array2d(dtype=wp.uint32),
    positions: wp.array2d(dtype=wp.vec3),
    linear_velocities: wp.array2d(dtype=wp.vec3),
    angular_velocities: wp.array2d(dtype=wp.vec3),
    wheel: int,
    force_threshold: float,
    speeds: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    max_speed = float(0.0)  # noqa: UP018 - Warp needs a mutable variable in dynamic loops.
    for partner in range(counts.shape[1]):
        start = int(starts[env_id, partner])
        count = int(counts[env_id, partner])
        for offset in range(count):
            idx = start + offset
            if wp.abs(forces[idx]) > force_threshold:
                normal = normals[idx]
                norm_sq = wp.dot(normal, normal)
                if norm_sq > 1.0e-12 and wp.isfinite(norm_sq):
                    normal = normal / wp.sqrt(norm_sq)
                    arm = points[idx] - positions[env_id, wheel]
                    velocity = linear_velocities[env_id, wheel] + wp.cross(angular_velocities[env_id, wheel], arm)
                    tangent = velocity - wp.dot(velocity, normal) * normal
                    max_speed = wp.max(max_speed, wp.length(tangent))
    speeds[env_id, wheel] = max_speed


def _warp_view(value, dtype):
    """PhysX uses Warp arrays; accepting torch also permits isolated adapter tests."""
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.uint32:
            value = value.view(torch.int32)
        return wp.from_torch(value.contiguous(), dtype=dtype)
    return value.view(dtype)


class WheelContactSlip(ManagerTermBase):
    """Penalize slip above a tolerance, using the worst loaded contact per wheel.

    ``slip_speed`` and ``is_slipping`` expose (num_envs, 4) diagnostics. The return
    value is sum(max(slip_speed - slip_threshold, 0)^2), for a negative weight.
    Static ground only. Reads the most recent physics step, not a substep average.
    PhysX 6.1 and 6.2 detailed contact APIs are supported. Missing support fails
    explicitly; it never falls back to a flat-ground approximation.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        ids = cfg.params["asset_cfg"].body_ids
        self.body_ids = list(range(len(self.asset.body_names)))[ids] if isinstance(ids, slice) else ids
        self.sensors = [env.scene.sensors[name] for name in cfg.params["sensor_names"]]
        if len(self.body_ids) != 4 or len(self.sensors) != 4:
            raise ValueError("WheelContactSlip requires four ordered wheel bodies and four sensors")
        self.readers = []
        for body_id, sensor in zip(self.body_ids, self.sensors, strict=True):
            if list(sensor.body_names) != [self.asset.body_names[body_id]]:
                raise ValueError("Each slip sensor must track its corresponding single wheel body")
            view = sensor.contact_view
            reader = getattr(view, "get_normal_contact_data", None)
            if reader is None:
                reader = getattr(view, "get_contact_data", None)
            if reader is None:
                raise RuntimeError("WheelContactSlip requires the PhysX detailed contact API")
            if view.filter_count < 1 or view.max_contact_data_count < 1:
                raise ValueError("Wheel slip needs terrain filters and allocated detailed contact buffers")
            self.readers.append(reader)
        self.slip_speed = torch.zeros((env.num_envs, 4), device=env.device)
        self.is_slipping = torch.zeros_like(self.slip_speed, dtype=torch.bool)
        self._speeds_wp = wp.from_torch(self.slip_speed)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.slip_speed[ids] = 0.0
        self.is_slipping[ids] = False

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        sensor_names: list[str],
        slip_threshold: float = 0.05,
        contact_threshold: float = 1.0,
    ):
        if not all(math.isfinite(x) and x >= 0.0 for x in (slip_threshold, contact_threshold)):
            raise ValueError("Slip and contact thresholds must be finite and non-negative")
        data = self.asset.data
        positions = data.body_link_pos_w.torch[:, self.body_ids].contiguous()
        linear = data.body_link_lin_vel_w.torch[:, self.body_ids].contiguous()
        angular = data.body_link_ang_vel_w.torch[:, self.body_ids].contiguous()
        positions_wp = wp.from_torch(positions, dtype=wp.vec3)
        linear_wp = wp.from_torch(linear, dtype=wp.vec3)
        angular_wp = wp.from_torch(angular, dtype=wp.vec3)
        # Torch writes state tensors; run Warp on the same CUDA stream before Torch reads the result.
        stream = (
            wp.ScopedStream(wp.stream_from_torch(self.slip_speed.device)) if self.slip_speed.is_cuda else nullcontext()
        )
        with stream:
            for wheel, reader in enumerate(self.readers):
                forces, points, normals, _, counts, starts = reader(dt=env.physics_dt)
                wp.launch(
                    _contact_slip_kernel,
                    dim=env.num_envs,
                    inputs=[
                        _warp_view(forces.reshape(-1), wp.float32),
                        _warp_view(points, wp.vec3),
                        _warp_view(normals, wp.vec3),
                        _warp_view(counts, wp.uint32),
                        _warp_view(starts, wp.uint32),
                        positions_wp,
                        linear_wp,
                        angular_wp,
                        wheel,
                        contact_threshold,
                    ],
                    outputs=[self._speeds_wp],
                    device=env.device,
                )
        self.is_slipping.copy_(self.slip_speed > slip_threshold)
        return (self.slip_speed - slip_threshold).clamp(min=0.0).square().sum(dim=-1)
