"""Static spherical sampling and sensor extrinsics; no simulator dependency."""

import numpy as np


def spherical_directions(count, elevation_range):
    """Equal-solid-angle Fibonacci samples in a full-azimuth elevation band.

    This is a static geometric approximation, not Livox's temporal scanning pattern.
    """
    if count < 1:
        raise ValueError("Ray count must be positive")
    low, high = elevation_range
    if not -90 <= low < high <= 90:
        raise ValueError("Require -90 <= lower elevation < upper elevation <= 90")
    index = np.arange(count, dtype=np.float64)
    z0, z1 = np.sin(np.deg2rad([low, high]))
    z = z0 + (z1 - z0) * (index + 0.5) / count
    azimuth = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z)).astype(np.float32)


def rpy_quaternion(rpy_degrees):
    """Sensor-to-parent quaternion xyzw, Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.deg2rad(rpy_degrees) / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def sensor_world_pose(parent_pos, parent_quat, offset_pos, offset_quat):
    """Compose mount offset with RayCaster's tracked-prim pose (base alignment)."""
    xyz = parent_quat[..., :3]
    w = parent_quat[..., 3:4]
    offset_pos = np.asarray(offset_pos)
    offset_quat = np.asarray(offset_quat)
    cross = 2 * np.cross(xyz, offset_pos)
    pos = parent_pos + offset_pos + w * cross + np.cross(xyz, cross)
    qxyz = w * offset_quat[:3] + offset_quat[3] * xyz + np.cross(xyz, offset_quat[:3])
    qw = w * offset_quat[3] - np.sum(xyz * offset_quat[:3], axis=-1, keepdims=True)
    return pos, np.concatenate((qxyz, qw), axis=-1)
