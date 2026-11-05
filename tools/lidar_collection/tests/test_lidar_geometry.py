"""Check spherical FOV and tilted mounting with independently known transforms."""

import numpy as np

from ..core.lidar_geometry import rpy_quaternion, sensor_world_pose, spherical_directions


def test_hemisphere():
    directions = spherical_directions(12000, (0, 90))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-6)
    assert np.all(directions[:, 2] > 0)
    assert np.unique(np.round(directions[:, 2], 6)).size == 12000
    np.testing.assert_allclose(directions.mean(axis=0), [0, 0, 0.5], atol=0.001)


def test_mid360_elevation_limits():
    directions = spherical_directions(20000, (-7, 52))
    elevations = np.rad2deg(np.arcsin(directions[:, 2]))
    assert elevations.min() >= -7 and elevations.max() <= 52
    assert elevations.min() < -6.9 and elevations.max() > 51.9


def test_mount_and_parent_rotation_are_composed_once():
    # Base turned +90 degrees in yaw: a forward mount is now world +Y.
    pos, quat = sensor_world_pose(
        np.array([[1, 2, 3]]),
        np.array([rpy_quaternion((0, 0, 90))]),
        (0.32, 0, 0.18),
        rpy_quaternion((0, 60, 0)),
    )
    np.testing.assert_allclose(pos, [[1, 2.32, 3.18]], atol=1e-6)
    # Mounted +X beam is down 60 degrees and world +Y after parent yaw.
    beam, _ = sensor_world_pose(np.zeros((1, 3)), quat, (1, 0, 0), (0, 0, 0, 1))
    np.testing.assert_allclose(beam, [[0, 0.5, -np.sqrt(3) / 2]], atol=1e-6)
