"""Contact measurements shared by rewards and privileged observations."""

from __future__ import annotations

import torch


def finger_object_forces(env) -> torch.Tensor:
    """Left/right finger forces against this environment's object, shape (N, 2, 3)."""
    return torch.stack(
        [
            env.scene[name].data.normal_force_matrix_w.torch[:, 0].sum(dim=1)
            for name in ("left_finger_contact", "right_finger_contact")
        ],
        dim=1,
    )


def finger_object_contact(env, threshold: float = 0.1) -> torch.Tensor:
    """Contact with the object, rather than any contact with the table or robot."""
    return finger_object_forces(env).norm(dim=-1) > threshold


def object_table_support(env) -> torch.Tensor:
    """Measured non-robot upward support near this scene's tabletop (local z=0).

    PhysX GPU filtering does not support the table's static instance collider.
    Subtract ALL robot normal contacts from the object's net normal force instead.
    In this scene the remaining colliders are the table and ground; the height
    band excludes the ground/legs. Height alone never counts as support. The
    8 cm upper bound accommodates all configured cube sizes and orientations.
    """
    contact = env.scene["object_contact"].data
    robot_force = contact.normal_force_matrix_w.torch[:, 0].sum(dim=1)
    external_force = contact.net_normal_forces_w.torch[:, 0] - robot_force
    height = env.scene["object"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (external_force[:, 2] > 0.1) & (height > 0.0) & (height < 0.08)
