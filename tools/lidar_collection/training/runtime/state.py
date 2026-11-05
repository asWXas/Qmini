"""Spatial recurrent state and relative-only SE(3) geometry for v3."""

from dataclasses import dataclass, field

import torch

from ..geometry.temporal import rotation_matrix
from .observation_memory import ObservedSurfaceMemory


def transform_points(points, transform):
    """T_current_from_previous: x_current = R @ x_previous + translation."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def validate_transform(transform):
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError("Expected finite T_current_from_previous with shape (4,4)")
    identity = torch.eye(3, device=transform.device, dtype=transform.dtype)
    r = transform[:3, :3]
    if not torch.allclose(r.T @ r, identity, atol=1e-4, rtol=1e-4) or not torch.isclose(
        r.det(), r.new_tensor(1.0), atol=1e-4
    ):
        raise ValueError("Relative transform rotation must belong to SO(3)")
    if not torch.allclose(transform[3], transform.new_tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("Invalid homogeneous transform bottom row")


def relative_transform(position, quaternion, previous_position, previous_quaternion):
    transform = torch.eye(4, device=position.device, dtype=position.dtype)
    if previous_position is not None:
        rotation = rotation_matrix(quaternion).T
        transform[:3, :3] = rotation @ rotation_matrix(previous_quaternion)
        transform[:3, 3] = rotation @ (previous_position - position)
    return transform


@dataclass
class FeatureMemory:
    # Coordinates are [batch,x,y,z] in finest-grid units; each row owns a scalar feature vector.
    coordinates: torch.Tensor
    features: torch.Tensor
    stride: int

    def detach(self):
        return FeatureMemory(self.coordinates, self.features.detach(), self.stride)


@dataclass
class RecurrentState:
    prediction: torch.Tensor | None = None  # previous centered local map (world axes by default)
    hidden: list = field(default_factory=list)
    observation_memory: ObservedSurfaceMemory = field(default_factory=ObservedSurfaceMemory)
    network_prediction: torch.Tensor | None = None
    retained_observations: torch.Tensor | None = None
    started: bool = False
    orientation: torch.Tensor | None = None  # R_world_from_body; relative deployment integrates rotation only
    # Offline adapter only: never supplied to the learned model or required by deployment.
    position: torch.Tensor | None = None
    quaternion: torch.Tensor | None = None

    def detach_hidden(self):
        self.hidden = [memory.detach() for memory in self.hidden]


def warp_features(memory, transform, size, resolution):
    """Forward splat cell centers into the new frame, averaging collisions.

    Features keep their autograd history. Coordinates are discrete; there is no
    gradient through quantization. Channels are learned scalar features, not XYZ vectors.
    """
    stride = memory.stride
    centers = (memory.coordinates[:, 1:].to(memory.features.dtype) + stride / 2) * (size / resolution) - size / 2
    moved = transform_points(centers, transform)
    scaled = (moved + size / 2) * (resolution / size) / stride
    # Relative rotation round-trips can place an exact grid boundary a few ulps below
    # its integer index. Snap only numerical ties before flooring, avoiding lost cells.
    nearest = scaled.round()
    scaled = torch.where((scaled - nearest).abs() <= 1e-6, nearest, scaled)
    grid = torch.floor(scaled).int() * stride
    valid = ((grid >= 0) & (grid < resolution)).all(1)
    coords = torch.cat((memory.coordinates[valid, :1], grid[valid]), dim=1)
    unique, inverse, counts = torch.unique(coords, dim=0, return_inverse=True, return_counts=True)
    values = memory.features.new_zeros((len(unique), memory.features.shape[1]))
    values = values.index_add(0, inverse, memory.features[valid]) / counts[:, None].clamp_min(1)
    return FeatureMemory(unique, values, stride)


def pack_hidden(states, transforms, size, resolution):
    packed = []
    for scale in range(5):
        memories = []
        for batch, (state, transform) in enumerate(zip(states, transforms, strict=True)):
            if state.hidden:
                memory = warp_features(state.hidden[scale], transform, size, resolution)
                coords = memory.coordinates.clone()
                coords[:, 0] = batch
                memories.append(FeatureMemory(coords, memory.features, memory.stride))
        packed.append(
            FeatureMemory(
                torch.cat([m.coordinates for m in memories]), torch.cat([m.features for m in memories]), 2**scale
            )
            if memories
            else None
        )
    return packed


def unpack_hidden(packed, batch):
    result = []
    for memory in packed:
        selected = memory.coordinates[:, 0] == batch
        coords = memory.coordinates[selected].clone()
        coords[:, 0] = 0
        result.append(FeatureMemory(coords, memory.features[selected], memory.stride))
    return result


def coverage_mask(cells, stride, config, quaternion):
    """Only supervise body-frame voxels wholly covered by recorded world-axis GT.

    Rotating a 3.2 m cube does not preserve its coverage. Unrecorded corners must
    not become negative labels. This mask affects supervision/metrics, never prediction.
    """
    if config.get("map_frame", "world") == "world":
        return ((cells >= 0) & (cells + stride <= config["resolution"])).all(1)
    rotation = rotation_matrix(quaternion)
    spacing = config["map_size"] / config["resolution"]
    local = (cells.to(rotation.dtype) + stride / 2) * spacing - config["map_size"] / 2
    world = local @ rotation.T
    extent = rotation.abs().sum(1) * (stride * spacing / 2)
    recorded_size = config.get("gt_map_size", config["map_size"])
    return ((world.abs() + extent) <= recorded_size / 2 + 1e-6).all(1)
