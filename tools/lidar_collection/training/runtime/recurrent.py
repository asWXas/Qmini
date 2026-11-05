"""Shared relative-only recurrent forward and recorded-data training adapter."""

import torch

from ..datasets.augmentation import augment_measurement
from ..geometry.sparse_voxel import SparsePrediction, decode_sparse, sparse_reconstruction_loss, sparse_voxelize
from ..geometry.temporal import canvas_voxels, observed_voxels, plane_voxels, rotation_matrix
from ..geometry.voxel import voxelize
from .observation_memory import preserve_observed_surface
from .state import coverage_mask, pack_hidden, relative_transform, transform_points, unpack_hidden


def relative_batch(model, observations, transforms, states, config, noise=None, orientations=None):
    """Body-frame sensor inputs + relative transforms; outputs use centered world axes.

    Optional R_world_from_body orientations align the first frame to an existing world
    frame. Otherwise the first body orientation defines the local world's fixed axes.
    No absolute translation or ground truth is supplied to the model.
    """
    size, resolution = config["map_size"], config["resolution"]
    coordinates, features, poses, effective, current_orientations = [], [], [], [], []
    measured_points = []
    for batch, (points, transform, state) in enumerate(zip(observations, transforms, states, strict=True)):
        transform = transform if state.started else torch.eye(4, device=points.device, dtype=points.dtype)
        world_aligned = config.get("map_frame", "world") == "world"
        if world_aligned:
            if orientations is not None:
                orientation = orientations[batch]
            elif state.started:
                orientation = state.orientation @ transform[:3, :3].T
            else:
                orientation = torch.eye(3, device=points.device, dtype=points.dtype)
            points = points @ orientation.T
            geometry = torch.eye(4, device=points.device, dtype=points.dtype)
            geometry[:3, 3] = orientation @ transform[:3, 3]
        else:
            orientation = torch.eye(3, device=points.device, dtype=points.dtype)
            geometry = transform
        current_orientations.append(orientation)
        effective.append(geometry)
        current = observed_voxels(points, size, resolution, noise)
        if config.get("observation_memory", False):
            measured_points.append((current[0] + current[1]) * (size / resolution) - size / 2)
        if not state.started and config.get("history_init", "plane") == "current":
            sources = (current, current)
        elif state.started:
            prior = transform_points(state.prediction, geometry)
            sources = (current, sparse_voxelize(prior, size, resolution))
        elif config.get("history_init", "plane") == "plane":
            cells, offsets = plane_voxels(config, points)
            prior = (cells + offsets) * (size / resolution) - size / 2
            sources = (current, sparse_voxelize(prior, size, resolution))
        else:
            sources = (current, sparse_voxelize(points.new_empty((0, 3)), size, resolution))
        if config.get("canvas", False):
            sources = (*sources, canvas_voxels(config, points))
        for source, (cells, offsets) in enumerate(sources):
            ids = torch.full((len(cells), 1), batch, device=points.device, dtype=torch.int32)
            coordinates.append(torch.cat((ids, cells, torch.full_like(ids, source)), dim=1))
            features.append(offsets)
        if model.pose_encoder is not None:
            poses.append(
                torch.cat(
                    (
                        transform[:3, 3] / size,
                        transform[:3, :2].T.reshape(-1),
                        points.new_tensor([float(state.started)]),
                    )
                )
            )
    hidden = pack_hidden(states, effective, size, resolution)
    output, next_hidden = model(
        torch.cat(coordinates), torch.cat(features), torch.stack(poses) if poses else None, hidden, len(states)
    )
    predictions, outputs = [], []
    for batch, state in enumerate(states):
        stages = [(c[c[:, 0] == batch], logits[c[:, 0] == batch], stride) for c, logits, stride in output.stages]
        selected = output.coordinates[:, 0] == batch
        current = SparsePrediction(
            stages, output.coordinates[selected], output.logits[selected], output.offsets[selected]
        )
        prediction = decode_sparse(current, size, resolution, config["threshold"], config["max_points"])
        state.network_prediction = prediction
        if config.get("observation_memory", False):
            state.retained_observations = state.observation_memory.update(
                measured_points[batch], effective[batch], config
            )
            prediction = preserve_observed_surface(prediction, state.retained_observations, config)
        outputs.append(current)
        predictions.append(prediction)
        state.prediction = prediction  # discrete geometry is detached; hidden features retain TBPTT graph
        state.hidden = unpack_hidden(next_hidden, batch)
        state.started = True
        state.orientation = current_orientations[batch].detach().clone()
    return outputs, predictions


def supervised_loss(output, target_local, quaternion, config):
    size, resolution = config["map_size"], config["resolution"]
    cells, offsets = sparse_voxelize(target_local, size, resolution)
    valid = coverage_mask(cells, 1, config, quaternion)
    cells, offsets = cells[valid], offsets[valid]
    stages = []
    for coordinates, logits, stride in output.stages:
        valid = coverage_mask(coordinates[:, 1:4], stride, config, quaternion)
        if bool(valid.any()):
            stages.append((coordinates[valid], logits[valid], stride))
    valid = coverage_mask(output.coordinates[:, 1:4], 1, config, quaternion)
    supervised = SparsePrediction(stages, output.coordinates[valid], output.logits[valid], output.offsets[valid])
    return sparse_reconstruction_loss(
        supervised, cells, offsets, resolution, config["offset_weight"], config.get("occupancy_loss", "bce")
    )


def mask_metric_grid(grid, quaternion, config):
    occupied = grid[0].nonzero()
    invalid = occupied[~coverage_mask(occupied, 1, config, quaternion)]
    if len(invalid):
        grid[:, invalid[:, 0], invalid[:, 1], invalid[:, 2]] = 0
    return grid


def recorded_batch(model, frames, states, config, device, augment=True, compute_metrics=False):
    """Absolute simulation poses ONLY derive relative odometry and supervision coordinates.

    Current sensor coordinates use the true body frame; paper pose augmentation
    perturbs relative translation via noisy successive origins. GT stays clean.
    """
    observations, targets, transforms, rotations, positions, estimated_positions, quaternions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for frame, state in zip(frames, states, strict=True):
        position = torch.as_tensor(frame["base_pos_w"], dtype=torch.float32, device=device)
        quaternion = torch.as_tensor(frame["base_quat_xyzw"], dtype=torch.float32, device=device)
        rotation = rotation_matrix(quaternion)
        observed = torch.as_tensor(frame["lidar_points_w"], dtype=torch.float32, device=device) - position
        target = torch.as_tensor(frame["gt_points_w"], dtype=torch.float32, device=device) - position
        world_aligned = config.get("map_frame", "world") == "world"
        if not world_aligned:
            observed, target = observed @ rotation, target @ rotation
        error = torch.zeros_like(position)
        if augment:
            observed, error = augment_measurement(observed, error, config)
            observed = observed - error
        estimated = position + (error if world_aligned else rotation @ error)
        transforms.append(relative_transform(estimated, quaternion, state.position, state.quaternion))
        observations.append(observed @ rotation if world_aligned else observed)
        targets.append(target)
        rotations.append(rotation)
        positions.append(position)
        estimated_positions.append(estimated)
        quaternions.append(quaternion)
    noise = config if augment and config.get("augmentation", "basic") != "none" else None
    outputs, predictions = relative_batch(model, observations, transforms, states, config, noise, rotations)
    losses, world_predictions, grids = [], [], []
    for i, (output, prediction, state) in enumerate(zip(outputs, predictions, states, strict=True)):
        state.position = estimated_positions[i].detach().clone()
        state.quaternion = quaternions[i].detach().clone()
        losses.append(supervised_loss(output, targets[i], quaternions[i], config))
        world_predictions.append((prediction if world_aligned else prediction @ rotations[i].T) + positions[i])
        if compute_metrics:
            measured = observations[i] @ rotations[i].T if world_aligned else observations[i]
            grids.append(
                tuple(
                    mask_metric_grid(voxelize(points, config["map_size"], config["resolution"]), quaternions[i], config)
                    for points in (prediction, measured, targets[i])
                )
            )
    return losses, world_predictions, grids
