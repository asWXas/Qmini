"""True sparse minibatches: independent histories, one 4D forward per time step."""

import torch

from ..datasets.augmentation import augment_measurement
from ..geometry.sparse_voxel import (
    SparsePrediction,
    decode_sparse,
    sparse_reconstruction_loss,
    sparse_voxelize,
    temporal_input,
)


def forward_sparse_batch(model, frames, histories, config, device, augment=True):
    coordinates, features, positions, targets = [], [], [], []
    size, resolution = config["map_size"], config["resolution"]
    noise = config if augment and config.get("augmentation", "basic") != "none" else None
    for batch, (frame, history) in enumerate(zip(frames, histories, strict=True)):
        position = torch.as_tensor(frame["base_pos_w"], dtype=torch.float32, device=device)
        observed = torch.as_tensor(frame["lidar_points_w"], dtype=torch.float32, device=device)
        target = torch.as_tensor(frame["gt_points_w"], dtype=torch.float32, device=device) - position
        if augment:
            observed, position = augment_measurement(observed, position, config)
        cells, offsets = temporal_input(observed - position, history - position, size, resolution, noise)
        cells[:, 0] = batch
        coordinates.append(cells)
        features.append(offsets)
        positions.append(position)
        targets.append(sparse_voxelize(target, size, resolution))
    output = model(torch.cat(coordinates), torch.cat(features))
    losses, predictions = [], []
    for batch, (position, (target_cells, target_offsets)) in enumerate(zip(positions, targets, strict=True)):
        stages = []
        for cells, logits, stride in output.stages:
            selected = cells[:, 0] == batch
            if selected.any():
                stages.append((cells[selected], logits[selected], stride))
        selected = output.coordinates[:, 0] == batch
        prediction = SparsePrediction(
            stages, output.coordinates[selected], output.logits[selected], output.offsets[selected]
        )
        losses.append(
            sparse_reconstruction_loss(prediction, target_cells, target_offsets, resolution, config["offset_weight"])
        )
        predictions.append(
            decode_sparse(prediction, size, resolution, config["threshold"], config["max_points"]) + position
        )
    return losses, predictions
