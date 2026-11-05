"""Common ground-truth-free forward path for training feedback, evaluation and deployment."""

from ..geometry.sparse_voxel import decode_sparse, temporal_input
from ..geometry.voxel import decode, model_input, voxelize
from ..models.architectures import is_sparse


def run_network(
    model,
    observed_world,
    position,
    quaternion,
    previous_world,
    config,
    compute_metrics=True,
    observation_noise=None,
):
    size, resolution = config["map_size"], config["resolution"]
    if is_sparse(config):
        coordinates, features = temporal_input(
            observed_world - position, previous_world - position, size, resolution, observation_noise
        )
        output = model(coordinates, features)
        prediction = decode_sparse(output, size, resolution, config["threshold"], config["max_points"])
        observed = voxelize(observed_world - position, size, resolution) if compute_metrics else None
    else:
        inputs = model_input(observed_world, position, quaternion, previous_world, size, resolution)
        output = model(inputs)
        logits, offsets = output
        prediction = decode(logits[0], offsets[0], size, config["threshold"], config["max_points"])
        observed = inputs[0, :4]
    return prediction + position, output, observed
