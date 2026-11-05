"""Shared steps with recursive predicted maps and optional persistent GRU state."""

from pathlib import Path

import numpy as np
import torch

from ..datasets.augmentation import augment_measurement
from ..datasets.sequence import check_successor, load_frame
from ..geometry.sparse_voxel import sparse_reconstruction_loss, sparse_voxelize
from ..geometry.temporal import rotation_matrix
from ..geometry.voxel import voxelize
from ..models.architectures import is_recurrent, is_sparse
from ..models.dense3d import reconstruction_loss
from .inference import run_network
from .recurrent import mask_metric_grid, recorded_batch
from .state import RecurrentState


def forward_frame(model, frame, previous_world, config, device, augment=False, compute_metrics=True):
    def tensor(key):
        return torch.as_tensor(frame[key], dtype=torch.float32, device=device)

    observed, position, quaternion = tensor("lidar_points_w"), tensor("base_pos_w"), tensor("base_quat_xyzw")
    target_local = tensor("gt_points_w") - position
    if augment:
        observed, position = augment_measurement(observed, position, config)
    prediction, output, observed_grid = run_network(
        model,
        observed,
        position,
        quaternion,
        previous_world,
        config,
        compute_metrics=compute_metrics,
        observation_noise=config if augment and config.get("augmentation", "basic") != "none" else None,
    )
    target = (
        voxelize(target_local, config["map_size"], config["resolution"])
        if compute_metrics or not is_sparse(config)
        else None
    )
    if is_sparse(config):
        cells, features = sparse_voxelize(target_local, config["map_size"], config["resolution"])
        loss = sparse_reconstruction_loss(output, cells, features, config["resolution"], config["offset_weight"])
    else:
        logits, offsets = output
        loss = reconstruction_loss(logits, offsets, target[None], config["offset_weight"])
    return loss, prediction, observed_grid, target


class Scores:
    def __init__(self):
        self.counts = np.zeros(3, dtype=np.int64)
        self.offset_sum = 0.0
        self.matches = 0

    def update(self, prediction, target, voxel_size):
        p, t = prediction[0] > 0, target[0] > 0
        matched = p & t
        self.counts += [int(matched.sum()), int((p & ~t).sum()), int((~p & t).sum())]
        self.offset_sum += (
            float(torch.linalg.vector_norm(prediction[1:, matched] - target[1:, matched], dim=0).sum()) * voxel_size
        )
        self.matches += int(matched.sum())

    def result(self):
        tp, fp, fn = (int(value) for value in self.counts)
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "matched_centroid_error_cm": 100 * self.offset_sum / self.matches if self.matches else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }


@torch.inference_mode()
def evaluate(model, runs, config, device, max_frames=0, export=None, export_frames=0, progress=None, export_every=1):
    """Keep feedback across the full episode, clear only on episode boundaries."""
    if export_every < 1:
        raise ValueError("export_every must be positive")
    model.eval()
    scores, baseline = Scores(), Scores()
    extra_scores = (
        {name: Scores() for name in ("network_prediction", "retained_observations")}
        if config.get("observation_memory", False)
        else {}
    )
    frames, total_loss, episodes, exported = 0, 0.0, 0, 0
    available = sum(len(ep["frames"]) for run in runs for ep in run["episodes"])
    for run in runs:
        for episode in run["episodes"]:
            previous_world = torch.empty((0, 3), device=device)
            previous_frame = None
            recurrent_state = RecurrentState()
            episodes += 1
            for path in episode["frames"]:
                frame = load_frame(path)
                check_successor(previous_frame, frame)
                previous_frame = frame
                if is_recurrent(config):
                    losses, predictions, grids = recorded_batch(
                        model, [frame], [recurrent_state], config, device, augment=False, compute_metrics=True
                    )
                    loss, prediction = losses[0], predictions[0]
                    grid, observed, target = grids[0]
                else:
                    loss, prediction, observed, target = forward_frame(model, frame, previous_world, config, device)
                    previous_world = prediction
                    position = torch.as_tensor(frame["base_pos_w"], device=device)
                    grid = voxelize(prediction - position, config["map_size"], config["resolution"])
                scores.update(grid, target, config["voxel_size"])
                baseline.update(observed, target, config["voxel_size"])
                extra_exports = {}
                if extra_scores:
                    position = torch.as_tensor(frame["base_pos_w"], device=device, dtype=torch.float32)
                    quaternion = torch.as_tensor(frame["base_quat_xyzw"], device=device, dtype=torch.float32)
                    for name, score in extra_scores.items():
                        local = getattr(recurrent_state, name)
                        metric_grid = mask_metric_grid(
                            voxelize(local, config["map_size"], config["resolution"]), quaternion, config
                        )
                        score.update(metric_grid, target, config["voxel_size"])
                        world = (
                            local
                            if config.get("map_frame", "world") == "world"
                            else local @ rotation_matrix(quaternion).T
                        )
                        extra_exports[f"{name}_points_w"] = world + position
                frames += 1
                total_loss += float(loss)
                if progress is not None:
                    progress(frames)
                if export is not None and exported < export_frames and (frames - 1) % export_every == 0:
                    destination = (
                        Path(export)
                        / "predictions"
                        / run["name"]
                        / Path(episode["path"]).parent.name
                        / Path(episode["path"]).name
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        destination / Path(path).name,
                        prediction_points_w=prediction.cpu().numpy(),
                        lidar_points_w=frame["lidar_points_w"],
                        gt_points_w=frame["gt_points_w"],
                        base_pos_w=frame["base_pos_w"],
                        base_quat_xyzw=frame["base_quat_xyzw"],
                        time_s=frame["time_s"],
                        step=frame["step"],
                        **{name: points.cpu().numpy() for name, points in extra_exports.items()},
                    )
                    exported += 1
                if max_frames and frames >= max_frames:
                    break
            if max_frames and frames >= max_frames:
                break
        if max_frames and frames >= max_frames:
            break
    if not frames:
        raise ValueError("No evaluation frames")
    return {
        "frames": frames,
        "available_frames": available,
        "episodes_started": episodes,
        "partial_evaluation": frames < available,
        "loss": total_loss / frames,
        "prediction": scores.result(),
        "observed_only": baseline.result(),
        **{name: score.result() for name, score in extra_scores.items()},
    }
