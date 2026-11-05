"""Stateful inference API: measured points and pose only, no ground truth."""

import numpy as np
import torch

from .models.architectures import is_recurrent, make_model
from .runtime.inference import run_network
from .runtime.observation_memory import validate_memory_config


class Reconstructor:
    def __init__(self, checkpoint, device="cpu", observation_memory=None):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = dict(saved["config"])
        if observation_memory is not None:
            self.config["observation_memory"] = bool(observation_memory)
        validate_memory_config(self.config)
        self.model = make_model(self.config).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.reset()

    def reset(self):
        """Clear history; plane checkpoints re-seed at the NEXT received pose.

        Call on every robot reset, localization discontinuity or new trajectory.
        """
        self.previous_world = torch.empty((0, 3), device=self.device)
        if is_recurrent(self.config):
            from .runtime.state import RecurrentState

            self.recurrent_state = RecurrentState()

    @torch.inference_mode()
    def update(self, lidar_points_w, base_pos_w, base_quat_xyzw):
        points = torch.as_tensor(np.asarray(lidar_points_w), dtype=torch.float32, device=self.device)
        position = torch.as_tensor(np.asarray(base_pos_w), dtype=torch.float32, device=self.device)
        quaternion = torch.as_tensor(np.asarray(base_quat_xyzw), dtype=torch.float32, device=self.device)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("lidar_points_w must have shape (N, 3)")
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Expected position (3,) and quaternion xyzw (4,)")
        if not torch.isfinite(position).all() or not torch.isfinite(quaternion).all() or quaternion.norm() < 1e-6:
            raise ValueError("Invalid pose")
        if is_recurrent(self.config):
            from .geometry.temporal import rotation_matrix
            from .runtime.recurrent import relative_batch
            from .runtime.state import relative_transform

            rotation = rotation_matrix(quaternion)
            state = self.recurrent_state
            transform = relative_transform(position, quaternion, state.position, state.quaternion)
            _, predictions = relative_batch(
                self.model, [(points - position) @ rotation], [transform], [state], self.config, orientations=[rotation]
            )
            state.position, state.quaternion = position.detach().clone(), quaternion.detach().clone()
            local = predictions[0] if self.config.get("map_frame", "world") == "world" else predictions[0] @ rotation.T
            return (local + position).cpu().numpy().copy()
        self.previous_world, _, _ = run_network(
            self.model,
            points,
            position,
            quaternion,
            self.previous_world,
            self.config,
        )
        return self.previous_world.cpu().numpy().copy()


class RelativeReconstructor:
    """V3 deployment: current body-frame XYZ and T_current_from_previous only.

    Outputs are centered local world-axis maps; the state owns their five feature scales.
    No absolute/navigation translation is required. With observation memory, relative
    motion is integrated into a private cache anchor. Call reset on robot reset,
    localization discontinuity, stream restart or switching to a different robot.
    """

    def __init__(self, checkpoint, device="cpu", initial_quaternion_xyzw=None, observation_memory=None):
        from .runtime.state import RecurrentState

        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = dict(saved["config"])
        if observation_memory is not None:
            self.config["observation_memory"] = bool(observation_memory)
        validate_memory_config(self.config)
        if not is_recurrent(self.config):
            raise ValueError("RelativeReconstructor requires a v3 checkpoint")
        self.model = make_model(self.config).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.state = RecurrentState()
        self.initial_orientation = None
        if initial_quaternion_xyzw is not None:
            from .geometry.temporal import rotation_matrix

            q = torch.as_tensor(initial_quaternion_xyzw, device=self.device, dtype=torch.float32)
            if q.shape != (4,) or not torch.isfinite(q).all() or q.norm() < 1e-6:
                raise ValueError("Invalid initial quaternion xyzw")
            self.initial_orientation = rotation_matrix(q)

    def reset(self, initial_quaternion_xyzw=None):
        from .runtime.state import RecurrentState

        self.state = RecurrentState()
        self.initial_orientation = None
        if initial_quaternion_xyzw is not None:
            from .geometry.temporal import rotation_matrix

            q = torch.as_tensor(initial_quaternion_xyzw, device=self.device, dtype=torch.float32)
            if q.shape != (4,) or not torch.isfinite(q).all() or q.norm() < 1e-6:
                raise ValueError("Invalid initial quaternion xyzw")
            self.initial_orientation = rotation_matrix(q)

    @torch.inference_mode()
    def update(self, lidar_points_body, T_current_from_previous=None):
        from .runtime.recurrent import relative_batch
        from .runtime.state import validate_transform

        points = torch.as_tensor(np.asarray(lidar_points_body), dtype=torch.float32, device=self.device)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("lidar_points_body must have shape (N,3)")
        identity = torch.eye(4, dtype=torch.float32, device=self.device)
        if T_current_from_previous is None:
            if self.state.started:
                raise ValueError("Each subsequent frame requires T_current_from_previous")
            transform = identity
        else:
            transform = torch.as_tensor(np.asarray(T_current_from_previous), dtype=torch.float32, device=self.device)
            validate_transform(transform)
        if not self.state.started and not torch.allclose(transform, identity, atol=1e-5):
            raise ValueError("First frame/reset requires identity transform (or None)")
        orientations = (
            [self.initial_orientation] if not self.state.started and self.initial_orientation is not None else None
        )
        _, predictions = relative_batch(
            self.model, [points], [transform], [self.state], self.config, orientations=orientations
        )
        return predictions[0].cpu().numpy().copy()
