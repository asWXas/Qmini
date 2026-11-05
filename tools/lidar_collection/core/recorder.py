"""Write synchronized point clouds and state trajectories, independent of policy loading."""

import json
from pathlib import Path

import numpy as np
import torch

from .episodes import EpisodePackets
from .geometry import sample_local_mesh
from .lidar_geometry import sensor_world_pose


def as_numpy(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy().copy()


def collect(env, policy, mesh, args, output):
    task = env.unwrapped
    robot = task.scene["robot"]
    lidar = task.scene["collection_lidar"]
    if mesh is None:
        raise RuntimeError("Collection requires a generated terrain mesh.")
    mesh.export(output / "terrain.ply")
    bounds = np.stack((mesh.triangles.min(axis=1), mesh.triangles.max(axis=1)), axis=1)
    rng = np.random.default_rng(args.seed)
    episodes = np.zeros(task.num_envs, dtype=np.int64)
    frames = np.zeros(task.num_envs, dtype=np.int64)
    episode_steps = np.zeros(task.num_envs, dtype=np.int64)
    dt = task.step_dt
    last_report_step = 0
    metadata = {
        "schema_version": 2,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "control_dt_s": dt,
        "lidar_interval_s": args.scan_every * dt,
        "point_frame": "world",
        "quaternion_order": "xyzw",
        "pose": "base frame to world; meters",
        "angular_velocity": "base frame; rad/s; ideal simulator state, not a noisy IMU measurement",
        "gt_region": "world-axis-aligned cube centered at current base position",
        "gt_sampling": "area-uniform mesh surface samples; includes occluded and internal mesh faces",
        "lidar_model": "instantaneous terrain-only rays; no self occlusion, motion distortion, or noise",
        "sensor_offset_pos": list(lidar.cfg.offset.pos),
        "sensor_offset_quat_xyzw": list(lidar.cfg.offset.rot),
        "packets": "each episode directory contains states.jsonl, episode.json and frame NPZs",
        "time": "simulation seconds; state sampled after step; reset states begin a new episode",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "states.jsonl").open("w") as states, EpisodePackets(output) as packets, torch.inference_mode():
        obs = env.get_observations()
        for step in range(1, args.steps + 1):
            obs, _, done, _ = env.step(policy(obs))
            policy.reset(done)
            resets = as_numpy(done).astype(bool)
            episodes += resets
            frames[resets] = 0
            episode_steps += 1
            episode_steps[resets] = 0
            pos = as_numpy(robot.data.root_pos_w)
            quat = as_numpy(robot.data.root_quat_w)
            omega = as_numpy(robot.data.root_ang_vel_b)
            for index in range(task.num_envs):
                record = {
                    "step": step,
                    "time_s": step * dt,
                    "env_id": index,
                    "episode_id": int(episodes[index]),
                    "reset": bool(resets[index]),
                    "episode_step": int(episode_steps[index]),
                    "episode_time_s": float(episode_steps[index] * dt),
                    "base_pos_w": pos[index].tolist(),
                    "base_quat_xyzw": quat[index].tolist(),
                    "angular_velocity_b": omega[index].tolist(),
                }
                states.write(json.dumps(record, allow_nan=False) + "\n")
                packets.write_state(index, int(episodes[index]), record)
            # Restart scan cadence for each episode; first valid post-reset step is a new frame.
            scan_due = ~resets & ((frames == 0) | ((episode_steps - 1) % args.scan_every == 0))
            if not scan_due.any():
                continue
            data = lidar.data
            hits = as_numpy(data.ray_hits_w)
            # This RayCaster reports tracked-prim pose; cfg.offset is baked into its rays.
            sensor_pos, sensor_quat = sensor_world_pose(
                as_numpy(data.pos_w), as_numpy(data.quat_w), lidar.cfg.offset.pos, lidar.cfg.offset.rot
            )
            for index in range(task.num_envs):
                # Auto-reset observations and ray buffers may belong to different instants.
                if not scan_due[index]:
                    continue
                distances = np.linalg.norm(hits[index] - sensor_pos[index], axis=1)
                valid = np.isfinite(hits[index]).all(axis=1)
                valid &= (distances >= args.min_range) & (distances <= args.max_range)
                gt = sample_local_mesh(mesh, bounds, pos[index], args.map_size, args.gt_density, rng)
                folder = output / f"env_{index:03d}" / f"episode_{episodes[index]:06d}"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    folder / f"frame_{frames[index]:06d}.npz",
                    step=np.int64(step),
                    time_s=np.float64(step * dt),
                    env_id=np.int64(index),
                    episode_id=episodes[index],
                    episode_step=episode_steps[index],
                    episode_time_s=np.float64(episode_steps[index] * dt),
                    lidar_points_w=hits[index][valid],
                    ray_indices=np.flatnonzero(valid),
                    base_pos_w=pos[index],
                    base_quat_xyzw=quat[index],
                    angular_velocity_b=omega[index],
                    sensor_pos_w=sensor_pos[index],
                    sensor_quat_xyzw=sensor_quat[index],
                    gt_points_w=gt,
                )
                frames[index] += 1
                packets.frame_saved(index)
            states.flush()
            if step - last_report_step >= args.scan_every * 10:
                print(f"Recorded step {step}/{args.steps} to {output}", flush=True)
                last_report_step = step
