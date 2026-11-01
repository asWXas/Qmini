"""Evaluate the current AS2W PPO configuration for a fixed, finite rollout.

Example: uv run python scripts/evaluate_as2w.py --checkpoint model.pt --terrain stairs_up --viz none
Checkpoints must match the current observation/action layout and PPO architecture.
"""

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains import inverted_pyramid_stairs_terrain
from isaaclab.utils import to_dict
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from Qmini.assets.as2w import LEG_JOINT_NAMES
from Qmini.tasks.quadruped.as2w.agent.rsl_rl_ppo_cfg import AS2WRoughPPORunnerCfg
from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import AS2WRoughEnvCfg

WHEEL_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
SETTLE_STEPS = 25


def evaluation_config(args):
    """Create repeatable forward commands and return the actual first-riser offset."""
    cfg = AS2WRoughEnvCfg()
    # Direct construction bypasses the task CLI's preset resolution. Wheel slip
    # observations/rewards require the concrete PhysX backend in this evaluator.
    cfg.sim.physics = cfg.sim.physics.isaacsim_physx
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 8.0
    cfg.episode_length_s = (args.steps + SETTLE_STEPS + 10) * cfg.sim.dt * cfg.decimation
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 0
    first_riser = None
    if args.terrain == "flat":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.max_init_terrain_level = None
        for name in ("fl_wheel_contact", "fr_wheel_contact", "rl_wheel_contact", "rr_wheel_contact"):
            getattr(cfg.scene, name).filter_prim_paths_expr = ["/World/ground/terrain/GroundPlane/CollisionPlane"]
    else:
        stairs = MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
            step_height_range=(args.step_height, args.step_height),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        )
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            size=stairs.size,
            border_width=20.0,
            num_rows=1,
            num_cols=args.num_envs,
            curriculum=False,
            use_cache=False,
            sub_terrains={"stairs_up": stairs},
        )
        # Native stair construction rounds the number of treads. Read its actual
        # center-floor box, rather than assuming platform_width / 2 is the riser.
        meshes, origin = inverted_pyramid_stairs_terrain(0.0, stairs)
        first_riser = float(meshes[-1].bounds[1, 0] - origin[0])

    command = cfg.commands.base_velocity
    command.debug_vis = False
    command.xbox = False
    command.rel_standing_envs = 0.0
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.ranges.heading = None
    command.moving_proportions = {"x": 1.0, "y": 0.0, "yaw": 0.0, "mix": 0.0}
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (cfg.episode_length_s + 1.0, cfg.episode_length_s + 1.0)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    # Evaluate nominal mechanics, retaining the CoM event used by observations.
    cfg.events.physics_material.params.update(
        static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0), restitution_range=(0.0, 0.0)
    )
    cfg.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.0)
    cfg.events.base_com.params["com_range"] = {axis: (0.0, 0.0) for axis in ("x", "y", "z")}
    cfg.events.joint_pd_gains.params.update(
        stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0)
    )
    return cfg, first_riser


def evaluate(env, runner, args, first_riser):
    """Measure only each environment's first trial; exclude states after auto-reset."""
    task = env.unwrapped
    robot = task.scene["robot"]
    sensor = task.scene.sensors["contact_forces"]
    leg_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    wheel_ids, _ = sensor.find_sensors(WHEEL_BODIES, preserve_order=True)
    policy = runner.get_inference_policy(device=task.device)
    failed = torch.zeros(task.num_envs, device=task.device, dtype=torch.bool)
    with torch.inference_mode():
        obs = env.get_observations()
        for _ in range(SETTLE_STEPS):
            obs, _, done, _ = env.step(torch.zeros_like(task.action_manager.action))
            failed |= done.bool()
        policy.reset(torch.ones_like(failed))
        start = robot.data.root_pos_w.torch.clone()
        last = start.clone()
        max_progress = torch.zeros(task.num_envs, device=task.device)
        max_rise = torch.zeros_like(max_progress)
        crossed = torch.zeros_like(failed)
        count = torch.zeros_like(max_progress)
        contact_count = torch.zeros_like(max_progress)
        leg_speed_sq = torch.zeros_like(max_progress)
        roll_sq = torch.zeros_like(max_progress)
        pitch_sq = torch.zeros_like(max_progress)
        for _ in range(args.steps):
            if not all(torch.isfinite(value).all() for value in obs.values()):
                raise RuntimeError("Non-finite observations during evaluation")
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Non-finite policy actions during evaluation")
            obs, _, done, _ = env.step(actions)
            policy.reset(done)
            failed |= done.bool()
            active = ~failed
            pos = robot.data.root_pos_w.torch
            last[active] = pos[active]
            progress = pos[:, 0] - start[:, 0]
            rise = pos[:, 2] - start[:, 2]
            max_progress[active] = torch.maximum(max_progress[active], progress[active])
            max_rise[active] = torch.maximum(max_rise[active], rise[active])
            if first_riser is not None:
                # The base must move past the first +X riser and rise at the same
                # instant. This measures first-riser crossing, not full-stair completion.
                crossed |= (
                    active
                    & (pos[:, 0] - task.scene.env_origins[:, 0] > first_riser + 0.1)
                    & (rise > 0.5 * args.step_height)
                )
            forces = sensor.data.net_forces_w.torch[:, wheel_ids]
            all_contact = (torch.linalg.vector_norm(forces, dim=-1) > 1.0).all(dim=-1)
            leg_sq = robot.data.joint_vel.torch[:, leg_ids].square().mean(dim=-1)
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w.torch)
            count += active
            contact_count += all_contact & active
            leg_speed_sq += torch.where(active, leg_sq, 0.0)
            roll_sq += torch.where(active, roll.square(), 0.0)
            pitch_sq += torch.where(active, pitch.square(), 0.0)

    samples = count.sum().item()

    def rms(total):
        return math.sqrt(total.sum().item() / samples) if samples else None

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "terrain": args.terrain,
        "num_envs": task.num_envs,
        "steps": args.steps,
        "duration_s": args.steps * task.step_dt,
        "command_speed_m_s": args.speed,
        "failures": failed.sum().item(),
        "failure_fraction": failed.float().mean().item(),
        "mean_forward_progress_m": (last[:, 0] - start[:, 0]).mean().item(),
        "mean_max_forward_progress_m": max_progress.mean().item(),
        "mean_final_base_rise_m": (last[:, 2] - start[:, 2]).mean().item(),
        "mean_max_base_rise_m": max_rise.mean().item(),
        "all_wheel_contact_fraction": contact_count.sum().item() / samples if samples else None,
        "leg_joint_speed_rms_rad_s": rms(leg_speed_sq),
        "base_roll_rms_rad": rms(roll_sq),
        "base_pitch_rms_rad": rms(pitch_sq),
        "valid_env_steps": int(samples),
        "forward_progress_per_env_m": (last[:, 0] - start[:, 0]).cpu().tolist(),
        "failed_per_env": failed.cpu().tolist(),
    }
    if first_riser is not None:
        summary.update(
            step_height_m=args.step_height,
            first_riser_x_from_origin_m=first_riser,
            first_riser_crossed_fraction=crossed.float().mean().item(),
            first_riser_crossed_without_failure_fraction=(crossed & ~failed).float().mean().item(),
            first_riser_criterion="base x > first riser + 0.1 m and simultaneous base rise > half step height",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--terrain", choices=("flat", "stairs_up"), default="stairs_up")
    parser.add_argument("--step_height", type=float, default=0.1)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num_envs and --steps must be positive")
    if not math.isfinite(args.step_height) or not 0.0 < args.step_height <= 0.3:
        parser.error("--step_height must be in (0, 0.3] meters")
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be finite and positive")
    if not Path(args.checkpoint).is_file():
        parser.error("--checkpoint must be an existing file")

    cfg, first_riser = evaluation_config(args)
    with launch_simulation(cfg, args):
        raw_env = gym.make("Qmini-Velocity-Rough-AS2W", cfg=cfg)
        try:
            agent_cfg = AS2WRoughPPORunnerCfg()
            agent_cfg.device = raw_env.unwrapped.device
            env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(
                args.checkpoint,
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
                map_location=agent_cfg.device,
            )
            summary = evaluate(env, runner, args, first_riser)
            result = json.dumps(summary, indent=2, allow_nan=False)
            print(result, flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(result + "\n", encoding="utf-8")
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
