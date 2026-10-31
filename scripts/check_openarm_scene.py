"""Check cube support in cloned OpenArm scenes without a trained policy.

Run: uv run python scripts/check_openarm_scene.py --viz none
Use --enable-collision-filter to reproduce the original collision filtering.
"""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from Qmini.tasks.manipulation.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg
from Qmini.tasks.manipulation.openarm.mdp.contacts import object_table_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--enable-collision-filter", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 25:
        parser.error("Use at least one environment and 25 steps to observe the cube settling.")

    cfg = OpenArmCubeLiftEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    if args.enable_collision_filter:
        cfg.scene.filter_collisions = True
    cfg.episode_length_s = (args.steps + 10) * cfg.sim.dt * cfg.decimation
    cfg.commands.object_pose.debug_vis = False

    with launch_simulation(cfg, args):
        env = gym.make("Qmini-Lift-Cube-OpenArm", cfg=cfg)
        try:
            task = env.unwrapped
            env.reset()
            actions = torch.zeros_like(task.action_manager.action)
            actions[:, -1] = 1.0  # Hold the default arm pose and open the gripper.
            dropped = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            for step in range(args.steps):
                with torch.inference_mode():
                    env.step(actions)
                    # Track termination before auto-reset hides the falling state.
                    step_drops = task.termination_manager.get_term("object_out_of_workspace")
                    dropped |= step_drops
                    if step_drops.any():
                        print(f"step={step + 1}, dropped={step_drops.sum().item()}", flush=True)

            height = task.scene["object"].data.root_pos_w.torch[:, 2] - task.scene.env_origins[:, 2]
            settled = (height > 0.005) & (height < 0.045)
            print(
                f"Cube support: dropped={dropped.sum().item()}/{task.num_envs}, "
                f"settled={settled.sum().item()}/{task.num_envs}, "
                f"height=[{height.min().item():.4f}, {height.max().item():.4f}] m",
                flush=True,
            )
            if dropped.any() or not settled.all():
                raise RuntimeError("Cube support check failed: inspect table/object collisions before training.")
            if not object_table_support(task).all():
                raise RuntimeError("External support detection missed settled cubes")
            # Move alternate cubes into free flight: support must disappear for
            # exactly those environments, without cross-talk.
            airborne = torch.arange(0, task.num_envs, 2, device=task.device)
            obj = task.scene["object"]
            pose = obj.data.root_pose_w.torch.clone()[airborne]
            pose[:, 2] += 0.3
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=airborne)
            obj.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(len(airborne), 6, device=task.device), env_ids=airborne
            )
            with torch.inference_mode():
                env.step(actions)
            expected = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
            expected[airborne] = False
            if not torch.equal(object_table_support(task), expected):
                raise RuntimeError("External support detected in free flight or missing in adjacent environments")
            print("Table support: settled support and per-environment free-flight rejection passed.", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
