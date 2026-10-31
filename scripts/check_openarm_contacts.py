"""Verify object-filtered finger contacts and critic observations in cloned scenes.

Run: uv run python scripts/check_openarm_contacts.py --viz none
Gravity is disabled only for this sensor test. A cube is placed in environment
0's gripper; other environments must not report finger-object contacts.
"""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from Qmini.tasks.manipulation.openarm import mdp
from Qmini.tasks.manipulation.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg
from Qmini.tasks.manipulation.openarm.mdp.state import tcp_pose_w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = OpenArmCubeLiftEnvCfg()
    cfg.scene.num_envs = 32
    cfg.seed = 42
    cfg.commands.object_pose.debug_vis = False
    for variant in cfg.scene.object.spawn.assets_cfg:
        variant.rigid_props.disable_gravity = True

    with launch_simulation(cfg, args):
        env = gym.make("Qmini-Lift-Cube-OpenArm", cfg=cfg)
        try:
            task = env.unwrapped
            env.reset()
            actions = torch.zeros_like(task.action_manager.action)
            actions[:, -1] = 1
            for _ in range(10):
                env.step(actions)
            assert not mdp.object_and_gripper_contacts(task)[:, -2:].any()

            position, orientation = tcp_pose_w(task)
            pose = torch.cat((position[:1], orientation[:1]), dim=-1)
            ids = torch.tensor([0], device=task.device)
            obj = task.scene["object"]
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=task.device), env_ids=ids)
            actions[:, -1] = -1
            detected = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            closing_progress = torch.zeros(task.num_envs, device=task.device)
            grasp_bonus = torch.zeros_like(closing_progress)
            peak_closing_progress = torch.zeros_like(closing_progress)
            stage = task.termination_manager.get_term_cfg("object_pose_success").func
            for _ in range(60):
                observations, *_ = env.step(actions)
                detected |= mdp.object_and_gripper_contacts(task)[:, -2:].bool().all(dim=-1)
                closing_progress += stage.rewards["closing_object"]
                peak_closing_progress = torch.maximum(peak_closing_progress, stage.rewards["closing_object"])
                grasp_bonus += stage.rewards["grasp_confirmed"] * cfg.rewards.grasp_confirmed.weight
                assert torch.isfinite(observations["critic"]).all()

            assert detected[0] and not detected[1:].any(), "Filtered contact missing or assigned to another environment"
            print(
                f"Closure progress: net={closing_progress[0].item():.4f}, "
                f"peak={peak_closing_progress[0].item():.4f}; grasp bonus={grasp_bonus[0].item():.1f}",
                flush=True,
            )
            assert peak_closing_progress[0] > 0, "Aligned physical closure must earn progress before grasp confirmation"
            # This gravity-free sensor fixture may only make transient contact;
            # it must never pay repeatedly or credit a different environment.
            assert ((grasp_bonus == 0) | (grasp_bonus == 5)).all() and not grasp_bonus[1:].any()
            print(f"Both-finger contact environments: {torch.nonzero(detected).flatten().tolist()}")
            print(f"Observation shapes: { {key: tuple(value.shape) for key, value in observations.items()} }")
        finally:
            env.close()


if __name__ == "__main__":
    main()
