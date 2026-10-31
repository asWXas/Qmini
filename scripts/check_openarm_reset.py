"""Sample OpenArm resets and check moving-link contact forces, including self-contact.

Run: uv run python scripts/check_openarm_reset.py --viz none
This is a sampled validation of the configured scene, not a collision-free planner.
"""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from Qmini.tasks.manipulation.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--resets", type=int, default=20)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.resets < 1:
        parser.error("num_envs and resets must be positive")

    cfg = OpenArmCubeLiftEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.commands.object_pose.debug_vis = False
    cfg.scene.robot.spawn.activate_contact_sensors = True
    for properties in cfg.scene.robot.spawn.articulation_props:
        if isinstance(properties, PhysxArticulationCfg):
            properties.enabled_self_collisions = True
    # Exclude the fixed mounting base; include all moving links and the gripper.
    cfg.scene.reset_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_(link[1-7]|hand|left_finger|right_finger)",
    )
    with launch_simulation(cfg, args):
        env = gym.make("Qmini-Lift-Cube-OpenArm", cfg=cfg)
        try:
            task = env.unwrapped
            robot = task.scene["robot"]
            sensor = task.scene["reset_contacts"]
            maximum_force = 0.0
            for _ in range(args.resets):
                env.reset()
                # Step physics directly: preserve the reset targets, with no policy action.
                for _ in range(10):
                    task.scene.write_data_to_sim()
                    task.sim.step(render=False)
                    task.scene.update(cfg.sim.dt)
                    forces = sensor.data.net_forces_w.torch.norm(dim=-1)
                    maximum_force = max(maximum_force, forces.max().item())
                    if (forces > 0.01).any():
                        bodies = torch.nonzero((forces > 0.01).any(dim=0)).flatten().tolist()
                        names = [sensor.body_names[i] for i in bodies]
                        raise RuntimeError(f"Reset collision on {names}; peak force={maximum_force:.6f} N")
                assert torch.isfinite(robot.data.joint_pos.torch).all()
            print(f"Checked {args.num_envs * args.resets} resets; peak moving-link contact={maximum_force:.6f} N")
        finally:
            env.close()


if __name__ == "__main__":
    main()
