"""Record terrain-only simulated LiDAR, base pose, angular velocity, and mesh supervision.

Run with --help for options. Requires a checkpoint matching the current AS2W environment.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym  # type: ignore
import numpy as np  # type: ignore

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

from Qmini.tasks.quadruped.as2w.env.as2w_env_cfg import AS2WRoughEnvCfg

from ..terrains.terrain import RESEARCH_KINDS, TERRAIN_KINDS
from ..terrains.terrain_config import RecordingTerrainGenerator, configure_terrain
from .config_audit import audit_training_config
from .lidar_config import make_lidar_config
from .policy import load_policy, make_agent_config
from .recorder import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--agent", choices=("gru", "mlp"), default="gru")
    parser.add_argument(
        "--cmd",
        choices=("random", "gamepad"),
        default="random",
        help="Velocity command source: random sampling or Linux Xbox-compatible gamepad",
    )
    parser.add_argument("--gamepad_path", default="/dev/input/js0", help="Linux joystick device")
    parser.add_argument("--gamepad_deadzone", type=float, default=0.1, help="Stick deadzone in [0, 1)")
    parser.add_argument("--gamepad_scale", type=float, default=2.0, help="Gamepad velocity command multiplier")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scan_every", type=int, default=5, help="LiDAR frame interval in control steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terrain", choices=("existing", *TERRAIN_KINDS), default="existing")
    parser.add_argument("--terrain_rows", type=int, help="Default 1 for research/research_complex, 2 otherwise")
    parser.add_argument("--terrain_cols", type=int, help="Default 1 for research/research_complex, 2 otherwise")
    parser.add_argument(
        "--terrain_difficulty", type=float, help="Fixed difficulty [0, 1]; research/research_complex default 0.5"
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--lidar_pattern", choices=("hemisphere", "mid360", "rings"), default="hemisphere")
    parser.add_argument("--lidar_rays", type=int, default=12000, help="Rays for spherical patterns")
    parser.add_argument("--vertical_fov", type=float, nargs=2, help="Override pattern elevation limits in degrees")
    parser.add_argument("--horizontal_res", type=float, default=1.0)
    parser.add_argument("--lidar_offset", type=float, nargs=3, default=(0.32, 0.0, 0.18))
    parser.add_argument(
        "--lidar_rpy", type=float, nargs=3, default=(0.0, 60.0, 0.0), help="Mount roll/pitch/yaw in degrees"
    )
    parser.add_argument("--lidar_debug_vis", action="store_true", help="Display ray hits in the active viewer")
    parser.add_argument("--min_range", type=float, default=0.2)
    parser.add_argument("--max_range", type=float, default=10.0)
    parser.add_argument("--map_size", type=float, default=3.2)
    parser.add_argument("--gt_density", type=float, default=4000, help="Ground-truth samples per square meter")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.terrain_rows is None:
        args.terrain_rows = 1 if args.terrain in RESEARCH_KINDS else 2
    if args.terrain_cols is None:
        args.terrain_cols = 1 if args.terrain in RESEARCH_KINDS else 2
    if args.terrain_difficulty is None and args.terrain in RESEARCH_KINDS:
        args.terrain_difficulty = 0.5
    if args.terrain_difficulty is not None and not 0 <= args.terrain_difficulty <= 1:
        parser.error("--terrain_difficulty must be in [0, 1]")
    if not 0 <= args.gamepad_deadzone < 1:
        parser.error("--gamepad_deadzone must be in [0, 1)")
    for name in (
        "num_envs",
        "terrain_rows",
        "terrain_cols",
        "steps",
        "scan_every",
        "gamepad_scale",
        "channels",
        "lidar_rays",
        "horizontal_res",
        "max_range",
        "map_size",
        "gt_density",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be finite and positive")
    if not 0 <= args.min_range < args.max_range:
        parser.error("Require 0 <= min_range < max_range")
    if args.vertical_fov is not None and not -90 <= args.vertical_fov[0] < args.vertical_fov[1] <= 90:
        parser.error("Require -90 <= vertical_fov[0] < vertical_fov[1] <= 90")
    if not np.isfinite((*args.lidar_offset, *args.lidar_rpy)).all() or args.horizontal_res > 360:
        parser.error("Invalid LiDAR offset or horizontal resolution")
    if not Path(args.checkpoint).is_file():
        parser.error("Checkpoint does not exist")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    cfg = AS2WRoughEnvCfg()
    cfg.play_mode()
    cfg.sim.physics = cfg.sim.physics.isaacsim_physx  # type: ignore
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.commands.base_velocity.xbox = args.cmd == "gamepad"
    cfg.commands.base_velocity.gamepad_path = args.gamepad_path
    cfg.commands.base_velocity.gamepad_deadzone = args.gamepad_deadzone
    if args.cmd == "gamepad":
        # Scale physical velocity commands, preserving the checkpoint's observation scaling.
        ranges = cfg.commands.base_velocity.ranges
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            setattr(ranges, name, tuple(value * args.gamepad_scale for value in getattr(ranges, name)))
    configure_terrain(cfg, args.terrain, args.seed, args.terrain_rows, args.terrain_cols, args.terrain_difficulty)
    cfg.scene.collection_lidar = make_lidar_config(cfg.scene.height_scanner.prim_path, args)  # type: ignore
    audit = audit_training_config(args.checkpoint, to_dict(cfg), to_dict(make_agent_config(args.agent)))
    with launch_simulation(cfg, args):
        raw_env = gym.make("Qmini-Velocity-Rough-AS2W", cfg=cfg)
        try:
            env, policy, agent_cfg = load_policy(raw_env, args.checkpoint, args.agent)
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "config_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            dump_yaml(str(args.output / "env.yaml"), cfg)
            dump_yaml(str(args.output / "agent.yaml"), agent_cfg)
            collect(env, policy, RecordingTerrainGenerator.mesh, args, args.output)
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
