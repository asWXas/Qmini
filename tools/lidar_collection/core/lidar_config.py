"""Head-mounted hemisphere and MID-360-FOV approximation for terrain ray casting."""

import torch

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .lidar_geometry import rpy_quaternion, spherical_directions


def spherical_pattern(cfg, device):
    directions = torch.as_tensor(spherical_directions(cfg.num_rays, cfg.elevation_range), device=device)
    return torch.zeros_like(directions), directions


@configclass
class SphericalPatternCfg(patterns.PatternBaseCfg):
    func = spherical_pattern
    num_rays: int = 12000
    elevation_range: tuple[float, float] = (0.0, 90.0)


def make_lidar_config(prim_path, args):
    defaults = {"hemisphere": (0.0, 90.0), "mid360": (-7.0, 52.0), "rings": (-30.0, 10.0)}
    elevation = tuple(args.vertical_fov) if args.vertical_fov is not None else defaults[args.lidar_pattern]
    if args.lidar_pattern == "rings":
        pattern = patterns.LidarPatternCfg(
            channels=args.channels,
            vertical_fov_range=elevation,
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=args.horizontal_res,
        )
    else:
        pattern = SphericalPatternCfg(num_rays=args.lidar_rays, elevation_range=elevation)
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(
            pos=tuple(args.lidar_offset), rot=tuple(float(value) for value in rpy_quaternion(args.lidar_rpy))
        ),
        ray_alignment="base",
        update_period=0.0,
        mesh_prim_paths=["/World/ground"],
        max_distance=args.max_range,
        pattern_cfg=pattern,
        debug_vis=args.lidar_debug_vis,
    )
