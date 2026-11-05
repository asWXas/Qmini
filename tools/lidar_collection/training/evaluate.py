"""Evaluate full episodes and optionally export predicted point clouds."""

import argparse
import json
from pathlib import Path

import torch

from .datasets.sequence import discover_runs
from .models.architectures import make_model
from .runtime.engine import evaluate
from .runtime.observation_memory import validate_memory_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evaluation directory")
    parser.add_argument("--data", type=Path, help="Optional new dataset root; default uses checkpoint validation split")
    parser.add_argument("--runs", nargs="+", help="Select run names within --data")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument(
        "--observation_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain measured surfaces; can enable on existing v3 weights without retraining",
    )
    parser.add_argument("--memory_size", type=float)
    parser.add_argument("--memory_max_voxels", type=int)
    parser.add_argument("--export_frames", type=int, default=10)
    parser.add_argument(
        "--export_every", type=int, default=1, help="Export every Nth evaluated frame; inference stays continuous"
    )
    parser.add_argument("--cpu_threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new directory")
    if min(args.max_frames, args.export_frames) < 0 or args.cpu_threads < 1 or args.export_every < 1:
        parser.error("Invalid frame limits or cpu_threads")
    if args.runs and args.data is None:
        parser.error("--runs requires --data")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    for key in ("observation_memory", "memory_size", "memory_max_voxels"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    try:
        validate_memory_config(config)
    except ValueError as error:
        parser.error(str(error))
    runs = discover_runs(args.data, config["map_size"]) if args.data else checkpoint["manifest"]["validation"]
    if args.runs:
        if set(args.runs) - {r["name"] for r in runs}:
            parser.error("Unknown run names")
        runs = [run for run in runs if run["name"] in args.runs]
    trained_hashes = {r["terrain_hash"] for r in checkpoint["manifest"]["train"]}
    overlaps = [r["name"] for r in runs if r["terrain_hash"] in trained_hashes]
    try:
        model = make_model(config).to(device)
    except (ImportError, ValueError) as error:
        parser.error(str(error))
    model.load_state_dict(checkpoint["model"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = evaluate(
        model, runs, config, device, args.max_frames, args.output, args.export_frames, export_every=args.export_every
    )
    result.update(
        checkpoint=str(args.checkpoint.resolve()),
        runs=[r["name"] for r in runs],
        training_terrain_overlap=overlaps,
        held_out_terrain=not bool(overlaps),
        checkpoint_epoch=checkpoint.get("epoch"),
        architecture=config["architecture"],
        export_every=args.export_every,
    )
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
