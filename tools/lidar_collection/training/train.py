"""Train paper-method 4D sparse reconstruction or the optional dense 3D baseline."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets.augmentation import mirror_sequence
from .datasets.sequence import SequenceDataset, collate_sequences, discover_runs, split_runs
from .geometry.temporal import DEFAULT_PLANE_Z
from .models.architectures import ARCHITECTURES, make_model
from .runtime.batched import forward_sparse_batch
from .runtime.display import TrainingDisplay
from .runtime.distributed import (
    DistributedContext,
    ParallelBatchSampler,
    broadcast_buffers,
    merge_validation,
    synchronize_gradients,
    validation_shards,
)
from .runtime.engine import evaluate, forward_frame
from .runtime.observation_memory import validate_memory_config
from .runtime.recurrent import recorded_batch
from .runtime.scheduling import SCHEDULE_DEFAULTS, make_scheduler, step_scheduler
from .runtime.state import RecurrentState


def checkpoint_save(path, content):
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/runs")
    parser.add_argument("--output", type=Path, required=True, help="New training directory")
    parser.add_argument("--model", choices=tuple(ARCHITECTURES), help="Default: sparse4d_v3; dense3d is the baseline")
    parser.add_argument(
        "--val_runs", nargs="+", help="Hold out these runs AND all runs with the identical terrain mesh"
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30, help="Total target epoch count, including resumed epochs")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument(
        "--canvas",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="v3: add a distinct canvas source every frame; new v3 training defaults on",
    )
    parser.add_argument(
        "--canvas_mode", choices=("plane", "volume"), help="Default plane; volume activates all XYZ cells"
    )
    parser.add_argument(
        "--occupancy_loss", choices=("bce", "balanced"), help="Default balanced for volume canvas, else bce"
    )
    parser.add_argument(
        "--observation_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="v3 comparison only: fuse cached measurements into outputs; new training defaults off",
    )
    parser.add_argument("--memory_size", type=float, help="Measurement cache side length in meters, default 4*map_size")
    parser.add_argument("--memory_max_voxels", type=int, help="Bound retained measurement cells, default 200000")

    parser.add_argument(
        "--window_sampling", choices=("sequential", "random"), help="Default sparse4d=random, dense3d=sequential"
    )
    parser.add_argument(
        "--history_init",
        choices=("empty", "previous_gt", "plane", "current"),
        help="v3 default: plane; current/current is optional for v3; previous_gt is legacy training only",
    )
    parser.add_argument(
        "--plane_z",
        type=float,
        help="Plane surface height relative to first base position, world-up meters; default -0.38",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Sequences per GPU per step; sparse4d uses true minibatches; dense3d accumulates serially",
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader processes PER GPU")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--allow_world_size_change",
        action="store_true",
        help="Explicitly allow resume with a different GPU count/global batch",
    )
    parser.add_argument(
        "--relative_pose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="v3: optional relative SE(3) embedding; new training defaults off, resume inherits checkpoint",
    )
    parser.add_argument(
        "--bptt_steps", type=int, help="v3 hidden-state gradient window; default 4, 1 detaches every frame"
    )
    parser.add_argument("--base_channels", type=int, help="Default: sparse4d=16, dense3d=8")
    parser.add_argument("--map_size", type=float, default=3.2)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--lr", type=float, help="Default: sparse4d=0.01, dense3d=0.001")
    parser.add_argument("--decay", type=float, help="Default: sparse4d decays to 0.0001 by final epoch; dense3d=0.98")
    parser.add_argument(
        "--lr_scheduler",
        choices=("plateau", "exponential"),
        help="New sparse4d default: plateau; legacy resume keeps its scheduler",
    )
    parser.add_argument("--lr_metric", choices=("val_loss", "val_f1"))
    parser.add_argument("--lr_factor", type=float, help="Plateau LR multiplier, default 0.5")
    parser.add_argument("--lr_patience", type=int, help="Tolerated bad validation epochs, default 3")
    parser.add_argument("--lr_threshold", type=float, help="Minimum absolute validation improvement, default 0.001")
    parser.add_argument("--min_lr", type=float, help="Plateau minimum LR, default 0.0001")
    parser.add_argument(
        "--reset_scheduler",
        action="store_true",
        help="On resume, reset scheduler only; retain weights, Adam and current LR",
    )
    parser.add_argument("--offset_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, help="0 disables output cap; default sparse4d=0, dense3d=20000")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Training-only Gaussian point noise, meters")
    parser.add_argument("--point_dropout", type=float, default=0.1)
    parser.add_argument(
        "--augmentation", choices=("paper", "basic", "none"), help="Default: sparse4d=paper, dense3d=basic"
    )
    parser.add_argument("--point_noise_bound", type=float, help="Paper uniform XYZ noise bound in meters; default 0.05")
    parser.add_argument(
        "--voxel_dropout", type=float, help="Sparse training current-observation voxel dropout; default 0"
    )
    parser.add_argument("--voxel_noise_std", type=float, help="Sparse centroid jitter std in voxel units; default 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0, help="Smoke-test limit; 0 means all")
    parser.add_argument("--max_val_frames", type=int, default=0, help="Smoke-test limit; 0 means all")
    parser.add_argument(
        "--resume", type=Path, help="Resume into a NEW output directory; retain identical data/model settings"
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    saved = checkpoint["config"] if checkpoint else {}
    if checkpoint and saved.get("architecture") not in ARCHITECTURES.values():
        parser.error("Unsupported/removed checkpoint architecture; v2 is removed, use a matching v1 or v3 checkpoint")
    if args.model is None:
        args.model = next(
            (name for name, value in ARCHITECTURES.items() if value == saved.get("architecture")), "sparse4d_v3"
        )
    sparse = args.model != "dense3d"
    recurrent = args.model == "sparse4d_v3"
    for key in ("canvas", "observation_memory"):
        if getattr(args, key) is None:
            default = recurrent and key == "canvas" and not saved
            setattr(args, key, saved.get(key, default))
        if getattr(args, key) and not recurrent:
            parser.error(f"{key} requires sparse4d_v3")
    if args.memory_size is None:
        args.memory_size = saved.get("memory_size", args.map_size * 4)
    if args.memory_max_voxels is None:
        args.memory_max_voxels = saved.get("memory_max_voxels", 200000)
    if saved and saved.get("canvas", False) != args.canvas:
        parser.error("Canvas changes source fusion weights; start a new training run instead of --resume")
    if args.canvas_mode is None:
        args.canvas_mode = saved.get("canvas_mode", "plane")
    if not args.canvas and args.canvas_mode == "volume":
        parser.error("--canvas_mode volume requires --canvas")
    if args.occupancy_loss is None:
        args.occupancy_loss = saved.get(
            "occupancy_loss",
            "bce" if saved else ("balanced" if args.canvas and args.canvas_mode == "volume" else "bce"),
        )
    if args.occupancy_loss != "bce" and not recurrent:
        parser.error("Balanced occupancy loss is currently supported by sparse4d_v3 only")
    if saved and any(
        saved.get(key, default) != getattr(args, key)
        for key, default in (("canvas_mode", "plane"), ("occupancy_loss", "bce"))
    ):
        parser.error("Canvas/loss recipe changed; start a new training run instead of --resume")

    if args.relative_pose is None:
        args.relative_pose = saved.get("relative_pose", bool(saved) and recurrent)
    if args.bptt_steps is None:
        args.bptt_steps = saved.get("bptt_steps", 4 if recurrent else 1)
    if args.relative_pose and not recurrent:
        parser.error("Relative pose embedding is only supported by sparse4d_v3")
    if args.lr_scheduler is None:
        args.lr_scheduler = saved.get("lr_scheduler", "exponential" if saved or not sparse else "plateau")
    for key, default in SCHEDULE_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, default))
    if args.reset_scheduler and not args.resume:
        parser.error("--reset_scheduler requires --resume")
    if args.augmentation is None:
        args.augmentation = saved.get("augmentation", "basic" if saved else ("paper" if sparse else "basic"))
    for key, default in (
        ("window_sampling", "random" if sparse else "sequential"),
        ("history_init", "plane" if recurrent else ("previous_gt" if sparse else "empty")),
    ):
        if getattr(args, key) is None:
            legacy = "sequential" if key == "window_sampling" else "empty"
            setattr(args, key, saved.get(key, legacy if saved else default))
    defaults = {
        "base_channels": 16 if sparse else 8,
        "lr": 0.01 if sparse else 0.001,
        "max_points": 0 if sparse else 20000,
        "point_noise_bound": 0.05,
        "voxel_dropout": 0.0,
        "voxel_noise_std": 0.0,
        "plane_z": DEFAULT_PLANE_Z,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, value))
    if args.decay is None:
        args.decay = saved.get(
            "decay", (0.0001 / args.lr) ** (1 / max(args.epochs, 1)) if sparse and args.lr > 0 else 0.98
        )
    for key in ("epochs", "seq_len", "batch_size", "base_channels", "cpu_threads", "prefetch_factor", "bptt_steps"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    for key in ("map_size", "voxel_size", "lr", "decay", "offset_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if args.history_init == "plane" and not recurrent:
        parser.error("Plane startup requires --model sparse4d_v3")
    if args.history_init == "current" and not recurrent:
        parser.error("Current/current startup requires --model sparse4d_v3")
    if recurrent and args.history_init == "previous_gt":
        parser.error("v3 uses deployment-consistent initialization: current, empty or plane")
    if not recurrent and args.bptt_steps != 1:
        parser.error("bptt_steps > 1 requires --model sparse4d_v3")
    if not math.isfinite(args.plane_z) or (
        (args.history_init == "plane" or (args.canvas and args.canvas_mode == "plane"))
        and not -args.map_size / 2 <= args.plane_z < args.map_size / 2
    ):
        parser.error("plane_z must be finite and inside the local map for plane initialization")
    if args.base_channels % 4 or not 0 < args.threshold < 1 or not 0 <= args.point_dropout < 1:
        parser.error("Require base_channels multiple of 4, threshold in (0,1), dropout in [0,1)")
    if not math.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("noise_std must be finite and nonnegative")
    if min(args.workers, args.max_train_batches, args.max_val_frames, args.max_points) < 0:
        parser.error("Workers, output cap and smoke-test limits must be nonnegative")
    if not 0 <= args.voxel_dropout < 1 or not math.isfinite(args.voxel_noise_std) or args.voxel_noise_std < 0:
        parser.error("Require voxel_dropout in [0,1), finite nonnegative voxel_noise_std")
    if not math.isfinite(args.point_noise_bound) or args.point_noise_bound < 0:
        parser.error("point_noise_bound must be finite and nonnegative")
    if not sparse and (args.voxel_dropout or args.voxel_noise_std):
        parser.error("Voxel augmentation is currently supported by sparse4d; dense3d supports point noise")
    if args.lr_patience < 0 or not 0 < args.lr_factor < 1:
        parser.error("Require lr_patience >= 0 and lr_factor in (0,1)")
    if not math.isfinite(args.lr_threshold) or args.lr_threshold < 0:
        parser.error("lr_threshold must be finite and nonnegative")
    if not math.isfinite(args.min_lr) or args.min_lr <= 0 or (args.lr_scheduler == "plateau" and args.min_lr > args.lr):
        parser.error("Require finite 0 < min_lr <= lr")
    resolution = round(args.map_size / args.voxel_size)
    stride = 16 if sparse else 8
    if resolution < stride or resolution % stride or not math.isclose(resolution * args.voxel_size, args.map_size):
        parser.error(f"map_size / voxel_size must be an integer multiple of {stride}, at least {stride}")
    if args.output.exists():
        parser.error("Output exists; select a new directory")
    parallel = DistributedContext(args.device)
    device = parallel.device
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu for a smoke test")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        architecture=ARCHITECTURES[args.model],
        resolution=resolution,
        torch_version=str(torch.__version__),
        world_size=parallel.world_size,
        global_batch_size=args.batch_size * parallel.world_size,
    )
    if recurrent:
        config.update(map_frame="world", update_radius=None, gt_map_size=args.map_size)
    try:
        validate_memory_config(config)
        model = make_model(config).to(device)
    except (ImportError, ValueError) as error:
        parser.error(str(error))
    try:
        manifest = split_runs(discover_runs(args.data, args.map_size), args.val_runs, args.seed)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    dataset = SequenceDataset(
        manifest["train"],
        args.seq_len,
        args.window_sampling,
        args.history_init,
        args.seed,
    )
    sampler = (
        ParallelBatchSampler(len(dataset), args.batch_size, parallel.rank, parallel.world_size, args.seed)
        if parallel.world_size > 1
        else None
    )
    loading = dict(num_workers=args.workers, collate_fn=collate_sequences, generator=generator)
    if args.workers:
        # CUDA has been initialized; fork would inherit unsafe CUDA/thread state.
        loading.update(multiprocessing_context="spawn", prefetch_factor=args.prefetch_factor)
    loader = (
        DataLoader(dataset, batch_sampler=sampler, **loading)
        if sampler
        else DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loading)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, config)
    start_epoch, best_f1 = 0, -1.0
    if args.resume:
        if saved["architecture"] != config["architecture"]:
            parser.error("Architecture changed; start a new training run or select the original model version")
        if recurrent and any(
            saved.get(key) != config[key] for key in ("relative_pose", "bptt_steps", "map_frame", "gt_map_size")
        ):
            parser.error("Resume recurrent configuration mismatch")
        if not args.reset_scheduler:
            old_kind = saved.get("lr_scheduler", "exponential")
            if old_kind != args.lr_scheduler:
                parser.error("Scheduler changed: add --reset_scheduler to intentionally restart its state")
            if old_kind == "plateau":
                for key, default in SCHEDULE_DEFAULTS.items():
                    if saved.get(key, default) != getattr(args, key):
                        parser.error(f"Scheduler setting changed: {key}; add --reset_scheduler")
        saved_world = saved.get("world_size", 1)
        if saved_world != parallel.world_size and not args.allow_world_size_change:
            parser.error(
                "GPU count changed: add --allow_world_size_change to accept a different global batch and data order"
            )
        for key, legacy in (("window_sampling", "sequential"), ("history_init", "empty")):
            if saved.get(key, legacy) != getattr(args, key):
                parser.error(f"Resume configuration mismatch: {key}; start a new training run for a new recipe")
        if args.history_init == "plane" and saved.get("plane_z", DEFAULT_PLANE_Z) != args.plane_z:
            parser.error("Resume configuration mismatch: plane_z")
        for key, legacy in (("point_noise_bound", 0.05), ("voxel_dropout", 0.0), ("voxel_noise_std", 0.0)):
            if saved.get(key, legacy) != getattr(args, key):
                parser.error(f"Resume configuration mismatch: {key}")
        if saved.get("augmentation", "basic") != args.augmentation:
            parser.error("Resume configuration mismatch: augmentation")
        for key in (
            "architecture",
            "resolution",
            "base_channels",
            "map_size",
            "voxel_size",
            "seq_len",
            "batch_size",
            "lr",
            "decay",
            "offset_weight",
            "threshold",
            "max_points",
            "noise_std",
            "point_dropout",
            "max_val_frames",
        ):
            if key == "decay" and args.reset_scheduler:
                continue
            if checkpoint["config"][key] != config[key]:
                parser.error(f"Resume configuration mismatch: {key}")
        if checkpoint["manifest"] != manifest:
            parser.error("Resume dataset split/file list changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.reset_scheduler:
            try:
                scheduler = make_scheduler(optimizer, config)
            except ValueError as error:
                parser.error(str(error))
        else:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_f1 = checkpoint["epoch"], checkpoint["best_f1"]
        if args.epochs <= start_epoch:
            parser.error("epochs must exceed the resumed epoch")
        if "rank_rng" in checkpoint and saved_world == parallel.world_size:
            state = checkpoint["rank_rng"][parallel.rank]
            torch.set_rng_state(state["torch_rng"])
            generator.set_state(state["loader_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
        elif parallel.world_size == 1 and saved_world == 1:
            torch.set_rng_state(checkpoint["torch_rng"])
            generator.set_state(checkpoint["loader_rng"])
            if device.type == "cuda" and checkpoint["cuda_rng"]:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        else:
            torch.manual_seed(args.seed + parallel.rank + start_epoch * 100003)
    elif parallel.world_size > 1:
        # Initialize identical weights, then independent measurement augmentations.
        torch.manual_seed(args.seed + parallel.rank)
    if parallel.primary:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (args.output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parallel.barrier()
    validation_total = sum(len(ep["frames"]) for run in manifest["validation"] for ep in run["episodes"])
    val_shards = validation_shards(manifest["validation"], parallel.world_size, args.max_val_frames)
    with TrainingDisplay(parallel.primary) as display:
        display.setup(config, manifest)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train()
            dataset.epoch = epoch
            if sampler:
                sampler.epoch = epoch
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            display.phase(f"训练 {epoch}/{args.epochs}", min(len(loader), args.max_train_batches or len(loader)))
            started = time.perf_counter()
            loss_sum, frame_count, batches = 0.0, 0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                broadcast_buffers(model)
                count = int(parallel.sum([sum(len(sequence) for sequence in batch)])[0])
                sequences, histories = [], []
                recurrent_states = [RecurrentState() for _ in batch]
                for sequence in batch:
                    if args.augmentation == "paper":
                        signs = [1 if torch.rand(()) >= 0.5 else -1 for _ in range(2)] + [1]
                        sequence = mirror_sequence(sequence, signs)
                    sequences.append(sequence)
                    histories.append(
                        torch.as_tensor(
                            sequence[0].get("initial_history_w", np.empty((0, 3))), dtype=torch.float32, device=device
                        ).detach()
                    )
                if recurrent:
                    chunk_loss = None
                    length = max((len(sequence) for sequence in sequences), default=0)
                    for time_index in range(length):
                        active = [i for i, sequence in enumerate(sequences) if time_index < len(sequence)]
                        current = [sequences[i][time_index] for i in active]
                        losses, _, _ = recorded_batch(
                            model, current, [recurrent_states[i] for i in active], config, device
                        )
                        total = torch.stack(losses).sum()
                        if not torch.isfinite(total):
                            raise RuntimeError("Non-finite recurrent loss")
                        chunk_loss = total if chunk_loss is None else chunk_loss + total
                        loss_sum += float(total.detach())
                        frame_count += len(current)
                        if (time_index + 1) % args.bptt_steps == 0 or time_index + 1 == length:
                            (chunk_loss / count).backward()
                            chunk_loss = None
                            for state in recurrent_states:
                                state.detach_hidden()  # preserve values; truncate only the gradient graph
                elif sparse:
                    for time_index in range(max((len(sequence) for sequence in sequences), default=0)):
                        active = [i for i, sequence in enumerate(sequences) if time_index < len(sequence)]
                        current = [sequences[i][time_index] for i in active]
                        losses, predictions = forward_sparse_batch(
                            model,
                            current,
                            [histories[i] for i in active],
                            config,
                            device,
                        )
                        total = torch.stack(losses).sum()
                        if not torch.isfinite(total):
                            raise RuntimeError(f"Non-finite batch loss: {[frame['path'] for frame in current]}")
                        (total / count).backward()
                        for index, prediction in zip(active, predictions, strict=True):
                            histories[index] = prediction
                        loss_sum += float(total.detach())
                        frame_count += len(current)
                else:
                    for sequence, previous in zip(sequences, histories, strict=True):
                        for frame in sequence:
                            loss, previous, _, _ = forward_frame(
                                model, frame, previous, config, device, augment=True, compute_metrics=False
                            )
                            if not torch.isfinite(loss):
                                raise RuntimeError(f"Non-finite loss: {frame['path']}")
                            (loss / count).backward()
                            loss_sum += float(loss.detach())
                            frame_count += 1
                synchronize_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batches += 1
                report_loss, report_frames = parallel.sum([loss_sum, frame_count])
                display.update(
                    batches,
                    f"loss={report_loss / report_frames:.4f} "
                    f"{report_frames / (time.perf_counter() - started):.1f} frames/s",
                )
                if args.max_train_batches and batches >= args.max_train_batches:
                    break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = max(parallel.gather(time.perf_counter() - started))
            loss_sum, frame_count = parallel.sum([loss_sum, frame_count])
            frame_count = int(frame_count)
            gpu_peaks = parallel.gather(torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0)
            broadcast_buffers(model)
            validation_started = time.perf_counter()
            local_total = sum(len(ep["frames"]) for run in val_shards[parallel.rank] for ep in run["episodes"])
            display.phase(f"验证 {epoch}/{args.epochs} · rank 0 分片", local_total)
            local_validation = (
                evaluate(model, val_shards[parallel.rank], config, device, progress=display.update)
                if val_shards[parallel.rank]
                else None
            )
            validation = merge_validation(
                [r for r in parallel.gather(local_validation) if r is not None], validation_total
            )
            validation_seconds = max(parallel.gather(time.perf_counter() - validation_started))
            step_scheduler(scheduler, config, validation)
            improved = validation["prediction"]["f1"] > best_f1
            best_f1 = max(best_f1, validation["prediction"]["f1"])
            record = {
                "epoch": epoch,
                "train_loss": loss_sum / frame_count,
                "train_frames": frame_count,
                "partial_training": batches < len(loader),
                "validation": validation,
                "learning_rate": learning_rate,
                "lr_reduced": scheduler.get_last_lr()[0] < learning_rate,
                "next_lr": scheduler.get_last_lr()[0],
                "world_size": parallel.world_size,
                "train_seconds": train_seconds,
                "train_frames_per_second": frame_count / train_seconds,
                "validation_seconds": validation_seconds,
                "peak_allocated_gib_per_rank": gpu_peaks,
            }
            rank_rng = parallel.gather(
                dict(
                    torch_rng=torch.get_rng_state(),
                    loader_rng=generator.get_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                )
            )
            if not parallel.primary:
                parallel.barrier()
                continue
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            checkpoint = {
                "epoch": epoch,
                "best_f1": best_f1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "manifest": manifest,
                "metrics": record,
                "rank_rng": rank_rng,
                "torch_rng": torch.get_rng_state(),
                "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            checkpoint_save(args.output / "last.pt", checkpoint)
            if improved:
                checkpoint_save(args.output / "best.pt", checkpoint)
            display.epoch(record)
            parallel.barrier()
    parallel.close()


if __name__ == "__main__":
    main()
