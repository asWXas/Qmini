"""torchrun data parallelism, synchronized once per autoregressive optimizer step.

Sequences may have different lengths and pruning may leave parameters unused.
Explicit gradient buckets avoid a collective on every frame/backward operation.
"""

import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedContext:
    def __init__(self, requested_device):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}" if self.world_size > 1 else requested_device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.world_size > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=self.device)

    @property
    def primary(self):
        return self.rank == 0

    def sum(self, values):
        result = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(result)
        return result.cpu().tolist()

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        results = [None] * self.world_size
        dist.all_gather_object(results, value)
        return results

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def close(self):
        if self.world_size > 1:
            dist.destroy_process_group()


class ParallelBatchSampler(Sampler):
    """Same number of optimizer steps on all ranks, without duplicate/drop padding.

    A final rank may receive an empty batch. It still joins gradient reduction.
    Every window appears exactly once globally in each epoch.
    """

    def __init__(self, size, batch_size, rank, world_size, seed):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(self.size / (self.batch_size * self.world_size))

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        width = self.batch_size * self.world_size
        for start in range(0, self.size, width):
            offset = start + self.rank * self.batch_size
            yield order[offset : min(offset + self.batch_size, start + width)]


@torch.no_grad()
def synchronize_gradients(model, bucket_bytes=16 * 1024 * 1024):
    """SUM gradients already normalized by GLOBAL frame count; preserve unused=None."""
    if not dist.is_initialized():
        return
    parameters = list(model.parameters())
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    globally_used = used.cpu().tolist()
    bucket, size = [], 0

    def reduce_bucket(items):
        flat = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in items])
        dist.all_reduce(flat)
        offset = 0
        for parameter in items:
            gradient = flat[offset : offset + parameter.numel()].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
            offset += parameter.numel()

    for parameter, active in zip(parameters, globally_used, strict=True):
        if not active:
            continue
        bucket.append(parameter)
        size += parameter.numel() * parameter.element_size()
        if size >= bucket_bytes:
            reduce_bucket(bucket)
            bucket, size = [], 0
    if bucket:
        reduce_bucket(bucket)


@torch.no_grad()
def broadcast_buffers(model):
    """Use rank 0 running BatchNorm statistics, as with conventional DDP buffers."""
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def validation_shards(runs, world_size, max_frames=0):
    """Greedy episode assignment: feedback never crosses rank/episode boundaries."""
    shards = [[] for _ in range(world_size)]
    loads = [0] * world_size
    remaining = max_frames or sum(len(ep["frames"]) for run in runs for ep in run["episodes"])
    episodes = []
    for run in runs:
        for episode in run["episodes"]:
            frames = episode["frames"][:remaining]
            remaining -= len(frames)
            if frames:
                episodes.append((run, dict(episode, frames=frames)))
    for run, episode in sorted(episodes, key=lambda item: len(item[1]["frames"]), reverse=True):
        rank = min(range(world_size), key=loads.__getitem__)
        shards[rank].append(dict(run, episodes=[episode]))
        loads[rank] += len(episode["frames"])
    return shards


def merge_validation(results, available_frames):
    frames = sum(r["frames"] for r in results)
    merged = {
        "frames": frames,
        "available_frames": available_frames,
        "episodes_started": sum(r["episodes_started"] for r in results),
        "partial_evaluation": frames < available_frames,
        "loss": sum(r["loss"] * r["frames"] for r in results) / frames,
    }
    for name in ("prediction", "observed_only", "network_prediction", "retained_observations"):
        present = [r for r in results if name in r]
        if not present:
            continue
        if any(r["frames"] and name not in r for r in results):
            raise ValueError(f"Missing validation metric on nonempty rank: {name}")
        tp, fp, fn = (sum(r[name][key] for r in present) for key in ("tp", "fp", "fn"))
        error = sum((r[name]["matched_centroid_error_cm"] or 0) * r[name]["tp"] for r in present)
        merged[name] = dict(
            precision=tp / max(tp + fp, 1),
            recall=tp / max(tp + fn, 1),
            f1=2 * tp / max(2 * tp + fp + fn, 1),
            matched_centroid_error_cm=error / tp if tp else None,
            tp=tp,
            fp=fp,
            fn=fn,
        )
    return merged
