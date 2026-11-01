"""LP-ACRL equations (5)--(7), arXiv:2601.17428v1, adapted to vector environments.

Stage duration, minimum samples, temperature and missing-data handling are explicit
engineering choices: these details are not specified in the paper.
"""

import math
from itertools import product

import torch
from isaaclab.utils import configclass


@configclass
class LPACRLCfg:
    temperature: float = 1.0  # beta, in units of undiscounted episode return
    stage_steps: int = 2400  # 100 PPO iterations at 24 steps/iteration
    min_episodes_per_task: int = 2
    adaptive: bool = True  # False: uniform baseline/evaluation
    terrain_names: tuple[str, ...] = ("stairs_up", "stairs_down", "rough", "slope_up", "slope_down")
    terrain_levels: int = 4
    vx_edges: tuple[float, ...] = (0., .5, 1., 1.5, 2., 2.5)
    yaw_edges: tuple[float, ...] = (0., .5, 1., 1.5, 2., 2.5, 3.)
    vy_range: tuple[float, float] = (-.5, .5)
    percentage_error_floor: float = .1  # avoids division by near-zero commands
    seed: int = 42


class LearningProgressSampler:
    """GPU batch aggregation; signed LP; no difficulty ordering or absolute value.

    A task needs enough returns in two consecutive stages before it gets a LP
    estimate. Missing tasks have LP=0, never a fabricated zero return. The first
    stage establishes a baseline and leaves sampling uniform.
    """

    def __init__(self, cfg: LPACRLCfg, device="cpu"):
        if not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if cfg.stage_steps < 1 or cfg.min_episodes_per_task < 1 or cfg.terrain_levels < 1:
            raise ValueError("stage_steps, min_episodes_per_task and terrain_levels must be positive")
        if not cfg.terrain_names or len(set(cfg.terrain_names)) != len(cfg.terrain_names):
            raise ValueError("terrain_names must be nonempty and unique")
        for edges in (cfg.vx_edges, cfg.yaw_edges):
            if len(edges) < 2 or any(not math.isfinite(x) or x < 0 for x in edges):
                raise ValueError("velocity bin edges must be finite and nonnegative")
            if any(b <= a for a, b in zip(edges, edges[1:])) and tuple(edges) != (0., 0.):
                raise ValueError("velocity bins must increase; (0, 0) represents a fixed zero command")
        if len(cfg.vy_range) != 2 or not all(math.isfinite(x) for x in cfg.vy_range) or cfg.vy_range[0] > cfg.vy_range[1]:
            raise ValueError("invalid lateral velocity range")
        if not math.isfinite(cfg.percentage_error_floor) or cfg.percentage_error_floor <= 0:
            raise ValueError("percentage_error_floor must be finite and positive")
        self.cfg = cfg
        self.device = torch.device(device)
        self.table = torch.tensor(list(product(range(len(cfg.terrain_names)), range(cfg.terrain_levels),
                                              range(len(cfg.vx_edges)-1), range(len(cfg.yaw_edges)-1))),
                                  device=device, dtype=torch.long)
        self.num_tasks = len(self.table)
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed)
        self.stage = 0
        self.steps_in_stage = 0
        self.return_sum = torch.zeros(self.num_tasks, device=device, dtype=torch.float64)
        self.count = torch.zeros(self.num_tasks, device=device, dtype=torch.long)
        self.previous_mean = torch.zeros_like(self.return_sum)
        self.previous_valid = torch.zeros_like(self.count, dtype=torch.bool)
        self.learning_progress = torch.zeros_like(self.return_sum)
        self.probabilities = torch.full_like(self.return_sum, 1. / self.num_tasks)
        self.total_count = torch.zeros_like(self.count)
        self.epte_sum = torch.zeros(self.num_tasks, 3, device=device, dtype=torch.float64)
        self.success_sum = torch.zeros_like(self.return_sum)
        self.last_count = torch.zeros_like(self.count)
        self.last_mean = torch.zeros_like(self.return_sum)
        self.last_epte = torch.zeros_like(self.epte_sum)
        self.last_success = torch.zeros_like(self.return_sum)

    def observe(self, task_ids, returns, epte=None, success=None):
        ids = task_ids.to(device=self.device, dtype=torch.long)
        values = returns.to(device=self.device, dtype=torch.float64)
        if ids.shape != values.shape or not torch.isfinite(values).all():
            raise ValueError("episode returns must be finite and match task_ids")
        self.return_sum.scatter_add_(0, ids, values)
        self.count.scatter_add_(0, ids, torch.ones_like(ids))
        self.total_count.scatter_add_(0, ids, torch.ones_like(ids))
        if epte is not None:
            self.epte_sum.index_add_(0, ids, epte.to(self.epte_sum))
        if success is not None:
            self.success_sum.scatter_add_(0, ids, success.to(self.success_sum))

    def advance(self):
        self.steps_in_stage += 1
        if self.steps_in_stage < self.cfg.stage_steps:
            return False
        self.finish_stage()
        return True

    def finish_stage(self):
        # Optional distributed training: all ranks aggregate once per common stage.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in (self.return_sum, self.count, self.epte_sum, self.success_sum):
                torch.distributed.all_reduce(value)
        valid = self.count >= self.cfg.min_episodes_per_task
        mean = self.return_sum / self.count.clamp_min(1)
        comparable = valid & self.previous_valid
        self.learning_progress = torch.where(comparable, mean-self.previous_mean, 0.)
        if self.cfg.adaptive:
            self.probabilities = torch.softmax(self.learning_progress / self.cfg.temperature, dim=0)
        else:
            self.probabilities.fill_(1. / self.num_tasks)
        self.last_count.copy_(self.count)
        self.last_mean.copy_(mean)
        self.last_epte.copy_(self.epte_sum / self.count.clamp_min(1)[:, None])
        self.last_success.copy_(self.success_sum / self.count.clamp_min(1))
        self.previous_mean.copy_(mean)
        self.previous_valid.copy_(valid)
        self.return_sum.zero_()
        self.count.zero_()
        self.epte_sum.zero_()
        self.success_sum.zero_()
        self.stage += 1
        self.steps_in_stage = 0

    def sample(self, count):
        return torch.multinomial(self.probabilities, count, replacement=True, generator=self.generator)

    def commands(self, task_ids):
        cells = self.table[task_ids]
        out = torch.empty(len(task_ids), 3, device=self.device)
        for axis, edges, column in ((0, self.cfg.vx_edges, 2), (2, self.cfg.yaw_edges, 3)):
            bounds = out.new_tensor(edges)
            low, high = bounds[cells[:, column]], bounds[cells[:, column]+1]
            magnitude = low + (high-low)*torch.rand(len(task_ids), device=self.device, generator=self.generator)
            sign = torch.randint(0, 2, (len(task_ids),), device=self.device, generator=self.generator)*2-1
            out[:, axis] = magnitude*sign
        lo, hi = self.cfg.vy_range
        out[:, 1] = lo+(hi-lo)*torch.rand(len(task_ids), device=self.device, generator=self.generator)
        return out

    def state_dict(self):
        return {"version": 1, "cfg": self.cfg.to_dict(), "stage": self.stage,
                "steps_in_stage": self.steps_in_stage, "rng": self.generator.get_state(),
                **{name: value.detach().cpu().clone() for name, value in vars(self).items()
                   if isinstance(value, torch.Tensor)}}

    def load_state_dict(self, state):
        if state["version"] != 1:
            raise ValueError("Unsupported curriculum checkpoint version")
        expected = self.cfg.to_dict()
        # Evaluation may freeze the curriculum; seed is restored from RNG state.
        for key in expected.keys() - {"adaptive", "seed"}:
            if state["cfg"][key] != expected[key]:
                raise ValueError(f"Curriculum checkpoint configuration mismatch: {key}")
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                if state[name].shape != value.shape:
                    raise ValueError(f"Curriculum checkpoint shape mismatch: {name}")
                value.copy_(state[name].to(value))
        self.stage, self.steps_in_stage = state["stage"], state["steps_in_stage"]
        self.generator.set_state(state["rng"].cpu())
        if not self.cfg.adaptive:
            self.probabilities.fill_(1. / self.num_tasks)


def epte_with_stability(error_sum, alive_steps, horizon, failed):
    """Equation (8); percentage errors are accumulated per alive step.

    Non-fall truncations are normalized over the observed duration. The caller
    records falls separately, and uses actual alive steps (not randomized timers).
    """
    duration = alive_steps.clamp_min(1).to(error_sum.dtype)
    remaining = (horizon-duration).clamp_min(0)
    denom = torch.where(failed, torch.full_like(duration, horizon), duration)
    return (error_sum + remaining[:, None]*failed[:, None]) / denom[:, None]
