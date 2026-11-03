"""LP-ACRL environment lifecycle: measure before reset, relocate before spawn."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from Qmini.algo.multi_critic.manager.reward_manager import MultiCriticRewardManager
from ..mdp.lp_curriculum import LearningProgressSampler, epte_with_stability


class LPCurriculumController:
    def __init__(self, env):
        self.env = env
        sampler_cfg = env.cfg.lp_acrl.copy()
        if getattr(env.cfg, "seed", None) is not None:
            sampler_cfg.seed = env.cfg.seed
        self.sampler = LearningProgressSampler(sampler_cfg, env.device)
        self.task_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.commands = torch.zeros(env.num_envs, 3, device=env.device)
        self.returns = torch.zeros(env.num_envs, device=env.device)
        self.alive_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.error_sum = torch.zeros(env.num_envs, 3, device=env.device)
        self.forced_task_ids = None  # finite evaluation can assign one task per environment
        self.in_step = False
        self.last_logs = {}
        self.record_evaluation = False
        self.evaluation_records = []
        self.crossed_tile = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        terrain = env.scene.terrain
        if terrain.terrain_origins is not None:
            generator = env.cfg.scene.terrain.terrain_generator
            if tuple(generator.sub_terrains) != tuple(env.cfg.lp_acrl.terrain_names):
                raise ValueError("LP terrain columns do not match the configured task types")
            if generator.num_rows != env.cfg.lp_acrl.terrain_levels:
                raise ValueError("LP terrain rows do not match terrain_levels")
            if generator.num_cols % len(env.cfg.lp_acrl.terrain_names):
                raise ValueError("LP terrain columns must be a multiple of terrain type count")

    def record_step(self, reward):
        if self.env.cfg.commands.base_velocity.xbox:
            return
        self.returns += reward
        self.alive_steps += 1
        data = self.env.scene["robot"].data
        actual = torch.stack((data.root_lin_vel_b.torch[:, 0], data.root_lin_vel_b.torch[:, 1],
                              data.root_ang_vel_b.torch[:, 2]), dim=1)
        denominator = self.commands.abs().clamp_min(self.env.cfg.lp_acrl.percentage_error_floor)
        # Error >100% remains >100%; the paper does not specify clipping.
        self.error_sum += (actual-self.commands).abs()/denominator
        terrain = self.env.scene.terrain
        if terrain.terrain_origins is not None:
            half_size = actual.new_tensor(self.env.cfg.scene.terrain.terrain_generator.size) / 2
            offset = (data.root_pos_w.torch[:, :2] - terrain.env_origins[:, :2]).abs()
            self.crossed_tile |= (offset >= half_size).any(dim=1)

    def reset(self, env_ids):
        ids = torch.arange(self.env.num_envs, device=self.env.device)[env_ids]
        # Manual reset / initial random episode timers are censored trajectories.
        complete = (self.alive_steps[ids] > 0) & (self.alive_steps[ids] == self.env.episode_length_buf[ids])
        done = ids[complete] if self.in_step else ids[:0]
        if len(done):
            failed = self.env.termination_manager.terminated[done]
            epte = epte_with_stability(self.error_sum[done], self.alive_steps[done],
                                       self.env.max_episode_length, failed)
            success = (self.alive_steps[done] > .9*self.env.max_episode_length) & (epte[:, 0] < .3) & (epte[:, 2] < .3)
            self.sampler.observe(self.task_ids[done], self.returns[done], epte, success)
            if self.record_evaluation:
                for i, slot in enumerate(done.tolist()):
                    self.evaluation_records.append(dict(
                        env_id=slot, task_id=self.task_ids[slot].item(),
                        alive_steps=self.alive_steps[slot].item(), epte=epte[i].tolist(),
                        crossed_tile=self.crossed_tile[slot].item(),
                        **{"return": self.returns[slot].item()},
                    ))
        self.assign(ids)
        self.returns[ids] = 0
        self.alive_steps[ids] = 0
        self.error_sum[ids] = 0
        self.crossed_tile[ids] = False

    def assign(self, ids):
        if self.forced_task_ids is None:
            tasks = self.sampler.sample(len(ids))
        else:
            tasks = self.forced_task_ids[ids]
        self.task_ids[ids] = tasks
        self.commands[ids] = self.sampler.commands(tasks)
        terrain = self.env.scene.terrain
        if terrain.terrain_origins is not None:
            cells = self.sampler.table[tasks]
            types, levels = cells[:, 0], cells[:, 1]
            repeats = terrain.terrain_origins.shape[1] // len(self.sampler.cfg.terrain_names)
            columns = types + len(self.sampler.cfg.terrain_names)*torch.randint(
                repeats, (len(ids),), device=self.env.device, generator=self.sampler.generator)
            terrain.terrain_types[ids] = columns
            terrain.terrain_levels[ids] = levels
            terrain.env_origins[ids] = terrain.terrain_origins[levels, columns]

    def advance(self):
        if self.env.cfg.commands.base_velocity.xbox:
            return
        if self.sampler.advance():
            s = self.sampler
            visited = s.last_count > 0
            self.last_logs = {
                "LP_ACRL/stage": float(s.stage),
                "LP_ACRL/task_coverage": visited.float().mean().item(),
                "LP_ACRL/entropy": -(s.probabilities*s.probabilities.clamp_min(1.e-300).log()).sum().item(),
                "LP_ACRL/max_probability": s.probabilities.max().item(),
                "LP_ACRL/mean_signed_progress": s.learning_progress.mean().item(),
            }
            if visited.any():
                self.last_logs.update({
                    "LP_ACRL/return": s.last_mean[visited].mean().item(),
                    "LP_ACRL/success_fraction_observed_tasks": s.last_success[visited].mean().item(),
                    "LP_ACRL/epte_x": s.last_epte[visited, 0].mean().item(),
                    "LP_ACRL/epte_yaw": s.last_epte[visited, 2].mean().item(),
                })
            for i, name in enumerate(s.cfg.terrain_names):
                self.last_logs[f"LP_ACRL/probability_{name}"] = s.probabilities[s.table[:, 0] == i].sum().item()


class _RecordLPReturn:
    def compute(self, dt):
        reward = super().compute(dt)
        self._env.lp_curriculum.record_step(reward)
        return reward


class LPRewardManager(_RecordLPReturn, RewardManager):
    pass


class LPMultiCriticRewardManager(_RecordLPReturn, MultiCriticRewardManager):
    pass


class LPACRLEnv(ManagerBasedRLEnv):
    reward_manager_type = LPRewardManager

    def load_managers(self):
        self.command_manager = CommandManager(self.cfg.commands, self)
        ManagerBasedEnv.load_managers(self)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = self.reward_manager_type(self.cfg.rewards, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        self.lp_curriculum = LPCurriculumController(self)
        self._configure_gym_env_spaces()
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def _reset_idx(self, env_ids):
        self.lp_curriculum.reset(env_ids)  # assigns terrain origins before reset events
        super()._reset_idx(env_ids)

    def step(self, action):
        self.lp_curriculum.in_step = True
        try:
            result = super().step(action)
        finally:
            self.lp_curriculum.in_step = False
        self.lp_curriculum.advance()
        if self.lp_curriculum.last_logs:
            result[-1].setdefault("log", {}).update(self.lp_curriculum.last_logs)
        return result


class LPACRLMultiCriticEnv(LPACRLEnv):
    reward_manager_type = LPMultiCriticRewardManager

    def step(self, action):
        result = super().step(action)
        result[-1]["group_rewards"] = self.reward_manager.group_rewards
        result[-1]["reward_group_names"] = self.reward_manager.reward_group_names
        return result
